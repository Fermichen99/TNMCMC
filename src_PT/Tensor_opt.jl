using LinearAlgebra
# using TensorOperations
using OMEinsum
using Random
import Base.GC: gc
export InitTensor, TNMH


struct MpsState
    Tensor::Array{Any}  # Allow for different array types
    lnZ::Array{Float64, 1}
end

mutable struct Tensor
    HorizontTensorsDw::MpsState 
    VerticalTensorsLe::MpsState
    VerticalTensorsRi::MpsState
    
    zlen::Int
    ylen::Int
    xlen::Int
    chi::Int
    spin_in_chunk::Array{Int,2}


    function Tensor(chi::Int, zlen::Int, ylen::Int, xlen::Int) 
        spin_Dim = 1<<zlen
        spin_in_chunk = Array{Int,2}(undef, spin_Dim, zlen)
        for d_idx_zero_based in 0:(spin_Dim-1)
            spin_in_chunk[d_idx_zero_based+1, :]=get_spin_config_from_index(d_idx_zero_based, zlen)
        end
        HorizontTensorsDw = MpsState(Array{Any,2}(undef, ylen, xlen), zeros(Float64, ylen))
        VerticalTensorsLe = MpsState(Array{Any,1}(undef, xlen), zeros(Float64, xlen))
        VerticalTensorsRi = MpsState(Array{Any,1}(undef, xlen), zeros(Float64, xlen))
        new(HorizontTensorsDw, VerticalTensorsLe, VerticalTensorsRi, zlen, ylen, xlen, chi, spin_in_chunk)
    end
end

export Tensor



"""
Maps a superspin index (0 to D-1) to a configuration of pa.zchunk spins.
Spin +1 for bit 0, -1 for bit 1.
spins_vector[1] corresponds to the LSB (first spin in the chunk).
"""
function get_spin_config_from_index(d_idx::Int, num_spins::Int)::Vector{Int}
    spins_vector = Vector{Int}(undef, num_spins)
    for k in 1:num_spins
        # (d_idx >> (k-1)) & 1 extracts the (k-1)-th bit (0-indexed)
        spins_vector[k] = ((d_idx >> (num_spins-k)) & 1) == 0 ? 1 : -1
    end
    return spins_vector # spins_vector[1] is s_1, ..., spins_vector[num_spins] is s_N
end

"""
Gets neighbor coordinates for Z-direction, respecting Open Boundary Conditions.
Returns nothing if the neighbor is out of pa.Lz bounds.
(Original Get_NNB used mod, this one is for specific Z field calculation)
"""
function Get_NNB(pa::Parament, sp::Spin, laz::Int, lay::Int, lax::Int, nnb_idx::Int)
    tz = mod(laz + sp.Dz[nnb_idx] - 1, pa.Lz) + 1
    ty = mod(lay + sp.Dy[nnb_idx] - 1, pa.L) + 1
    tx = mod(lax + sp.Dx[nnb_idx] - 1, pa.L) + 1
    return (tz, ty, tx)
end

"""
Returns the anti-parallel bond index for a given nnb_idx.
"""
function GetAntiPara(pa::Parament, sp::Spin, nnb::Int, laz::Int, lay::Int, lax::Int)
    if sp.bond[laz, lay, lax, nnb] == 0 
        return 1
    else
        if nnb in [3,5]
            if sp.bond[laz, lay, lax, nnb] == 1
                return 2
            else
                return 3
            end
        else
            return 4
        end
    end
end

##############  缓存区  #################################################
"""
    COPY_TENSOR_CACHE[key]  :  key = (h0_sign::Int8,
                                      h1_sign::Int8,
                                      Tuple{Vararg{Int8}}(J_signs))

    SUPERSPIN_BOND_CACHE[key] : key = (dir,             # nnb 方向 2/3/5 …
                                       Tuple{Vararg{Int8}}(bondtype_vec))
"""
# const NTHR = Threads.nthreads()

# const COPY_TENSOR_CACHE_THR =
#       [Dict{Tuple{Int8,Int8,Vararg{Int8}}, Array{Float64,4}}()
#        for _ in 1:NTHR]

# const SUPERSPIN_BOND_CACHE_THR =
#       [Dict{Tuple{Int8,Vararg{Int8}}, Matrix{Float64}}()
#        for _ in 1:NTHR]

# # ② 简单的线程安全 get!(…) 封装
# @inline function _tl_get!(builder::Function, cache::Dict, key)
#     if haskey(cache, key)
#         return cache[key]
#     else
#         val = builder()
#         cache[key] = val
#         return val
#     end
# end

# 预编译一次 einsum
const einsum_copy_bond = ein"ijkl,ai,bj,ck->abcl"   # 4 + 3*2 legs ⇒ 4 legs out


###############################################################################
# --- copy-tensor （向量化 + thread-local） -----------------------------------
###############################################################################
const _EMPTY_F64 = Vector{Float64}()

@inline function _copy_tensor_cached(pa::Parament, te::Tensor,
                                     h0_sign::Int8, h1_sign::Int8, beta::Float64,
                                     Js::NTuple{N,Int8}) where {N}

    # key    = (h0_sign, h1_sign, Js...)
    # cache  = COPY_TENSOR_CACHE_THR[threadid()]

    # return _tl_get!(cache, key) do
        β      = beta
        zchunk = pa.zchunk
        D      = 1 << zchunk
        S = te.spin_in_chunk                # D × zchunk

        # 向量化能量
        E = β .* (Float64(h0_sign).*S[:,1] .+
                  Float64(h1_sign).*S[:,zchunk])

        if zchunk > 1
            Js_f = N==0 ? _EMPTY_F64 : Float64.(collect(Js))
            pair = @views S[:,1:zchunk-1] .* S[:,2:zchunk]
            E   .+= β .* (pair * Js_f)
        end

        diag = exp.(E)
        C = zeros(Float64, D, D, D, D)
        @inbounds @simd for i in 1:D
            C[i,i,i,i] = diag[i]
        end
        C
    # end
end


###############################################################################
# --- superspin bond  ---------------------------------------------------------
###############################################################################
@inline function _superspin_bond_cached(pa::Parament, sp::Spin,
                                        zin::Int, lay::Int, lax::Int,
                                        nnb_dir::Int, beta::Float64)

    bond_idx_vec = ntuple(k -> Int8(GetAntiPara(pa, sp, nnb_dir,
                                                zin+k-1, lay, lax)),
                          pa.zchunk)
    key    = (Int8(nnb_dir), bond_idx_vec...)
    # cache  = SUPERSPIN_BOND_CACHE_THR[threadid()]

    # return _tl_get!(cache, key) do
        m = exp.(beta .* pa.BondTensor[bond_idx_vec[1]])
        for k in 2:pa.zchunk
            m = kron(m, exp.(beta .* pa.BondTensor[bond_idx_vec[k]]))
        end
        m
    # end
end
###############################################################################

#==============================================================================#
function OutTensorGrid(pa::Parament, sp::Spin, te::Tensor,
                       spin_full_system::Array{Int,3},
                       zin::Int, yin::Int, xin::Int, beta::Float64)

    StoreTensor = Array{Any,2}(undef, pa.ychunk, pa.xchunk)
    nnb_list = (2,5,3)      # ← 这里保持与旧代码一致

    D = 1 << pa.zchunk

    @inbounds for tey in 1:pa.ychunk, tex in 1:pa.xchunk
        lax = tex + xin - 1
        lay = tey + yin - 1

        #--------------- 先得到 (h0,h1,Jₖ) 的 ±1 符号 ------------------#
        h0s::Int8 = 0
        if pa.zchunk > 0
            s_below = spin_full_system[Get_NNB(pa, sp, zin, lay, lax, 1)...]
            h0s = Int8(sp.bond[zin, lay, lax, 1] * s_below)
        end

        h1s::Int8 = 0
        if pa.zchunk > 0
            s_above = spin_full_system[Get_NNB(pa, sp, zin+pa.zchunk-1, lay, lax, 4)...]
            h1s = Int8(sp.bond[zin+pa.zchunk-1, lay, lax, 4] * s_above)
        end

        Js = ntuple(k -> Int8(sp.bond[zin+k-1, lay, lax, 4]), pa.zchunk>1 ? pa.zchunk-1 : 0)

        #------------  从 cache 里直接拿 copy-tensor --------------------#
        copy_tensor = _copy_tensor_cached(pa, te, h0s, h1s, beta, Js)

        #------------  三个横向 bond 矩阵 (亦可缓存)  -------------------#
        bonds = ntuple(i -> _superspin_bond_cached(pa, sp, zin, lay, lax, nnb_list[i], beta), 3)

        #------------  做一次张量缩并  -------------------------------#
        StoreTensor[tey, tex] = einsum_copy_bond(copy_tensor, bonds[1], bonds[2], bonds[3])
    end


    # 旧代码里对最后一行做 reshape，这里保持一致
    ylast = pa.ychunk
    if ylast > 0
        @inbounds for x in 1:pa.xchunk
            t = StoreTensor[ylast, x]
            nd = ndims(t)
            if nd == 4
                StoreTensor[ylast, x] = reshape(t, size(t,1), size(t,2), size(t,4))
            end
        end
    end
    return StoreTensor
end




function StorelnZ(pa::Parament,te::Tensor,StoreTensor)
    lnZ=0
    res, te.HorizontTensorsDw.Tensor[pa.L,:] = Compress(StoreTensor[pa.L,:], pa.chi)
    lnZ += res
    te.HorizontTensorsDw.lnZ[pa.L] = lnZ
    for y in pa.L:-1:2
        res,te.HorizontTensorsDw.Tensor[y-1,:] = Compress(Eat(pa,te.HorizontTensorsDw.Tensor[y,:] ,StoreTensor[y-1,:]),pa.chi)
        lnZ += res
        te.HorizontTensorsDw.lnZ[y-1] = lnZ
    end
end


function Eat(pa::Parament, mps::Array{T,S}, mpo::Array{T,O}) where {T,S,O}
    result = similar(mps)  # 预分配同类型内存
    physical_dim = size(mps[1],3)
    for i in 1:pa.L
        tmp = ein"ikj,abjc->aibkc"(mps[i], mpo[i])
        result[i] = reshape(tmp, size(mps[i],1)*size(mpo[i],1), :, physical_dim)
    end
    return result
end

function Compress(mps::Array{T,S},chi::Int) where {T,S}
    residual=0
    len=length(mps); physical_dim=size(mps[1],3)
    for i in 1:len 
        mps[i]=permutedims(mps[i],(1,3,2))
    end
    for i in 1:len-1
        F=qr(reshape(mps[i],size(mps[i],1)*physical_dim,:))
        Q=Matrix(F.Q)
        mps[i] = reshape(Q,size(mps[i],1),physical_dim,:)
        mps[i+1] = reshape(ein"ij,jab->iab"(F.R,mps[i+1]), size(F.R,1), physical_dim, size(mps[i+1],3))  
        if mod(i,20) == 0
            tnorm = norm(mps[i+1])
            mps[i+1] /= tnorm
            residual += log(tnorm)
        end
    end   
    for i in len:-1:2
        A = reshape(ein"ijk,kab->ijab"(mps[i-1],mps[i]), size(mps[i-1],1)*physical_dim,size(mps[i],3)*physical_dim)
        F=svd(A)  
        if size(F.S,1)>chi
            Vt=F.Vt[1:chi,:]; U=F.U[:,1:chi]
            mps[i]=reshape(Vt, : , physical_dim, size(mps[i],3))
            mps[i-1]=reshape(U*Diagonal(F.S[1:chi]), size(mps[i-1],1),physical_dim,:)
        else 
            mps[i]=reshape(F.Vt, :, physical_dim, size(mps[i],3)) 
            mps[i-1]=reshape(F.U*Diagonal(F.S), size(mps[i-1],1),physical_dim,:)            
        end   
    end       
    tnorm=norm(mps[1])
    mps[1] /= tnorm
    residual += log(tnorm)
    for i in 1:len
        mps[i]=permutedims(mps[i],(1,3,2))
    end
    return residual,mps
end


function InitTensor(zin::Int, yin::Int, xin::Int,pa::Parament,sp::Spin, spin::Array{Int,3},te::Tensor, beta::Float64)
    StoreTensor = OutTensorGrid(pa, sp, te, spin,zin,yin,xin,beta)
    StorelnZ(pa, te, StoreTensor)
end

function Contruction(pa::Parament, te::Tensor, J::Array{Int}, mps::Array{T,S}, Beta::Float64) where {T,S}
    lnZ = 0
    len = length(mps)
    tnorm = norm(mps[len])
    mps[pa.L] /= tnorm
    te.VerticalTensorsRi.Tensor[len] = mps[len]
    lnZ += log(tnorm)
    te.VerticalTensorsRi.lnZ[len] = lnZ
    te.VerticalTensorsLe.lnZ[1] = 0
    for i in len:-1:2
        mps[i-1] = ein"ijk,k,lin->ljn"(mps[i],Out_superspin_Field(J[:,i],pa,Beta),mps[i-1])
        tnorm = norm(mps[i-1])
        mps[i-1] /= tnorm
        te.VerticalTensorsRi.Tensor[i-1] = mps[i-1]
        lnZ += log(tnorm)
        te.VerticalTensorsRi.lnZ[i-1] = lnZ
    end
end

function Out_superspin_Field(ju::Array{Int,1},pa::Parament,beta::Float64)
    field_dim = length(ju)
    if pa.zchunk == 0
        return Matrix{Float64}(I, 1, 1) # Should not happen if zchunk >= 1
    elseif field_dim == 1 # No Kronecker product needed, just get the 2x2 matrix
        return ju[1] == 1 ? exp.(beta .* pa.field) :  ju[1] == -1 ? exp.(beta .* pa.Antifield) : ones(2)
    end

    field_matrices = Vector{Any}(undef, field_dim)
    for i in 1:field_dim
        field_matrices[i] = ju[i] == 1 ? exp.(beta * pa.field) :  ju[i] == -1 ? exp.(beta * pa.Antifield) : ones(2)
    end

    superspin_field_m = field_matrices[1]
    for i in 2:field_dim
        superspin_field_m = kron(superspin_field_m, field_matrices[i])
    end
    return superspin_field_m
end 

function OutConditionalZ(x::Int,te::Tensor)
    if x == 1
        return log.(abs.(te.VerticalTensorsRi.Tensor[1][:]))
    else
        tensor =  ein"ij,jkl->ikl"(te.VerticalTensorsLe.Tensor[1], te.VerticalTensorsRi.Tensor[x])
        return log.(abs.(tensor[:]))
    end
end


function GetlnZ(te::Tensor, J::Array{Int,1})
    lnZ = zeros(Float64, 1<<te.zlen)
    for d_idx_zero_based in 0:(1<<te.zlen-1) # Iterate through all D states of the superspin
        lnZ[d_idx_zero_based+1] = sum(te.spin_in_chunk[d_idx_zero_based+1, :] .* J)
    end
    return lnZ
end


function ChangeVerticalTensorsLe(x::Int,y::Int,s::Int,pa::Parament,te::Tensor) 
    vecs = zeros(Int, 1<<pa.zchunk)
    vecs[s] = 1
    C = ein"k,ijk->ij"(vecs,te.HorizontTensorsDw.Tensor[y,x])
    tnorm = norm(C)
    C /= tnorm; lnZ = log(tnorm)
    if x == 1
        te.VerticalTensorsLe.Tensor[1] = C
        te.VerticalTensorsLe.lnZ[1] = lnZ
    else
        te.VerticalTensorsLe.Tensor[1] *= C
        tnorm = norm(te.VerticalTensorsLe.Tensor[1])
        te.VerticalTensorsLe.Tensor[1] /= tnorm
        lnZ += log(tnorm)
        te.VerticalTensorsLe.lnZ[1] += lnZ
    end
end

"""
TNMH(pa::Parament, sp::Spin, te::Tensor)

This function performs the TNMH algorithm for a 2D RBIM system.

# Arguments
- `pa::Parament`: The parameters of the RBIM system.
- `sp::Spin`: The spin configuration of the RBIM system.
- `te::Tensor`: The tensor representation of the RBIM system.

# Returns
- `spin`: The updated spin configuration after the TNMH algorithm.
- `lnpr`: The logarithm of the acceptance probability.
- `lnZU[1]`: The logarithm of the partition function.

"""
function TNMH_oe(pa::Parament, sp::Spin, te::Tensor,spin::Array{Int,3},zin::Int,yin::Int,xin::Int,beta::Float64)
    lnZpast = 0; dim = 1<<pa.zchunk
    lnpr = 0; lnZq = zeros(dim); lnZU = zeros(dim)
    J = zeros(Int, pa.zchunk, pa.xchunk)

    for tey in 1:te.ylen
        y = yin+tey-1
        J = spin[zin:zin+pa.zchunk-1,max(y-1,1),:].*sp.bond[zin:zin+pa.zchunk-1,y,:,6]
        Contruction(pa,te,J,te.HorizontTensorsDw.Tensor[y,:],beta)
        for tex in 1:te.xlen
            x = xin+tex-1
            ZU = GetlnZ(te, J[:,tex]); lnZU .+= ZU
            lnZq = OutConditionalZ(tex,te)
            lnZq .+= lnZU*beta
            lnZq .+=  te.HorizontTensorsDw.lnZ[y] + te.VerticalTensorsRi.lnZ[x] + te.VerticalTensorsLe.lnZ[1]
            pi = [1.0/(sum(exp.(lnZq[:].-lnZq[i]))) for i in 1:1<<pa.zchunk]
    
    
            # nz = lnZq[1]+log(1+sum(exp(lnZq[q]-lnZq[1]) for q in 2:1<<pa.zchunk) )-lnZpast
            # if nz > 1e-4
            #     println(pi,' ',lnZq[:],' ',lnZpast,' ',nz,' ',tey,' ',tex)
            # end
            ####--
            s = searchsortedfirst(cumsum(pi), rand())
            spin[zin:zin+pa.zchunk-1,y,x] = te.spin_in_chunk[s,:]
            ChangeVerticalTensorsLe(tex,y,s,pa,te)
            lnZU .-= ZU; lnZU .+= ZU[s]
            lnZpast = lnZq[s]
            lnpr += log(pi[s])               
        end
        for x in 1:pa.L
            for laz in 1:pa.zchunk
                if laz == 1 nnblist = [1,4,5] else nnblist = [4,5] end
                for nnb in nnblist
                    spin_index = Get_NNB(pa,sp,zin+laz-1,y,x,nnb)
                    lnZU .+= spin[zin+laz-1,y,x] * spin[spin_index...] *  sp.bond[zin+laz-1,y,x,nnb] 
                end
            end
        end
        # @show y,lnpr,lnZU[1],typeof(lnpr)
    end
    return spin,lnpr,lnZU[1]
end

function Accept(pa::Parament,sp::Spin,st::Statistics, da::DataFile, spinold::Array{Int,3},spinnew::Array{Int,3},lnprold::Float64,lnprnew::Float64,Energyold::Float64,Energynew::Float64,beta::Float64,prt::Bool=false)
    deltaE = beta*(Energynew-Energyold)
    spinnext = zeros(Int, pa.Lz, pa.L, pa.L)
    pi = exp(deltaE+lnprold-lnprnew)
    # @show deltaE, lnprold-lnprnew, pi
    # error()

    st.Acc[2] += 1.0 
    if pi>=1
        spinnext .= spinnew
        prnext= lnprnew
        Ennext= Energynew
        st.Acc[1] += 1.0 
        if prt
            print(da.temdoc["Acc"],  @sprintf("%8i",1), " ")
        end
    else
        if rand()<pi
            spinnext .= spinnew
            prnext = lnprnew
            Ennext= Energynew
            st.Acc[1] += 1.0 
            if prt
                print(da.temdoc["Acc"],  @sprintf("%8i",1), " ")
            end
        else
            spinnext .= spinold
            prnext= lnprold
            Ennext= Energyold
            if prt
                print(da.temdoc["Acc"],  @sprintf("%8i",0), " ")
            end
        end
    end
    return spinnext,prnext,Ennext
end 



function select_zeros!(zin::Int, yin::Int,xin::Int,zlen::Int,ylen::Int,xlen::Int,ite::Int,ibeta::Int,spin::Array{Int,5})
    spin[ibeta,zin:zin+zlen-1,yin:yin+ylen-1,xin:xin+xlen-1,ite] .= 0
end

function flip_model!(sp::Spin, spin::Array{Int,5})
    new_spin = similar(spin)
    new_bond = similar(sp.bond)
    # the index here need to be flip inverse
    for z in 1:size(spin, 2)
        for y in 1:size(spin, 3)
            for x in 1:size(spin, 4)
                new_spin[:, z, y, x, :] .= spin[:, x, z, y, :]  
                new_bond[z, y, x, :] .= sp.bond[x, z, y, :]
            end
        end
    end
    spin .= new_spin
    indices = [(1, 6), (2, 1), (3, 5), (4, 3), (5, 4), (6, 2)]
    for (i, j) in indices
        sp.bond[:, :, :, i] .= new_bond[:, :, :, j]
    end
end

