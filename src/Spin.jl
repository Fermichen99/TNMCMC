
mutable struct Spin
    Dx::Vector{Int}
    Dy::Vector{Int}
    Dz::Vector{Int}
    D2x::Vector{Int}
    D2y::Vector{Int}
    spin::Array{Int, 4}
    bond::Array{Int, 4}

    function Spin(pa::Parament)
        Dx = [0, -1, 0, 0, 1,  0]
        Dy = [0,  0, 1, 0, 0, -1]
        Dz = [-1, 0, 0, 1, 0,  0]
        D2x = [ -1, 0, 1,  0]
        D2y = [  0, 1, 0, -1]
        spin = zeros(Int, 2, pa.Lz, pa.L, pa.L)
        bond = zeros(Int, pa.Lz, pa.L, pa.L, 6)
        new(Dx, Dy, Dz, D2x, D2y, spin, bond)
    end
end
export Spin

function InitBondFBC(pa::Parament,sp::Spin)
    for z in 1:pa.Lz, y in 1:pa.L, x in 1:pa.L
        for k in 1:pa.nnb÷2
            jx,jy,jz=x+sp.Dx[k],y+sp.Dy[k],z+sp.Dz[k]
            if 0<jx<=pa.L && 0<jy<=pa.L && 0<jz<=pa.Lz
                if rand() < pa.P
                    bond = -1
                else 
                    bond = 1
                end
                sp.bond[z,y,x,k] = bond
                sp.bond[jz,jy,jx,k+3] = bond
            end
        end
    end
end

function InitSpin(pa::Parament,sp::Spin)
    for re in 1:2, z in 1:pa.Lz, y in 1:pa.L, x in 1:pa.L
        sp.spin[re,z,y,x] = rand([1,-1])
    end
end

        
export InitBondFBC