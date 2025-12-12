
struct Parament
    judge::String
    L::Int
    Lz::Int
    zchunk::Int
    ychunk::Int
    xchunk::Int
    Dimension::Int
    chi::Int
    Nblck::Int
    Ndisorder::Int
    Nsample::Int
    Ntherm::Int
    Nreplic::Int
    Nstep::Int
    seed::Int
    P::Float64
    Beta_u::Float64
    Beta_d::Float64
    nBeta::Int
    dn :: Float64

    Vol::Int
    nnb::Int
    field::Array{Float64, 1}
    Antifield::Array{Float64, 1}
    BondTensor::Array{Any, 1}
    
    # Constructor for Parament
    function Parament(judge, Beta_u, Beta_d, P, nBeta, L, Lz, zchunk, ychunk, xchunk, Dimension, chi, Ndisorder, Nsample, Ntherm, Nreplic, seed)
        Vol = L^Dimension
        Nblck = Ndisorder
        nnb = Dimension * 2
        Nstep = 2
        dn = Dimension*(L-1)/L
        field = vec([1.0 -1.0])
        Antifield = vec([-1.0 1.0])
        Barray = [1.0 -1.0; -1.0 1.0]
        BAntiarray = [-1.0 1.0; 1.0 -1.0]
        I2 = [0.0 -Inf; -Inf 0.0]
        BondTensor = [[0.0 0.0], copy(Barray), copy(BAntiarray), I2]
        new(judge, L, Lz, zchunk, ychunk, xchunk, Dimension, chi, Nblck,Ndisorder, Nsample, Ntherm, Nreplic, Nstep, seed, P, Beta_u, Beta_d, nBeta, dn, Vol, nnb, field, Antifield, BondTensor
        )
    end
end
export Parament
