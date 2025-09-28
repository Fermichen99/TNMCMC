
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
    verbose::Int
    P::Float64
    T::Float64
    Beta::Float64
    dn :: Float64

    Vol::Int
    nnb::Int
    Barray::Array{Float64, 2}
    BAntiarray::Array{Float64, 2}
    field::Array{Float64, 1}
    Antifield::Array{Float64, 1}
    I1::Array{Float64, 1}
    I2::Array{Float64, 2}
    I3::Array{Float64, 3}
    I4::Array{Float64, 4}
    I5::Array{Float64, 5}
    I6::Array{Float64, 6}
    FieldTensor::Array{Any, 1}
    BondTensor::Array{Any, 1}
    
    # Constructor for Parament
    function Parament(judge, Beta, P, L, Lz, zchunk, ychunk, xchunk, Dimension, chi, Ndisorder, Nsample, Ntherm, Nstep, Nreplic, seed, verbose)
        Vol = L^Dimension
        # Ntherm = div(round(Int, Nsample), 5)
        Nblck = Ndisorder
        nnb = Dimension * 2
        T = 1.0 / Beta
        dn = Dimension*(L-1)/L
        Barray = exp.(Beta * [1.0 -1.0; -1.0 1.0])
        BAntiarray = exp.(-Beta * [1.0 -1.0; -1.0 1.0])
        field = vec(exp.(Beta * [1.0 -1.0]))
        Antifield = vec(exp.(-Beta * [1.0 -1.0]))
        I1 = vec([1.0 1.0])
        I2 = zeros(2, 2)
        I3 = zeros(2, 2, 2)
        I4 = zeros(2, 2, 2, 2)
        I5 = zeros(2, 2, 2, 2, 2)
        I6 = zeros(2, 2, 2, 2, 2, 2)
        for i in 1:2
            I2[i ,i] = 1.0
            I3[i, i, i] = 1.0
            I4[i, i, i, i] = 1.0
            I5[i, i, i, i, i] = 1.0            
            I6[i, i, i, i, i, i] = 1.0
        end
        FieldTensor = [I1, copy(field), copy(Antifield)]
        BondTensor = [[1.0 1.0], copy(Barray), copy(BAntiarray), exp.(Beta * [1.0 -1.0]), exp.(-Beta * [1.0 -1.0]), I2]
        new(judge, L, Lz, zchunk, ychunk, xchunk, Dimension, chi, Nblck,Ndisorder, Nsample, Ntherm, Nreplic, Nstep, seed, verbose, P, T, Beta, dn, Vol, nnb, 
            Barray, BAntiarray, field, Antifield, I1, I2, I3, I4, I5, I6,
            FieldTensor, BondTensor)
    end
end
export Parament
