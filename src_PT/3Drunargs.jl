include("SpinGlass.jl")
using .SpinGlass
using Random
using FileIO



function parse_input()
    # Initialize variables with default values
    judge = ""
    Lz = 0
    L = 0
    Dimension = 0
    D = 0
    Beta_u = 0.0
    Beta_d = 0.0
    nBeta = 0
    P = 0.0
    chi = 0
    Ndisorder = 0
    Nsample = 0
    Ntherm = 0
    Nreplic = 0
    seed = 0
    zchunk = 0
    ychunk = 0
    xchunk = 0

    # Parse the command-line arguments
    for i in 1:2:length(ARGS)
        key = ARGS[i]
        value = ARGS[i + 1]
        if key == "--Type"
            judge = value
        elseif key == "--zchunk"
            zchunk = parse(Int, value)
        elseif key == "--ychunk"
            ychunk = parse(Int, value)
        elseif key == "--xchunk"
            xchunk = parse(Int, value)
        elseif key == "--P"
            P = parse(Float64, value)
        elseif key == "--Lz"
            Lz = parse(Int, value)
        elseif key == "--L"
            L = parse(Int, value)
        elseif key == "--Dimension"
            Dimension = parse(Int, value)
        elseif key == "--Beta_u"
            Beta_u = parse(Float64, value)
        elseif key == "--Beta_d"
            Beta_d = parse(Float64, value)
        elseif key == "--nBeta"
            nBeta = parse(Int, value)
        elseif key == "--chi"
            chi = parse(Int, value)
        elseif key == "--Ndisorder"
            Ndisorder = parse(Int, value)
        elseif key == "--Nsample"
            Nsample = parse(Int, value)
        elseif key == "--Ntherm"
            Ntherm = parse(Int, value)
        elseif key == "--Nreplic"
            Nreplic = parse(Int, value)
        elseif key == "--seed"
            seed = parse(Int, value)
        end
    end
    return judge, [Beta_u, Beta_d, P], [nBeta, L, Lz,zchunk, ychunk, xchunk, Dimension, chi, Ndisorder, Nsample, Ntherm, Nreplic, seed]
end


T = Array{Float64}(undef, 3)
parsed_params = Array{Int64}(undef, 13)
judge, T, parsed_params = parse_input()
pa=Parament(judge,T...,parsed_params...)
sp=Spin(pa)

st=Statistics(pa)
Betalist = range(pa.Beta_u, stop=pa.Beta_d, length=pa.nBeta) |> collect
da=DataFile(pa, Betalist)
Random.seed!(parsed_params[end])


if pa.chi != 0
    SimulationTNMH(pa,sp,st,da)
else
    SimulationMetro(pa,sp,st,da)
end

"""
julia 3Drunargs.jl --Type FBC --Dimension 3 --Lz 4 --L 4 --zchunk 1 --ychunk 4 --xchunk 4 --P 0.5 --Beta_u 1.1 --Beta_d 1.4 --nBeta 51 --chi 2 --Ndisorder 1 --Nsample 10 --Ntherm 10 --Nreplic 2 --seed 12
"""
