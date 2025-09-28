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
    Beta = 0.0
    P = 0.0
    chi = 0
    Ndisorder = 0
    Nsample = 0
    Ntherm = 0
    step = 0
    Nreplic = 0
    seed = 0
    zchunk = 0
    ychunk = 0
    xchunk = 0
    verbose = 0


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
        elseif key == "--Beta"
            Beta = parse(Float64, value)
        elseif key == "--chi"
            chi = parse(Int, value)
        elseif key == "--Ndisorder"
            Ndisorder = parse(Int, value)
        elseif key == "--Nsample"
            Nsample = parse(Int, value)
        elseif key == "--Ntherm"
            Ntherm = parse(Int, value)
        elseif key == "--step"
            step = parse(Int, value)
        elseif key == "--Nreplic"
            Nreplic = parse(Int, value)
        elseif key == "--seed"
            seed = parse(Int, value)
        elseif key == "--verbose"
            verbose = parse(Int, value)
        end
    end

    # Return the parsed values as a tuple
    return judge, [Beta,P], [L, Lz,zchunk, ychunk, xchunk, Dimension, chi, Ndisorder, Nsample, Ntherm, step, Nreplic, seed, verbose]
end



T = Array{Float64}(undef, 2)
parsed_params = Array{Int64}(undef, 14)
judge, T, parsed_params = parse_input()
pa=Parament(judge,T...,parsed_params...)
sp=Spin(pa)

st=Statistics(pa)
da=DataFile(pa)
te=Tensor(pa.chi,pa.zchunk,pa.ychunk,pa.xchunk)
Random.seed!(parsed_params[end])


if pa.chi != 0
    SimulationTNMH(pa,sp,st,da)
else
    SimulationMetro(pa,sp,st,da)
end

"""
julia 3Drunargs.jl  --Type FBC --P 0.5 --zchunk 1 --ychunk 4 --xchunk 4 --Lz 4 --L 4 --Dimension 3 --Beta 1.5 --chi 4 --Ndisorder 10 --Nsample 100 --Ntherm 100 --step 1 --Nreplic 2 --seed 1 --verbose 1
"""
