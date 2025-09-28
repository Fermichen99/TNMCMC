include("DicDat.jl")
using Printf

listtem = ["Ene1","Mag1","Ene2","Mag2","Qvrt","Acc","time"]
single = String[]
complx = String[]
for key in listtem
    if get(Dat, key, "") == "single"
        push!(single, key)
    elseif get(Dat, key, "") == "complx"
        push!(complx, key)
    end
end

mutable struct Statistics
    NObs_b::Int
    NObs_c::Int
    NObs::Int
    Acc::Array{Float64, 1}
    count::Int

    function Statistics(pa::Parament)
        NObs_b = length(single)
        NObs_c = length(complx)
        NObs = NObs_b + NObs_c 
        Acc = zeros(Float64, 2)
        count = 0
        new(NObs_b, NObs_c, NObs, Acc, count)
    end
end
export Statistics



struct DataFile
    folder_name::String
    temdoc :: Dict{String, IO}

    function DataFile(pa::Parament)
        Pstr=@sprintf("%.5f",pa.Beta)
        Lstr=@sprintf("%.0f",pa.L)
        Zstr=@sprintf("%.0f",pa.zchunk)
        Chistr=@sprintf("%.0f",pa.chi)
        folder_name = joinpath("dat", string("L=",Lstr) ,string("Beta=", Pstr), string("Zchunk=", Zstr), string("chi=", Chistr),string("seed=",pa.seed))
        if !ispath(folder_name)
            mkpath(folder_name)
        end

        temdoc = Dict{String, IO}()
        for para in listtem
            temdoc[para] = open(joinpath(folder_name, "$(pa.Dimension)D$(para)"), "a")
            print(temdoc[para], @sprintf("%6s", para), @sprintf("%8i",pa.L), " ", @sprintf("%10.6f",pa.Beta), " ", @sprintf("%10.6f",pa.P), " ", @sprintf("%6i", pa.Nsample), " ", @sprintf("%6i", pa.Ndisorder), " ", @sprintf("%6i", pa.Ntherm), " ", @sprintf("%6i", pa.chi), " ", @sprintf("%6i", pa.seed), "\n")
        end
        new(folder_name, temdoc)
    end
end
export DataFile

function measure(pa::Parament,sp::Spin,st::Statistics,da::DataFile,idis::Int,Spin::Array{Int,4},prt) 
    q = 0
    for i in 1:pa.Nreplic
        E = GetEnergy(pa,sp,Spin[:,:,:,i])/pa.Vol
        M = sum(Spin[:,:,:,i])/pa.Vol
	    if prt
            if i == 1
	            print(da.temdoc["Ene1"],  @sprintf("%10.6f",E), " ")
        	    print(da.temdoc["Mag1"],  @sprintf("%10.6f",M), " ")
            elseif i == 2
	            print(da.temdoc["Ene2"],  @sprintf("%10.6f",E), " ")
        	    print(da.temdoc["Mag2"],  @sprintf("%10.6f",M), " ")
            end
	    end
    end
    q = sum(Spin[:,:,:,1] .* Spin[:,:,:,2])/pa.Vol
    if prt
	    print(da.temdoc["Qvrt"],  @sprintf("%10.6f",q), " ")
    end

    for para in listtem
        flush(da.temdoc[para])
    end
end


function NormSample(pa::Parament,st::Statistics,da::DataFile,idis::Int)
    for para in listtem
        print(da.temdoc[para], "\n")
        flush(da.temdoc[para])
    end
end


function closedoc(da::DataFile)
    for para in listtem
        close(da.temdoc[para])
    end
end


export write2file




