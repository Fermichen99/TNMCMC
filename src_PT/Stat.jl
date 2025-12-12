include("DicDat.jl")
using Printf

listtem = ["Ene1","Mag1","Qvr1","Acc","time","memory"]
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
    SpinGlass_dir :: String
    folder_names::Vector{String}
    temdoc_list :: Vector{Dict{String, IO}}

    function DataFile(pa::Parament, Betalist::Vector{Float64})
        script_dir = dirname(@__FILE__)
        SpinGlass_dir = joinpath(script_dir)

        indices = collect(1:10:length(Betalist))

        folder_names = String[]
        temdoc_list = Vector{Dict{String, IO}}(undef, length(indices))

        for (i, idx) in enumerate(indices)
            Beta = Betalist[idx]
            Pstr = @sprintf("%.5f", Beta)
            Lstr = @sprintf("%.0f", pa.L)
            Zstr = @sprintf("%.0f", pa.zchunk)
            Chistr = @sprintf("%.0f", pa.chi)
            folder_name = joinpath(SpinGlass_dir, "dat", string("L=",Lstr), string("Beta=",Pstr), string("Zchunk=",Zstr), string("chi=",Chistr), string("seed=",pa.seed))
            if !ispath(folder_name)
                mkpath(folder_name)
            end

            temdoc = Dict{String, IO}()
            for para in listtem
                temdoc[para] = open(joinpath(folder_name, "$(pa.Dimension)D$(para)"), "a")
                print(temdoc[para], @sprintf("%6s", para), @sprintf("%8i",pa.L), " ", @sprintf("%10.6f",Beta), " ", @sprintf("%10.6f",pa.P), " ", @sprintf("%6i", pa.Nsample), " ", @sprintf("%6i", pa.Ndisorder), " ", @sprintf("%6i", pa.Ntherm), " ", @sprintf("%6i", pa.chi), " ", @sprintf("%6i", pa.seed), "\n")
            end
            folder_names = push!(folder_names, folder_name)
            temdoc_list[i] = temdoc
        end

        new(SpinGlass_dir, folder_names, temdoc_list)
    end
end
export DataFile


function measure(pa::Parament,sp::Spin,st::Statistics,da::DataFile,idis::Int,Spin::Array{Int,5}, Betalist::Vector{Float64}) 

    indices = collect(1:10:length(Betalist))
    for (in,ind) in enumerate(indices)
        #for i in 1:pa.Nreplic
            i = 1
            E = GetEnergy(pa,sp,Spin[ind,:,:,:,i])/pa.Vol
            M = sum(Spin[ind,:,:,:,i])/pa.Vol
            if M < 0 st.count += 1 end
            #Magbul =  sum(Spin[qtL:haL, qtL:haL, qtL:haL,i])*6/pa.Vol
            print(da.temdoc_list[in]["Ene1"],  @sprintf("%10.6f",E), " ")
            print(da.temdoc_list[in]["Mag1"],  @sprintf("%10.6f",M), " ")
            #print(da.temdoc["Magbul"],  @sprintf("%10.6f",Magbul), " ")
        #end
        if pa.Nreplic == 2
            q = sum(Spin[ind,:,:,:,1] .* Spin[ind,:,:,:,2])/pa.Vol
            print(da.temdoc_list[in]["Qvr1"],  @sprintf("%10.6f",q), " ")
        end

        for para in listtem
            flush(da.temdoc_list[in][para])
        end
    end
end


function NormSample(pa::Parament,st::Statistics,da::DataFile,idis::Int, Betalist::Vector{Float64})
    indices = collect(1:10:length(Betalist))
    for (in,ind) in enumerate(indices)
        for para in listtem
            print(da.temdoc_list[in][para], "\n")
            flush(da.temdoc_list[in][para])
        end
    end
end


function closedoc(da::DataFile, Betalist::Vector{Float64})
    indices = collect(1:10:length(Betalist))
    for (in,ind) in enumerate(indices)
        for para in listtem
            close(da.temdoc_list[in][para])
        end
    end
end


export write2file




