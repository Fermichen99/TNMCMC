using JLD2
using Dates
using Logging
using Base.Threads

function SimulationTNMH(pa::Parament,sp::Spin,st::Statistics,da::DataFile)
    start_time = now()
    st.Acc[:] .= 0.0
    # makesure the directory exists
    io = open(joinpath(da.folder_name, "logfile.log"), "w")
    global_logger(SimpleLogger(io)); last_logged_progress = 0.0
    Ntot = pa.Ndisorder*(pa.Ntherm+pa.Nsample)
    spin_dir = "spinwbond"
    if !isdir(spin_dir)
        mkdir(spin_dir)
    end
    ldir = joinpath(spin_dir, "L=$(pa.L)", "seed=$(pa.seed)")
    if !isdir(ldir)
        mkpath(ldir)
    end

    directions = [:zyx2yxz,:yxz2xzy,:xzy2zyx]

    for idis in 1:pa.Ndisorder
        step_long = (idis-1)*(pa.Nsample+pa.Ntherm)
        file_path = joinpath(ldir, "bond_L$(pa.L)_n$(idis).jld2")

        if isfile(file_path)
            @load file_path bond
            sp.bond .= bond
        else
            InitBondFBC(pa,sp)
            bond = copy(sp.bond)
            @save file_path bond
        end
	    spinold = ones(Int, pa.Lz, pa.L, pa.L, pa.Nreplic)
    
        spinnew = zeros(Int, pa.Lz, pa.L, pa.L,pa.Nreplic)
        lnprold = zeros(Float64, pa.Nreplic); lnprnew = zeros(Float64, pa.Nreplic)
        Energyold = zeros(Float64, pa.Nreplic); Energynew = zeros(Float64, pa.Nreplic)

        for itherm in 1:pa.Ntherm
            if pa.verbose == 1
                println("Thermalization $(itherm)th step")
            end
            for direction in directions
                if pa.verbose == 1
                    println("Direction $(direction)")
                end
                Threads.@threads for ite in 1:pa.Nreplic
                    te_thr=Tensor(pa.chi,pa.zchunk,pa.ychunk,pa.xchunk)
                    lnprold_t  = similar(lnprold)   
                    lnprnew_t  = similar(lnprnew)
                    Energyold_t = similar(Energyold)
                    Energynew_t = similar(Energynew)
                    for zin in 1:(pa.Lz-pa.zchunk+1)
                        for yin in 1:pa.ychunk:pa.L, xin in 1:pa.xchunk:pa.L
                            select_zeros!(zin,yin,xin,pa.zchunk,pa.ychunk,pa.xchunk,ite,spinold)
                            InitTensor(zin,yin,xin,pa,sp,spinold[:,:,:,ite],te_thr)
                            if pa.verbose == 1
                                println("Contracted tensor network layers ($(zin)-$(zin+pa.zchunk-1))")
                            end
                            spinold[:,:,:,ite],lnprold_t[ite],Energyold_t[ite] = TNMH_oe(pa,sp,te_thr,spinold[:,:,:,ite],zin,yin,xin)
                            if pa.verbose == 1
                                println("Finished the 0th sampling")
                            end
                            for step in 1:pa.Nstep
                                spinnew[:,:,:,ite] .= spinold[:,:,:,ite]
                                select_zeros!(zin,yin,xin,pa.zchunk,pa.L,pa.L,ite,spinnew)
                                spinnew[:,:,:,ite],lnprnew_t[ite],Energynew_t[ite] = TNMH_oe(pa,sp,te_thr,spinnew[:,:,:,ite],zin,yin,xin)
                                if pa.verbose == 1
                                    println("Finished the $(step)th sampling")
                                end
                                spinold[:,:,:,ite],lnprold_t[ite],Energyold_t[ite] = Accept(pa,sp,st,da,spinold[:,:,:,ite],spinnew[:,:,:,ite],lnprold_t[ite],lnprnew_t[ite],Energyold_t[ite],Energynew_t[ite],true) 
                            end  
                        end
                    end
                    spinold .= Metropolis(pa,sp,spinold,pa.Beta)
                end
                flip_model!(sp,spinold)
                if pa.verbose == 1
                    println("Finished flipping the model")
                end
            end
            if pa.verbose == 1
                println()
            end
            progress = (step_long + itherm) / Ntot * 100
            if progress - last_logged_progress >= 1.0
                last_logged_progress = progress 
                @info "done $(round(progress, digits=2))%"
                flush(io)  
            end
        end
        st.count = 0
        for isamp in 1:pa.Nsample
            if pa.verbose == 1
                println("After equilibrium $(isamp)th step")
            end
            for direction in directions
                if pa.verbose == 1
                    println("Direction $(direction)")
                end
                time1 = now()
                Threads.@threads for ite in 1:pa.Nreplic
                    te_thr=Tensor(pa.chi,pa.zchunk,pa.ychunk,pa.xchunk)
                    lnprold_t  = similar(lnprold)   
                    lnprnew_t  = similar(lnprnew)
                    Energyold_t = similar(Energyold)
                    Energynew_t = similar(Energynew)
                    for zin = 1:(pa.Lz-pa.zchunk+1)
                        for yin in 1:pa.ychunk:pa.L, xin in 1:pa.xchunk:pa.L
                            select_zeros!(zin,yin,xin,pa.zchunk,pa.L,pa.L,ite,spinold)
                            InitTensor(zin,yin,xin,pa,sp,spinold[:,:,:,ite],te_thr)
                            if pa.verbose == 1
                                println("Contracted tensor network layers ($(zin)-$(zin+pa.zchunk-1))")
                            end
                            spinold[:,:,:,ite],lnprold_t[ite],Energyold_t[ite] = TNMH_oe(pa,sp,te_thr,spinold[:,:,:,ite],zin,yin,xin)
                            if pa.verbose == 1
                                println("Finished the 0th sampling")
                            end
                            for step in 1:pa.Nstep
                                spinnew[:,:,:,ite] .= spinold[:,:,:,ite]
                                select_zeros!(zin,yin,xin,pa.zchunk,pa.L,pa.L,ite,spinnew)
                                spinnew[:,:,:,ite],lnprnew_t[ite],Energynew_t[ite] = TNMH_oe(pa,sp,te_thr,spinnew[:,:,:,ite],zin,yin,xin)
                                if pa.verbose == 1
                                    println("Finished the $(step)th sampling")
                                end
                                spinold[:,:,:,ite],lnprold_t[ite],Energyold_t[ite] = Accept(pa,sp,st,da,spinold[:,:,:,ite],spinnew[:,:,:,ite],lnprold_t[ite],lnprnew_t[ite],Energyold_t[ite],Energynew_t[ite],true) 
                            end  
                        end
                    end
                    spinold .= Metropolis(pa,sp,spinold,pa.Beta)
                end
                time2 = now()
                elapsed_seconds = Dates.value(time2-time1) / 1e3
                print(da.temdoc["time"],  @sprintf("%10.6f",elapsed_seconds), " ") 
                measure(pa,sp,st,da,idis,spinold,true)
                if pa.verbose == 1
                    println("Finished collecting physical quantities")
                end
                flip_model!(sp,spinold)
                if pa.verbose == 1
                    println("Finished flipping the model")
                end
            end
            if pa.verbose == 1
                println()
            end

            progress = (step_long + (pa.Ntherm+isamp)) / Ntot * 100
            if progress - last_logged_progress >= 1.0
                last_logged_progress = progress 
                @info "done $(round(progress, digits=2))%"
                flush(io)  
            end
        end
        NormSample(pa,st,da,idis)
    end
    
    closedoc(da)
    if pa.Nstep > 0
        @show st.Acc[1]/st.Acc[2]
    end
    end_time = now()
    elapsed_time = end_time - start_time
    println("Elapsed time (seconds): ", Dates.value(elapsed_time) / 1e3)
end


function SimulationMetro(pa::Parament,sp::Spin,st::Statistics,da::DataFile)
    start_time = now()
    st.Acc[:] .= 0.0

    spin_dir = "spinwbond"
    if !isdir(spin_dir)
        mkdir(spin_dir)
    end
    ldir = joinpath(spin_dir, "L=$(pa.L)", "seed=$(pa.seed)")
    if !isdir(ldir)
        mkpath(ldir)
    end

    for idis in 1:pa.Ndisorder
        file_path = joinpath(ldir, "bond_L$(pa.L)_n$(idis).jld2")

        if isfile(file_path)
            @load file_path bond
            sp.bond .= bond
        else
            InitBondFBC(pa,sp)
            bond = copy(sp.bond)
            @save file_path bond
        end
        spinold = ones(Int, pa.Lz, pa.L, pa.L, pa.Nreplic)

        for itherm in 1:pa.Ntherm
            spinold .= Metropolis(pa,sp,spinold,pa.Beta)
        end

        measure(pa,sp,st,da,idis,spinold,false)
        st.count = 0
        for isamp in 1:pa.Nsample
            spinold .= Metropolis(pa,sp,spinold,pa.Beta)
            measure(pa,sp,st,da,idis,spinold,false)
        end
        NormSample(pa,st,da,idis)
    end
    closedoc(da)
end

