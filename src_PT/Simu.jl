using JLD2
using Dates
using Logging
using Base.Threads

function SimulationTNMH(pa::Parament,sp::Spin,st::Statistics,da::DataFile)
    start_time = now()
    st.Acc[:] .= 0.0
    # makesure the directory exists
    # io = open(joinpath(da.folder_name, "logfile.log"), "w")
    # global_logger(SimpleLogger(io)); last_logged_progress = 0.0
    # Ntot = pa.Ndisorder*(pa.Ntherm+pa.Nsample)
    spin_dir = "spinwbond"
    if !isdir(spin_dir)
        mkdir(spin_dir)
    end
    ldir = joinpath(spin_dir, "L=$(pa.L)", "seed=$(pa.seed)")
    if !isdir(ldir)
        mkpath(ldir)
    end

    Betalist = range(pa.Beta_u, stop=pa.Beta_d, length=pa.nBeta) |> collect


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
        #spinold = ones(Int, pa.Lz, pa.L, pa.L, pa.Nreplic)
		spinold = ones(Int, pa.nBeta, pa.Lz, pa.L, pa.L, pa.Nreplic)
    
        spinnew = zeros(Int, pa.nBeta, pa.Lz, pa.L, pa.L,pa.Nreplic)
        lnprold = zeros(Float64, pa.nBeta, pa.Nreplic); lnprnew = zeros(Float64, pa.nBeta, pa.Nreplic)
        Energyold = zeros(Float64, pa.nBeta, pa.Nreplic); Energynew = zeros(Float64, pa.nBeta, pa.Nreplic)

        for itherm in 1:pa.Ntherm
            for direction in directions
                for (ibeta, beta) in enumerate(Betalist)
                    for ite in 1:pa.Nreplic
                        lnprold_t  = similar(lnprold)   # size = pa.Nreplic
                        lnprnew_t  = similar(lnprnew)
                        Energyold_t = similar(Energyold)
                        Energynew_t = similar(Energynew)
                        for zin in 1:(pa.Lz-pa.zchunk+1)
                            for yin in 1:pa.ychunk:pa.L, xin in 1:pa.xchunk:pa.L
                                select_zeros!(zin,yin,xin,pa.zchunk,pa.ychunk,pa.xchunk,ite,ibeta,spinold)
                                te_thr=Tensor(pa.chi,pa.zchunk,pa.ychunk,pa.xchunk)
                                InitTensor(zin,yin,xin,pa,sp,spinold[ibeta,:,:,:,ite],te_thr,beta)
                                spinold[ibeta,:,:,:,ite],lnprold_t[ibeta,ite],Energyold_t[ibeta,ite] = TNMH_oe(pa,sp,te_thr,spinold[ibeta,:,:,:,ite],zin,yin,xin,beta)
                                for step in 1:pa.Nstep
                                    spinnew[ibeta,:,:,:,ite] .= spinold[ibeta,:,:,:,ite]
                                    select_zeros!(zin,yin,xin,pa.zchunk,pa.L,pa.L,ite,ibeta,spinnew)
                                    spinnew[ibeta,:,:,:,ite],lnprnew_t[ibeta,ite],Energynew_t[ibeta,ite] = TNMH_oe(pa,sp,te_thr,spinnew[ibeta,:,:,:,ite],zin,yin,xin,beta)
                                    spinold[ibeta,:,:,:,ite],lnprold_t[ibeta,ite],Energyold_t[ibeta,ite] = Accept(pa,sp,st,da,spinold[ibeta,:,:,:,ite],spinnew[ibeta,:,:,:,ite],lnprold_t[ibeta,ite],lnprnew_t[ibeta,ite],Energyold_t[ibeta,ite],Energynew_t[ibeta,ite],beta,false) 
                                end  
                            end
                        end
                        spinold[ibeta,:,:,:,:] .= Metropolis(pa,sp,spinold[ibeta,:,:,:,:],beta)
                    end
                end
                attempt_replica_exchange!(pa, spinold, Betalist, Energyold)
                flip_model!(sp,spinold)
            end
            # progress = (step_long + itherm) / Ntot * 100
            # if progress - last_logged_progress >= 1.0
            #     last_logged_progress = progress 
            #     @info "done $(round(progress, digits=2))%"
            #     flush(io)  
            # end
        end
        st.count = 0
        for isamp in 1:pa.Nsample
            for direction in directions
                time1 = now()
                for (ibeta, beta) in enumerate(Betalist)
                    for ite in 1:pa.Nreplic
                        te_thr=Tensor(pa.chi,pa.zchunk,pa.ychunk,pa.xchunk)
                        lnprold_t  = similar(lnprold)   # size = pa.Nreplic
                        lnprnew_t  = similar(lnprnew)
                        Energyold_t = similar(Energyold)
                        Energynew_t = similar(Energynew)
                        for zin in 1:(pa.Lz-pa.zchunk+1)
                            for yin in 1:pa.ychunk:pa.L, xin in 1:pa.xchunk:pa.L
                                select_zeros!(zin,yin,xin,pa.zchunk,pa.ychunk,pa.xchunk,ite,ibeta,spinold)
                                InitTensor(zin,yin,xin,pa,sp,spinold[ibeta,:,:,:,ite],te_thr,beta)
                                spinold[ibeta,:,:,:,ite],lnprold_t[ibeta,ite],Energyold_t[ibeta,ite] = TNMH_oe(pa,sp,te_thr,spinold[ibeta,:,:,:,ite],zin,yin,xin,beta)
                                for step in 1:pa.Nstep
                                    spinnew[ibeta,:,:,:,ite] .= spinold[ibeta,:,:,:,ite]
                                    select_zeros!(zin,yin,xin,pa.zchunk,pa.L,pa.L,ite,ibeta,spinnew)
                                    spinnew[ibeta,:,:,:,ite],lnprnew_t[ibeta,ite],Energynew_t[ibeta,ite] = TNMH_oe(pa,sp,te_thr,spinnew[ibeta,:,:,:,ite],zin,yin,xin,beta)
                                    spinold[ibeta,:,:,:,ite],lnprold_t[ibeta,ite],Energyold_t[ibeta,ite] = Accept(pa,sp,st,da,spinold[ibeta,:,:,:,ite],spinnew[ibeta,:,:,:,ite],lnprold_t[ibeta,ite],lnprnew_t[ibeta,ite],Energyold_t[ibeta,ite],Energynew_t[ibeta,ite],beta,false) 
                                end  
                            end
                        end
                        spinold[ibeta,:,:,:,:] .= Metropolis(pa,sp,spinold[ibeta,:,:,:,:],beta)
                    end
                end
                time2 = now()
                # print(da.temdoc["memory"],  @sprintf("%12.4f",round(get_memory_usage_mb(), digits=2)), " ")
                time2 = now()
                elapsed_seconds = Dates.value(time2-time1) / 1e3
                # print(da.temdoc["time"],  @sprintf("%10.6f",elapsed_seconds), " ") 
                attempt_replica_exchange!(pa, spinold, Betalist, Energyold)
                measure(pa,sp,st,da,idis,spinold, Betalist)
                # if st.count > 1 break end
                flip_model!(sp,spinold)
            end
            #if st.count > 1 break end

            # progress = (step_long + (pa.Ntherm+isamp)) / Ntot * 100
            # if progress - last_logged_progress >= 1.0
            #     last_logged_progress = progress 
            #     @info "done $(round(progress, digits=2))%"
            #     flush(io)  
            # end
        end
        NormSample(pa,st,da,idis, Betalist)
    end
    
    closedoc(da, Betalist)
    @show st.Acc[1]/st.Acc[2]
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

    Betalist = range(pa.Beta_u, stop=pa.Beta_d, length=pa.nBeta) |> collect
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
        spinold = ones(Int, pa.nBeta, pa.Lz, pa.L, pa.L, pa.Nreplic)
        Energyold = zeros(Float64, pa.nBeta, pa.Nreplic)

        for itherm in 1:pa.Ntherm
            for (ibeta, beta) in enumerate(Betalist)
                spinold[ibeta,:,:,:,:] .= Metropolis(pa,sp,spinold[ibeta,:,:,:,:],beta)
            end
        end

        st.count = 0
        for isamp in 1:pa.Nsample
            for (ibeta, beta) in enumerate(Betalist)
                spinold[ibeta,:,:,:,:] .= Metropolis(pa,sp,spinold[ibeta,:,:,:,:],beta)
            end
            for (ibeta, beta) in enumerate(Betalist)
                for ite in 1:pa.Nreplic
                    Energyold[ibeta,ite] = GetEnergy(pa,sp,spinold[ibeta,:,:,:,ite])
                end
            end

            attempt_replica_exchange!(pa, spinold, Betalist, Energyold)
            measure(pa,sp,st,da,idis,spinold, Betalist)
        end
        NormSample(pa,st,da,idis, Betalist)
    end
    closedoc(da, Betalist)
end


function get_memory_usage_mb()
    pid = getpid()  # 获取当前进程ID
    command = `ps -o rss= -p $pid`  # 构造获取常驻集大小的命令
    rss = read(command, String)  # 执行命令并读取输出
    rss = strip(rss)  # 去除前后空格
    memory_usage_kb = parse(Int, rss)  # 常驻集大小（KB）
    memory_usage_mb = memory_usage_kb / 1024  # 转换为 MB
    return memory_usage_mb
end

function total_memory_usage()
    total_memory = 0
    
    # 获取所有全局变量的内存占用
    for var in names(Main)
        total_memory += Base.summarysize(Main.eval(Symbol(var)))
    end
    
    # 获取当前函数的内存占用
    total_memory += Base.summarysize(stacktrace()[1].func)
    
    return total_memory
end

function attempt_replica_exchange!(pa::Parament, spin::Array{Int, 5}, betalist::Vector{Float64}, energy::Array{Float64, 2})
    for i in 1:length(betalist)-1
        for j in 1:pa.Nreplic
            beta1 = betalist[i]
            beta2 = betalist[i+1]
            Delta = (beta1 - beta2) * (energy[i,j] - energy[i+1,j])
            if Delta >= 0 || rand() < exp(Delta)
                temp = copy(spin[i,:,:,:,j])
                spin[i,:,:,:,j] .= spin[i+1,:,:,:,j]
                spin[i+1,:,:,:,j] .= temp
            end
        end
    end
end