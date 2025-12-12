function Metropolis(pa::Parament, sp::Spin, spin::Array{Int, 4}, Beta::Float64)
    for i in 1:pa.Nreplic
        for z in 1:pa.Lz, y in 1:pa.L, x in 1:pa.L
            ns = spin[z, y, x, i]
            Nei = 0
            for k in 1:pa.nnb
                kz = (z + sp.Dz[k] + pa.Lz - 1) % pa.Lz + 1
                ky = (y + sp.Dy[k] + pa.L - 1) % pa.L + 1
                kx = (x + sp.Dx[k] + pa.L - 1) % pa.L + 1
                Nei += spin[kz, ky, kx, i] * sp.bond[z, y, x, k]
            end
            DeltaE = 2 * ns * Nei * Beta 
            if DeltaE < 0 || rand() < exp(-DeltaE)
                spin[z, y, x, i] *= -1
            end
        end
    end
    return spin
end


function GetEnergy(pa::Parament,sp::Spin,spin::Array{Int, 3})
    Energy = 0.0
    for z in 1:pa.Lz, y in 1:pa.L, x in 1:pa.L
        ns = spin[z, y, x]
        Nei = 0
        for k in 1:pa.nnb
            kz = (z + sp.Dz[k] + pa.Lz - 1) % pa.Lz + 1
            ky = (y + sp.Dy[k] + pa.L - 1) % pa.L + 1
            kx = (x + sp.Dx[k] + pa.L - 1) % pa.L + 1
            Nei += spin[kz, ky, kx] * sp.bond[z, y, x, k]
        end
        Energy += -ns * Nei/2
    end
    return Energy
end

