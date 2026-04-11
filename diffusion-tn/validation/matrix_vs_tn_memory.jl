include("../src/utils.jl")

using ITensors, ITensorMPS
using LinearAlgebra
using ProgressMeter

function main()
    ITensors.disable_warn_order()
    outfile = joinpath(@__DIR__, "matrix_vs_tn_memory_n=7.csv")

    ns = 4:10
    mat_max_n = 7
    cfl = 0.1
    steps = 100

    cutoff = 1e-12
    maxdim = 128


    # =============
    # GLOBAL WARMUP
    # =============

    n_warm = 4
    Nx = 2^n_warm
    Ny = 2^n_warm
    x = range(0, 1, length=Nx+1)[1:end-1]
    y = range(0, 1, length=Ny+1)[1:end-1]
    u0 = [sin(2π * xi) * sin(2π * yj) for xi in x, yj in y]
    sites_2d = siteinds("S=1/2", 2n_warm)

    mps = grid_to_qtt_mps(u0, sites_2d; cutoff=cutoff)
    A_mpo = timestep_mpo_2d(sites_2d, cfl)
    apply(A_mpo, mps; alg="naive", cutoff=cutoff, maxdim=maxdim)

    u_vec = grid_to_standard_vector(u0)
    A_mat = timestep_operator_2d(Nx, Ny, cfl, cfl)
    A_mat * u_vec

    # =============



    out = open(outfile, "w")
    println(out, "n,Nx,Ny,step,mat_mem_mb,tn_mem_mb,mps_build_mb,mpo_build_mb,mat_build_mb")

    total_iters = length(ns) * steps
    p = Progress(total_iters; desc="Benchmarking")

    for n in ns
        Nx = 2^n
        Ny = 2^n

        x = range(0, 1, length=Nx+1)[1:end-1]
        y = range(0, 1, length=Ny+1)[1:end-1]

        u0 = [sin(2π * xi) * sin(2π * yj) for xi in x, yj in y]

        sites_2d = siteinds("S=1/2", 2n)

        # Build initial MPS and timestep MPO
        m_mps = @allocated mps = grid_to_qtt_mps(u0, sites_2d; cutoff=cutoff)
        m_mpo = @allocated A_mpo = timestep_mpo_2d(sites_2d, cfl)

        m_mps = m_mps / 1024^2
        m_mpo = m_mpo / 1024^2

        use_mat = (n <= mat_max_n)
        if use_mat
            u_vec = grid_to_standard_vector(u0)
            m_mat_build = @allocated begin
                A_mat = timestep_operator_2d(Nx, Ny, cfl, cfl)
            end
            m_mat_build = m_mat_build / 1024^2

            # matrix smoke run
            A_mat * u_vec
        end

        # TN smoke run
        apply(A_mpo, mps; alg="naive", cutoff=cutoff, maxdim=maxdim)

        for step in 1:steps
            m_mat   = missing

            if use_mat
                # matrix step
                m_mat = @allocated u_vec = A_mat * u_vec
                m_mat = m_mat / 1024^2
            end

            # TN step
            m_tn = @allocated mps = apply(A_mpo, mps; alg="naive", cutoff=cutoff, maxdim=maxdim)
            m_tn = m_tn / 1024^2

            bond_dim = maxlinkdim(mps)

            println(out,
                "$n,$Nx,$Ny,$step," *
                "$m_mat,$m_tn," *
                "$m_mps,$m_mpo,$(use_mat ? m_mat_build : missing)"
            )
            
            next!(p; showvalues = [
                (:n, n),
                (:step, step),
                (:bond_dim, bond_dim)
            ])
        end
        flush(out)
    end

    close(out)
    println("\nSaved results to $outfile")
end

main()