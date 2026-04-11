include("../src/utils.jl")

using ITensors, ITensorMPS
using LinearAlgebra
using ProgressMeter

function main()
    ITensors.disable_warn_order()
    outfile = joinpath(@__DIR__, "matrix_vs_tn_time_n=7.csv")

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
    println(out, "n,Nx,Ny,step,mat_time,tn_time,mps_build,mpo_build,mat_build")

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
        t_mps = @elapsed mps = grid_to_qtt_mps(u0, sites_2d; cutoff=cutoff)
        t_mpo = @elapsed A_mpo = timestep_mpo_2d(sites_2d, cfl)

        use_mat = (n <= mat_max_n)
        if use_mat
            u_vec = grid_to_standard_vector(u0)
            t_mat_build = @elapsed begin
                A_mat = timestep_operator_2d(Nx, Ny, cfl, cfl)
            end

            # matrix smoke run
            A_mat * u_vec
        end

        # TN smoke run
        apply(A_mpo, mps; alg="naive", cutoff=cutoff, maxdim=maxdim)

        for step in 1:steps
            t_mat   = missing

            if use_mat
                # matrix step
                t_mat = @elapsed u_vec = A_mat * u_vec
            end

            # TN step
            t_tn = @elapsed mps = apply(A_mpo, mps; alg="naive", cutoff=cutoff, maxdim=maxdim)

            bond_dim = maxlinkdim(mps)

            println(out,
                "$n,$Nx,$Ny,$step," *
                "$t_mat,$t_tn," *
                "$t_mps,$t_mpo,$(use_mat ? t_mat_build : missing)"
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