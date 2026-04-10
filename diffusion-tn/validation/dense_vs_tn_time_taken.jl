# compares time taken per step for matrix evolution vs TN evolution
# my laptop crashes for matrix evolution at n=8, but luckily, n=7 is the threshold for when each TN step takes less time than each dense step

include("../src/utils.jl")
ITensors.disable_warn_order()
outfile = joinpath(@__DIR__, "diffusion_benchmark.csv")

ns = 4:8
mat_max_n = 7
cfl = 0.1
steps = 100

out = open(outfile, "w")
println(out, "n,Nx,Ny,step,mat_time,tn_time,bond_dim,abs_err,rel_err,mps_build,mpo_build,mat_build")

for n in ns

    println()
    println("===== n = $n =====")

    Nx = 2^n
    Ny = 2^n

    x = range(0, 1, length=Nx+1)[1:end-1]
    y = range(0, 1, length=Ny+1)[1:end-1]

    u0 = [sin(2π * xi) * sin(2π * yj) for xi in x, yj in y]

    sites_2d = siteinds("S=1/2", 2n)

    if n == first(ns) # warm up
        grid_to_qtt_mps(u0, sites_2d)
        timestep_mpo_2d(sites_2d, cfl)
    end

    # Build initial MPS and timestep MPO
    t_mps = @elapsed mps = grid_to_qtt_mps(u0, sites_2d)
    t_mpo = @elapsed A_mpo = timestep_mpo_2d(sites_2d, cfl)

    println("Initial max bond dim of MPS = ", maxlinkdim(mps))
    println("Max bond dim of MPO = ", maxlinkdim(A_mpo))
    println()

    use_mat = (n <= mat_max_n)
    if use_mat
        # matrix reference in QTT / site ordering
        u_qtt = grid_to_qtt_vector(u0, n)
        t_mat_build = @elapsed A_mat_qtt = mpo_to_site_matrix(A_mpo, sites_2d)

        # matrix smoke run
        A_mat_qtt * u_qtt
    end

    # TN smoke run
    apply(A_mpo, mps; alg="naive", cutoff=1e-12, maxdim=128)

    for step in 1:steps
        abs_err = missing
        rel_err = missing
        t_mat   = missing

        if use_mat
            # Dense step
            t_mat = @elapsed u_qtt = A_mat_qtt * u_qtt
            println("Dense step $step time = $t_mat s")
        end

        # TN step
        t_tn = @elapsed mps = apply(A_mpo, mps; alg="naive", cutoff=1e-12, maxdim=128)
        bond_dim = maxlinkdim(mps)
        println("TN step $step time    = $t_tn s, bond dim = ", bond_dim)

        if use_mat
            u_mat_qtt = qtt_vector_to_grid(u_qtt, n)
            u_tn = qtt_mps_to_grid(mps, sites_2d)

            abs_err = norm(u_mat_qtt - u_tn)
            rel_err = abs_err / norm(u_mat_qtt)

            println("Step $step:")
            println("  abs error = ", abs_err)
            println("  rel error = ", rel_err)
        end

        println(out, "$n,$Nx,$Ny,$step,$t_mat,$t_tn,$bond_dim,$abs_err,$rel_err,$t_mps,$t_mpo,$(use_mat ? t_mat_build : missing)")
        flush(out)
    end
end

close(out)
println("\nSaved results to $outfile")