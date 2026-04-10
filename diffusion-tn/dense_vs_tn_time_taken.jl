# compares time taken per step for dense evolution vs TN evolution
# my laptop crashes for dense evolution at n=8, but luckily, n=7 is the threshold for when each TN step takes less time than each dense step

include("utils.jl")
ITensors.disable_warn_order()
outfile = joinpath(@__DIR__, "diffusion_benchmark.csv")

ns = 4:10
dense_max_n = 7
cfl = 0.1
steps = 100

out = open(outfile, "w")
println(out, "n,Nx,Ny,step,dense_time,tn_time,bond_dim,abs_err,rel_err,mps_build,mpo_build,dense_build")

for n in ns

    println()
    println("===== n = $n =====")

    Nx = 2^n
    Ny = 2^n

    x = range(0, 1, length=Nx+1)[1:end-1]
    y = range(0, 1, length=Ny+1)[1:end-1]

    u0 = [sin(2π * xi) * sin(2π * yj) for xi in x, yj in y]

    sites2d = siteinds("S=1/2", 2n)

    if n == first(ns) # warm up
        grid_to_grouped_mps_2d(u0, sites2d)
        timestep_mpo_2d(sites2d, cfl)
    end

    # Build initial MPS and timestep MPO
    t_mps = @elapsed mps = grid_to_grouped_mps_2d(u0, sites2d)
    t_mpo = @elapsed A_mpo = timestep_mpo_2d(sites2d, cfl)

    println("Initial max bond dim of MPS = ", maxlinkdim(mps))
    println("Max bond dim of MPO = ", maxlinkdim(A_mpo))
    println()

    use_dense = (n <= dense_max_n)
    if use_dense
        # Dense grouped reference
        u_grouped = grid_to_grouped_vector_2d(u0, n)
        t_dense_build = @elapsed A_dense = mpo_to_matrix(A_mpo, sites2d)

        # dense smoke run
        A_dense * u_grouped
    end

    # TN smoke run
    apply(A_mpo, mps; alg="naive", cutoff=1e-12, maxdim=128)

    for step in 1:steps
        abs_err = missing
        rel_err = missing
        t_dense = missing

        if use_dense
            # Dense step
            t_dense = @elapsed u_grouped = A_dense * u_grouped
            println("Dense step $step time = $t_dense s")
        end

        # TN step
        t_tn = @elapsed mps = apply(A_mpo, mps; alg="naive", cutoff=1e-12, maxdim=128)
        bond_dim = maxlinkdim(mps)
        println("TN step $step time    = $t_tn s, bond dim = ", bond_dim)

        if use_dense
            u_dense = grouped_vector_to_grid_2d(u_grouped, n)
            u_tn = grouped_mps_to_grid_2d(mps, sites2d)

            abs_err = norm(u_dense - u_tn)
            rel_err = abs_err / norm(u_dense)

            println("Step $step:")
            println("  abs error = ", abs_err)
            println("  rel error = ", rel_err)
        end

        println(out, "$n,$Nx,$Ny,$step,$t_dense,$t_tn,$bond_dim,$abs_err,$rel_err,$t_mps,$t_mpo,$(use_dense ? t_dense_build : missing)")
        flush(out)
    end
end

close(out)
println("\nSaved results to $outfile")