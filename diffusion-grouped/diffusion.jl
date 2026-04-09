include("utils.jl")

n = 5
cfl = 0.1
steps = 20

Nx = 2^n
Ny = 2^n

x = range(0, 1, length=Nx+1)[1:end-1]
y = range(0, 1, length=Ny+1)[1:end-1]

u0 = [sin(2π * xi) * sin(2π * yj) for xi in x, yj in y]

sites2d = siteinds("S=1/2", 2n)

# Build initial MPS and timestep MPO
mps = grid_to_grouped_mps_2d(u0, sites2d)
A_mpo = timestep_mpo_2d(sites2d, cfl)

# Dense grouped reference
u_grouped = grid_to_grouped_vector_2d(u0, n)
A_dense = mpo_to_matrix(A_mpo, sites2d)


println("Initial max bond dim of MPS = ", maxlinkdim(mps))
println("Max bond dim of MPO = ", maxlinkdim(A_mpo))
println()

for step in 1:steps
    # Dense step
    u_grouped = A_dense * u_grouped

    # TN step
    mps = apply(A_mpo, mps; alg="naive", cutoff=1e-12, maxdim=128)

    # Express 
    T_grouped = reshape(u_grouped, ntuple(_ -> 2, 2n)...)
    u_dense = grouped_tensor_to_grid_2d(T_grouped, n)
    u_tn = grouped_mps_to_grid_2d(mps, sites2d)

    abs_err = norm(u_dense - u_tn)
    rel_err = abs_err / norm(u_dense)

    println("Step $step:")
    println("  abs error = ", abs_err)
    println("  rel error = ", rel_err)
    println("  max bond dim = ", maxlinkdim(mps))
end