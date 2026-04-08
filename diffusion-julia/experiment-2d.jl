include("utils.jl")

# DEFINE 2D GRID
# we only allow n to be specified, where grid size is 2^n by 2^n

n = 2
cfl = 0.1

Nx = 2^n
Ny = 2^n
Ntot = Nx * Ny

x = range(0, 1, length=Nx+1)[1:end-1]
y = range(0, 1, length=Ny+1)[1:end-1]

# INITIAL CONDITIONS

u0 = [sin(2π * xi) * sin(2π * yj) for xi in x, yj in y] 


# standard-basis dense vector
u0_std = grid2d_to_standard_vec(u0)

# interleaved-basis dense vector
u0_inter = grid2d_to_interleaved_vec(u0, n)

println("Standard vec length = ", length(u0_std))
println("Interleaved vec length = ", length(u0_inter))

P = zeros(Float64, Ntot, Ntot)

for k in 1:Ntot
    e = zeros(Float64, Ntot)
    e[k] = 1.0

    # interpret basis vector in standard grid ordering
    Egrid = standard_vec_to_grid2d(e, n)

    # map that basis element into interleaved vector ordering
    P[:, k] = grid2d_to_interleaved_vec(Egrid, n)
end

println("Permutation check ||P'P - I|| = ", norm(P' * P - I))


A_std = A_exact_2d(Nx, Ny, cfl, cfl)

# same operator, but expressed in interleaved basis
A_inter = P * A_std * P'

# dense one-step evolution in standard basis
u1_std = A_std * u0_std
u1_grid_dense = standard_vec_to_grid2d(u1_std, n)

println("Dense one-step grid:")
println(u1_grid_dense)

u1_inter_dense = A_inter * u0_inter

# map interleaved evolved vector back to grid
T1_inter = reshape(u1_inter_dense, ntuple(_ -> 2, 2 * n)...)
u1_grid_from_inter_dense = interleaved_qtt_tensor_to_grid2d(T1_inter, n)

println("Dense interleaved-basis one-step error = ",
        norm(u1_grid_dense - u1_grid_from_inter_dense))

nsites = 2 * n
sites = siteinds("S=1/2", nsites)

mps0 = dense_2d_to_interleaved_qtt_mps(u0, sites)
A_mpo_2d = dense_matrix_to_qtt_mpo(A_inter, sites; cutoff=1e-12)

A_mpo_dense = mpo_to_matrix(A_mpo_2d, sites)
println("MPO matrix error = ", norm(A_mpo_dense - A_inter))

mps1 = apply(A_mpo_2d, mps0; alg="naive", cutoff=1e-12, maxdim=128)
u1_grid_tn = interleaved_qtt_mps_to_grid2d(mps1, sites)

println("Dense vs TN one-step grid error = ", norm(u1_grid_dense - u1_grid_tn))
println("TN one-step grid:")
println(u1_grid_tn)