# this notebook checks 

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


# create two vectors, one in standard basis and one in interleaved basis
# then check that they have the same length

u0_std = grid_to_standard_vector_2d(u0)
u0_inter = grid_to_interleaved_vector_2d(u0, n)
println("Standard vec length = ", length(u0_std))
println("Interleaved vec length = ", length(u0_inter))


# CREATE PERMUTATION MATRIX

# generate permutation matrix to convert between standard dense vector and interleaved standard ordering
P = zeros(Float64, Ntot, Ntot)

# for each column in P
for k in 1:Ntot

    # generate the k-th basis vector in standard ordering
    e = zeros(Float64, Ntot)
    e[k] = 1.0

    # reshape the vector to N*N grid: this is the corresponding 2D grid basis state
    Egrid = standard_vector_to_grid_2d(e, n)

    # convert grid basis state to interleaved vector basis state
    # and set it to the k-th column of P
    P[:, k] = grid_to_interleaved_vector_2d(Egrid, n)
end

# verify that P is unitary
println("Permutation check ||P'P - I|| = ", norm(P' * P - I))
println()



# GENERATE TIMESTEP OPERATORS

# generate timestep operator in both dense and interleaved bases
A_std = timestep_operator_2d(Nx, Ny, cfl, cfl)
A_inter = P * A_std * P'



# TIME EVOLUTION

# dense one-step evolution in standard basis
u1_std = A_std * u0_std
u1_grid_dense = standard_vector_to_grid_2d(u1_std, n)
println("Dense one-step grid:")
for row in eachrow(round.(u1_grid_dense; digits=4))
    println(row)
end
println()

# dense one-step evolution in interleaved basis
u1_inter_dense = A_inter * u0_inter
T1_inter = reshape(u1_inter_dense, ntuple(_ -> 2, 2 * n)...) # need to do inter vec -> tensor -> grid because i dont have an inter vec -> grid function
u1_grid_from_inter_dense = interleaved_tensor_to_grid_2d(T1_inter, n)

println("Dense interleaved-basis one-step error = ",
        norm(u1_grid_dense - u1_grid_from_inter_dense))
println()

nsites = 2 * n
sites = siteinds("S=1/2", nsites)

# create mps
mps0 = grid_to_interleaved_mps_2d(u0, sites)

# convert interleaved timestep to MPO and back and check norm (should be ~0)
A_mpo_2d = dense_matrix_to_mpo(A_inter, sites; cutoff=1e-12)
A_mpo_dense = mpo_to_dense_matrix(A_mpo_2d, sites)
println("MPO matrix error = ", norm(A_mpo_dense - A_inter))

# one MPS-MPO timestep contraction
mps1 = apply(A_mpo_2d, mps0; alg="naive", cutoff=1e-12, maxdim=128)
u1_grid_tn = interleaved_mps_to_grid_2d(mps1, sites)

println("Dense vs TN one-step grid error = ", norm(u1_grid_dense - u1_grid_tn))
println("TN one-step grid:")
for row in eachrow(round.(u1_grid_tn; digits=4))
    println(row)
end
println()

println("Max bond dim of mps0 = ", maxlinkdim(mps0))
println("Max bond dim of mps1 = ", maxlinkdim(mps1))
println("Max bond dim of A_mpo_2d = ", maxlinkdim(A_mpo_2d))