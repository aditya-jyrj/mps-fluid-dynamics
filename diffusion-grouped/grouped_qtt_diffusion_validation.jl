include("utils.jl")

n = 2
cfl = 0.1

steps = 20

Nx = 2^n
Ny = 2^n

x = range(0, 1, length=Nx+1)[1:end-1]
y = range(0, 1, length=Ny+1)[1:end-1]

u0 = [sin(2π * xi) * sin(2π * yj) for xi in x, yj in y]

sites1d = siteinds("S=1/2", n)
sites2d = siteinds("S=1/2", 2n)


# ==========================
# CHECK STATE REPRESENTATION
# ==========================

# GRID vs GRID -> MPS -> GRID
mps0 = grid_to_grouped_mps_2d(u0, sites2d)
u0_back = grouped_mps_to_grid_2d(mps0, sites2d)
println("Grouped MPS reconstruction error = ", norm(u0 - u0_back))

# GRID -> GROUPED vs GRID -> MPS -> GROUPED
u0_grouped_dense = grid_to_grouped_vector_2d(u0, n)
u0_grouped_from_mps = grouped_mps_to_vector(mps0, sites2d)
println("Grouped state vector error = ", norm(u0_grouped_dense - u0_grouped_from_mps))

# =============================
# CHECK OPERATOR REPRESENTATION
# =============================
println()

# 1D LAPLACIAN: 1D ANALYTICAL MPO -> MATRIX vs MATRIX
L1_mpo = laplacian_mpo_1d(sites1d)
L1_dense_from_mpo = mpo_to_matrix(L1_mpo, sites1d)
L1_dense_ref = laplacian_1d(2^n, :dirichlet)

println("1D Laplacian MPO error = ", norm(L1_dense_from_mpo - L1_dense_ref))

# 2D LAPLACIAN: 2D ANALYTICAL MPO -> MATRIX vs MATRIX
L2_mpo = laplacian_mpo_2d(sites2d)
L2_dense_from_mpo = mpo_to_matrix(L2_mpo, sites2d)
L2_dense_ref = laplacian_2d(Nx, Ny)

println("2D Laplacian MPO error = ", norm(L2_dense_from_mpo - L2_dense_ref))

# 2D TIMESTEP: 2D ANALYTICAL MPO -> MATRIX vs MATRIX
A_mpo = timestep_mpo_2d(sites2d, cfl)
A_dense_from_mpo = mpo_to_matrix(A_mpo, sites2d)
A_dense_ref = timestep_operator_2d(Nx, Ny, cfl, cfl)

println("2D timestep MPO error = ", norm(A_dense_from_mpo - A_dense_ref))


# ==========================
# CHECK END-TO-END EVOLUTION
# ==========================
println()

# DENSE TIME EVOLUTION
u0_grouped = grid_to_grouped_vector_2d(u0, n)
u_grouped = copy(u0_grouped)

for _ in 1:steps
    u_grouped = A_dense_from_mpo * u_grouped
end

u_grid_dense_grouped = grouped_vector_to_grid_2d(u_grouped, n)

# GROUPED TN TIME EVOLUTION
mps = evolve_mps_with_mpo(mps0, A_mpo, steps; cutoff=1e-12, maxdim=128)

u_grid_tn = grouped_mps_to_grid_2d(mps, sites2d)

println("Grouped dense vs TN error after $steps steps = ",
        norm(u_grid_dense_grouped - u_grid_tn))
println("Relative grouped dense vs TN error after $steps steps = ",
        norm(u_grid_dense_grouped - u_grid_tn) / norm(u_grid_dense_grouped))
println("Max bond dim of mps after $steps steps = ", maxlinkdim(mps))
println("Max bond dim of A_mpo = ", maxlinkdim(A_mpo))