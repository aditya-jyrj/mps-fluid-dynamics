include("utils.jl")

using ITensors, ITensorMPS
using LinearAlgebra

ITensors.disable_warn_order()

n = 6
cfl = 0.1
steps = 20
cutoff = 1e-20
maxdim = 1000

Nx = 2^n
Ny = 2^n

x = range(0, 1, length=Nx + 1)[1:end-1]
y = range(0, 1, length=Ny + 1)[1:end-1]

u0 = [exp(-50 * ((xi - 0.5)^2 + (yj - 0.5)^2)) for xi in x, yj in y]

sites_2d = siteinds("S=1/2", 2n)

# ==========================
# CHECK STATE REPRESENTATION
# ==========================

# GRID vs GRID -> QTT MPS -> GRID
mps0 = grid_to_qtt_mps_2d(u0, sites_2d; cutoff=cutoff)
u0_back = qtt_mps_to_grid_2d(mps0, sites_2d)
println("QTT MPS reconstruction error = ", norm(u0 - u0_back))

# GRID -> QTT VECTOR vs GRID -> QTT MPS -> SITE VECTOR
u0_qtt_dense = grid_to_qtt_vector_2d(u0, n)
u0_qtt_from_mps = mps_to_site_vector(mps0, sites_2d)
println("QTT state vector error = ", norm(u0_qtt_dense - u0_qtt_from_mps))



# ==========================
# CHECK END-TO-END EVOLUTION
# ==========================
println()

A_mpo = timestep_mpo_2d(sites_2d, cfl)

A_dense_std = timestep_operator_2d(Nx, Ny, cfl, cfl)
A_dense_qtt = standard_to_qtt_matrix_2d(A_dense_std, n)

A_from_mpo = mpo_to_site_matrix(A_mpo, sites_2d)
println("Operator mismatch = ", norm(A_from_mpo - A_dense_qtt))
println()

# DENSE TIME EVOLUTION IN QTT/SITE ORDERING
u0_qtt = grid_to_qtt_vector_2d(u0, n)
u_qtt = copy(u0_qtt)

for _ in 1:steps
    u_qtt = A_dense_qtt * u_qtt
end

u_grid_dense_qtt = qtt_vector_to_grid_2d(u_qtt, n)

# TN TIME EVOLUTION
mps = evolve_mps_with_mpo(mps0, A_mpo, steps; cutoff=cutoff, maxdim=maxdim)
u_grid_tn = qtt_mps_to_grid_2d(mps, sites_2d)

println("QTT dense vs TN error after $steps steps = ",
        norm(u_grid_dense_qtt - u_grid_tn))
println("Relative QTT dense vs TN error after $steps steps = ",
        norm(u_grid_dense_qtt - u_grid_tn) / norm(u_grid_dense_qtt))
println("Max bond dim of mps after $steps steps = ", maxlinkdim(mps))
println("Max bond dim of A_mpo = ", maxlinkdim(A_mpo))