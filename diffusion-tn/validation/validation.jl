include("../src/utils.jl")

using ITensors, ITensorMPS
using LinearAlgebra

ITensors.disable_warn_order()

n = 5
cfl = 0.1
steps = 300
cutoff = 1e-20
maxdim = 1000

Nx = 2^n
Ny = 2^n

x = range(0, 1, length=Nx + 1)[1:end-1]
y = range(0, 1, length=Ny + 1)[1:end-1]

u0 = [exp(-50 * ((xi - 0.5)^2 + (yj - 0.5)^2)) for xi in x, yj in y]

sites2d = siteinds("S=1/2", 2n)

# Check QTT state conversion consistency
u0_qtt = grid_to_qtt_vector_2d(u0, n)
mps0 = grid_to_qtt_mps_2d(u0, sites2d; cutoff=cutoff)
u0_from_mps = mps_to_site_vector(mps0, sites2d)

println("initial qtt vector vs mps vector mismatch = ", norm(u0_qtt - u0_from_mps))

# TN setup
mps = grid_to_qtt_mps_2d(u0, sites2d; cutoff=cutoff)
A_mpo = timestep_mpo_2d(sites2d, cfl)

# Dense setup in QTT ordering
u_qtt = grid_to_qtt_vector_2d(u0, n)
A_dense_std = timestep_operator_2d(Nx, Ny, cfl, cfl)
A_dense_qtt = standard_to_qtt_matrix_2d(A_dense_std, n)

# ==========================
# OPERATOR ACTION CHECK
# ==========================

v = randn(Float64, Nx * Ny)
# pass a random vector and check that both operators evolve to the same result

w_dense = A_dense_qtt * v
w_mpo = apply_mpo_to_site_vector(A_mpo, v, sites2d;
                                cutoff=cutoff, maxdim=maxdim)

println("operator action error = ", norm(w_dense - w_mpo))
println("relative operator action error = ",
        norm(w_dense - w_mpo) / norm(w_dense))
println()

for step in 0:steps
    u_tn_grid = qtt_mps_to_grid_2d(mps, sites2d)
    u_dense_grid = qtt_vector_to_grid_2d(u_qtt, n)

    bond_dim = maxlinkdim(mps)
    rel_err = norm(u_dense_grid - u_tn_grid) / norm(u_dense_grid)
    println("step = $step, rel err = $rel_err, bond dim = $bond_dim")

    if step < steps
        mps = apply(A_mpo, mps; alg="naive", cutoff=cutoff, maxdim=maxdim)
        u_qtt = A_dense_qtt * u_qtt
    end
end
