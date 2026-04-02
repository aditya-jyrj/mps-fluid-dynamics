# MPS parameters
n = 5
steps = 10
N = 2^n

# Mesh parameters
nu = 1e-3
cfl = 0.1
x = range(0, 1, length=N+1)[1:end-1]
dx = x[2] - x[1]
dt = cfl * dx^2 / nu

sites = siteinds("S=1/2", n)

# Initial state sampled at x
u0 = @. sin(2 * pi * 2 * x) + 0.5 * sin(2 * pi * 7 * x) # same initial condition as in the notebook

mps0 = dense_to_qtt_mps(u0, sites)

# Compare the A obtained via MPO and exact matrix
A_mpo_network = A_mpo(sites, cfl)
A_mpo_mat = mpo_to_matrix(A_mpo_network, sites)

A_exact_mat = A_exact(N, cfl, :dirichlet)

diff = norm(A_mpo_mat - A_exact_mat)
println("Max difference between A_mpo_mat and A_exact_mat: ", diff)

final_mps = evolve_mps(mps0, A_mpo_network, steps)