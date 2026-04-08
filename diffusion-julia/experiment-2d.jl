include("utils.jl")

# DEFINE 2D GRID

nx = 2
ny = 2
cfl = 0.1

Nx = 2^nx
Ny = 2^ny

x = range(0, 1, length=Nx+1)[1:end-1]
y = range(0, 1, length=Ny+1)[1:end-1]

# INITIAL CONDITIONS

u = [sin(2π * xi) * sin(2π * yj) for xi in x, yj in y] 
u_vec = reshape(u, :) # picks the convention (1,1), (2,1), ... (Nx,1), (1,2), (2,2), ... (Nx,2), .. (Nx,Ny)


A = A_exact_2d(Nx, Ny, cfl, cfl)
u_next = A * u_vec

n = nx + ny   # ensure integer
sites = siteinds("S=1/2", n)

mps = dense_to_qtt_mps(u_vec, sites)

u_next_grid = reshape(u_next, Nx, Ny)
println("u_next reshaped =")
println(u_next_grid)

u_from_mps = qtt_mps_to_dense(mps, sites)
println("MPS reconstruction error = ", norm(u_vec - u_from_mps))