include("utils.jl")

# DEFINE 2D GRID

nx = 2
ny = 2
cfl = 0.1

Nx = 2^nx
Ny = 2^ny

x = range(0, 1, length=Nx+1)[1:end-1]
y = range(0, 1, length=Ny+1)[1:end-1]

u_test = [10*i + j for i in 1:Nx, j in 1:Ny]
println("u_test =")
println(u_test)

u_test_vec = reshape(u_test, :)
println("u_test_vec = ", u_test_vec)

# the ordering convention is (1,1) (2,1) ... (Nx,1) (1,2) (2,2) ... (Nx,2) .. (Nx,Ny)
