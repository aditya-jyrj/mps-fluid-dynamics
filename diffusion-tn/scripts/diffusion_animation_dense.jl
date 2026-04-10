include("../src/utils.jl")
using Plots

n = 6
cfl = 0.1
steps = 50

Nx = 2^n
Ny = 2^n

x = range(0, 1, length=Nx+1)[1:end-1]
y = range(0, 1, length=Ny+1)[1:end-1]

u0 = [exp(-50*((xi-0.5)^2 + (yj-0.5)^2)) for xi in x, yj in y]

u_std = grid_to_standard_vector_2d(u0)
A_dense = timestep_operator_2d(Nx, Ny, cfl, cfl)

clims = (0.0, maximum(abs.(u0)))
p = Progress(steps + 1)
anim = @animate for step in 0:steps
    next!(p)

    global u_std
    u = standard_vector_to_grid_2d(u_std, n)

    heatmap(
        x, y, u';
        aspect_ratio = 1,
        xlabel = "x",
        ylabel = "y",
        title = "2D Diffusion, step = $step",
        colorbar = true,
        grid = false,
        size = (800, 800),
        xlims = (0, 1),
        ylims = (0, 1),
        interpolate = true,
        clims = clims
    )

    if step < steps
        u_std = A_dense * u_std
    end
end

gif(anim, joinpath(@__DIR__, "diffusion_dense.gif"), fps=20)
println("Saved animation to ", joinpath(@__DIR__, "diffusion_dense.gif"))