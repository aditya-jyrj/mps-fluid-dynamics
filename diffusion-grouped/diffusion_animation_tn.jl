include("utils.jl")

using ITensors, ITensorMPS
using ProgressMeter
using Plots

ITensors.disable_warn_order()

n = 6
cfl = 0.1
steps = 50
cutoff = 1e-30
maxdim = 1000

Nx = 2^n
Ny = 2^n

x = range(0, 1, length=Nx+1)[1:end-1]
y = range(0, 1, length=Ny+1)[1:end-1]

u0 = [((0.3 < xi < 0.7) && (0.3 < yj < 0.7)) ? 1.0 : 0.0
      for xi in x, yj in y]

sites2d = siteinds("S=1/2", 2n)

# Build initial MPS and timestep MPO
mps = grid_to_grouped_mps_2d(u0, sites2d; cutoff=cutoff)
A_mpo = timestep_mpo_2d(sites2d, cfl)

# Warm-up apply
apply(A_mpo, mps; alg="naive", cutoff=cutoff, maxdim=maxdim)

clims = (0.0, maximum(abs.(u0)))
p = Progress(steps + 1)
anim = @animate for step in 0:steps
    next!(p)

    global mps
    u = grouped_mps_to_grid_2d(mps, sites2d)

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
        mps = apply(A_mpo, mps; alg="naive", cutoff=cutoff, maxdim=maxdim)
    end
end

gif(anim, joinpath(@__DIR__, "diffusion_tn.gif"), fps=10)
println("Saved animation to ", joinpath(@__DIR__, "diffusion_tn.gif"))