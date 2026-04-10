include("utils.jl")

using ITensors, ITensorMPS
using ProgressMeter
using Plots

ITensors.disable_warn_order()

n = 6
cfl = 0.1
steps = 2000
cutoff = 1e-12
maxdim = 128

Nx = 2^n
Ny = 2^n

x = range(0, 1, length=Nx+1)[1:end-1]
y = range(0, 1, length=Ny+1)[1:end-1]

u0 = [sin(4π * xi) * sin(4π * yj) +
      0.5 * sin(8π * xi + 2π * yj)
      for xi in x, yj in y]

sites2d = siteinds("S=1/2", 2n)

# Build initial MPS and timestep MPO
mps = grid_to_qtt_mps_2d(u0, sites2d; cutoff=cutoff)
A_mpo = timestep_mpo_2d(sites2d, cfl)

# Warm-up apply
apply(A_mpo, mps; alg="naive", cutoff=cutoff, maxdim=maxdim)

p = Progress(steps + 1)
anim = @animate for step in 0:steps
    next!(p)

    global mps
    u = qtt_mps_to_grid_2d(mps, sites2d)

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
        c=:balance
    )

    if step < steps
        mps = apply(A_mpo, mps; alg="naive", cutoff=cutoff, maxdim=maxdim)
    end
end

mp4(anim, joinpath(@__DIR__, "diffusion_tn.mp4"), fps=200)
println("Saved animation to ", joinpath(@__DIR__, "diffusion_tn.gif"))