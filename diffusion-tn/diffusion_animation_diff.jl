include("utils.jl")

using ITensors, ITensorMPS
using ProgressMeter
using Plots

ITensors.disable_warn_order()

n = 6
cfl = 0.1
steps = 50
cutoff = 1e-20
maxdim = 512

Nx = 2^n
Ny = 2^n

x = range(0, 1, length=Nx+1)[1:end-1]
y = range(0, 1, length=Ny+1)[1:end-1]

u0 = [exp(-50*((xi-0.5)^2 + (yj-0.5)^2)) for xi in x, yj in y]

sites2d = siteinds("S=1/2", 2n)

# Build initial MPS and timestep MPO
mps = grid_to_grouped_mps_2d(u0, sites2d; cutoff=cutoff)
A_mpo = timestep_mpo_2d(sites2d, cfl)

u_std = grid_to_standard_vector_2d(u0)
A_dense = timestep_operator_2d(Nx, Ny, cfl, cfl)

p = Progress(steps + 1)
anim = @animate for step in 0:steps
    next!(p)

    global mps
    global u_std

    u_tn = grouped_mps_to_grid_2d(mps, sites2d)
    u_dense = standard_vector_to_grid_2d(u_std, n)

    err = u_dense - u_tn
    abs_err = norm(err)
    rel_err = abs_err / norm(u_dense)


    heatmap(
        x, y, err';
        aspect_ratio = 1,
        xlabel = "x",
        ylabel = "y",
        title = "Error field, step = $step, rel err = $(round(rel_err, sigdigits=3))",
        colorbar = true,
        grid = false,
        size = (800, 800),
        xlims = (0, 1),
        ylims = (0, 1),
        interpolate = true,
    )

    if step < steps
        mps = apply(A_mpo, mps; alg="naive", cutoff=cutoff, maxdim=maxdim)
        u_std = A_dense * u_std
    end
end

gif(anim, joinpath(@__DIR__, "diffusion_error.gif"), fps=20)
println("Saved animation to ", joinpath(@__DIR__, "diffusion_error.gif"))