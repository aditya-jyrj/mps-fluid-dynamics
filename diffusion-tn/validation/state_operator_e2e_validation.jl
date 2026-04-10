# =============================
# CHECKING STATE REPRESENTATION
# =============================

# QTT MPS reconstruction error = 8.225612988129258e-15
# QTT state vector error = 8.225612988129257e-15



# ===========================
# CHECKING ACTION OF OPERATOR
# ===========================

# operator action error = 3.530747963922594e-13
# relative operator action error = 1.8128352598543243e-14



# ==========================
# CHECK END-TO-END EVOLUTION
# ==========================

# QTT matrix vs TN error after 100 steps = 3.518283956603718e-10
# Relative QTT matrix vs TN error after 100 steps = 1.0659931866145545e-10
# Max bond dim of mps after 100 steps = 8
# Max bond dim of A_mpo = 4



include("../src/utils.jl")

using ITensors, ITensorMPS
using LinearAlgebra

function main()
        ITensors.disable_warn_order()

        n = 5
        cfl = 0.1
        steps = 100
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
        mps0 = grid_to_qtt_mps(u0, sites_2d; cutoff=cutoff)
        u0_back = qtt_mps_to_grid(mps0, sites_2d)
        println("QTT MPS reconstruction error = ", norm(u0 - u0_back))

        # GRID -> QTT VECTOR vs GRID -> QTT MPS -> SITE VECTOR
        u0_qtt_vec = grid_to_qtt_vector(u0, n)
        u0_qtt_vec_from_mps = mps_to_site_vector(mps0, sites_2d)
        println("QTT state vector error = ", norm(u0_qtt_vec - u0_qtt_vec_from_mps))



        println()

        A_mpo = timestep_mpo_2d(sites_2d, cfl)
        A_mat_std = timestep_operator_2d(Nx, Ny, cfl, cfl)
        A_mat_qtt = standard_to_qtt_matrix(A_mat_std, n)

        # =====================
        # OPERATOR ACTION CHECK
        # =====================

        # VEC_1 = MATRIX * VEC_0 
        # vs
        # VEC_0 -> MPS_0
        # MPS_1 = MPO CONTRACTED WITH MPS_0
        # MPS_1 -> VEC_1
        # -> compare both VEC_1
        v = randn(Float64, Nx * Ny)

        w_mat = A_mat_qtt * v

        mps_tmp = site_vector_to_mps(v, sites_2d; cutoff=cutoff)
        mps_tmp = apply(A_mpo, mps_tmp; alg="naive", cutoff=cutoff, maxdim=maxdim)
        w_mpo = mps_to_site_vector(mps_tmp, sites_2d)


        println("operator action error = ", norm(w_mat - w_mpo))
        println("relative operator action error = ",
                norm(w_mat - w_mpo) / norm(w_mat))
        println()


        # ==========================
        # CHECK END-TO-END EVOLUTION
        # ==========================

        # MATRIX TIME EVOLUTION IN QTT/SITE ORDERING
        u0_qtt = grid_to_qtt_vector(u0, n)
        u_qtt = copy(u0_qtt)

        for _ in 1:steps
        u_qtt = A_mat_qtt * u_qtt
        end

        u_grid_mat_qtt = qtt_vector_to_grid(u_qtt, n)

        # TN TIME EVOLUTION
        mps = evolve_mps_with_mpo(mps0, A_mpo, steps; cutoff=cutoff, maxdim=maxdim)
        u_grid_tn = qtt_mps_to_grid(mps, sites_2d)

        println("QTT matrix vs TN error after $steps steps = ",
                norm(u_grid_mat_qtt - u_grid_tn))
        println("Relative QTT matrix vs TN error after $steps steps = ",
                norm(u_grid_mat_qtt - u_grid_tn) / norm(u_grid_mat_qtt))
        println("Max bond dim of mps after $steps steps = ", maxlinkdim(mps))
        println("Max bond dim of A_mpo = ", maxlinkdim(A_mpo))
end

main()