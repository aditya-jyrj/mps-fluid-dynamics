import numpy as np
import scipy.sparse as sp
import quimb.tensor as qtn
import time

# ========================
# OPERATOR-BASED EVOLUTION
# ========================

# laplacian_1d returns a laplacian matrix, calling one of the following 4 helper functions
# depending on whether boundary condition is dirichlet or periodic, and if the format required
# is a dense matrix (ie all 0s are stored) or sparse matrix

def laplacian(N, dx, bc="dirichlet", fmt="dense"):
    if bc == "periodic" and fmt == "dense":
        return laplacian_dense_periodic(N, dx)
    elif bc == "dirichlet" and fmt == "dense":
        return laplacian_dense_dirichlet(N, dx)
    elif bc == "periodic" and fmt == "sparse":
        return laplacian_sparse_periodic(N, dx)
    elif bc == "dirichlet" and fmt == "sparse":
        return laplacian_sparse_dirichlet(N, dx)
    else:
        raise ValueError("bc must be 'periodic' or 'dirichlet', fmt must be 'dense' or 'sparse'")

def laplacian_dense_periodic(N, dx):
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = -2.0
        L[i, (i - 1) % N] = 1.0
        L[i, (i + 1) % N] = 1.0
    return L / (dx * dx)

def laplacian_dense_dirichlet(N, dx):
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = -2.0
        if i > 0:
            L[i, i-1] = 1.0
        if i < N-1:
            L[i, i+1] = 1.0
    return L / (dx * dx)

def laplacian_sparse_periodic(N, dx):
    main = -2.0 * np.ones(N)
    off  = 1.0 * np.ones(N - 1)
    wrap = 1.0 * np.ones(1)
    L = sp.diags(
        [wrap, off, main, off, wrap],
        offsets=[-(N - 1), -1, 0, 1, N - 1],
        shape=(N, N),
        format="csr"
    )
    return L / (dx * dx)

def laplacian_sparse_dirichlet(N, dx):
    main = -2.0 * np.ones(N)
    off  = 1.0 * np.ones(N - 1)
    L = sp.diags(
        [off, main, off],
        offsets=[-1, 0, 1],
        shape=(N, N),
        format="csr"
    )
    return L / (dx * dx)



# the following function returns a list of the time-step matrices 
# using linear product expansion, where first-order corresponds to
# forward euler step. 

def time_step(laplacian, order, dt, nu):
    L = laplacian
    
    coeffs = {
        1: [1.0],
        2: [(1.0 - 1.0j) / 2.0, (1.0 + 1.0j) / 2.0],
        3: [0.6265,
            0.1867 - 0.4808j,
            0.1867 + 0.4808j],
        4: [0.0426 - 0.3946j,
            0.0426 + 0.3946j,
            0.4573 - 0.2351j,
            0.4573 + 0.2351j],
    }

    if order not in coeffs:
        raise ValueError("Supported orders are 1, 2, 3, 4.")
    
    is_sparse = sp.issparse(L)
    N = L.shape[0]

    dtype = np.complex128 if order >= 2 else np.float64

    if is_sparse:
        I = sp.eye(N, dtype=dtype)
    else:
        I = np.eye(N, dtype=dtype)

    L = L.astype(dtype)

    return [I + a * dt * nu * L for a in coeffs[order]]


def delta_t(cfl, dx, nu):
    return cfl * dx * dx / nu

# the following function executes time evolution
# save_every defines how frequently we want to save snapshots of the evolution (eg every 50 timestep advancements)
# based on this, the function returns three numpy arrays:
# 1. np.array(times): the time = steps * dt at each snapshot
# 2. np.array(saved): the function at the time of the snapshot, discretised into a vector of N elements separated by width dx
# 3. np.array(norms): the euclidean norm of the vector at each snapshot (helps to compare error) 


def evolve_operator(u0, steps, A_list, dt, save_every=50):
    u = u0.copy()

    saved = []
    times = []
    norms = []

    t = 0.0

    for i in range(steps):
        if i % save_every == 0:
            # save current state
            saved.append(u.copy())
            times.append(t)
            norms.append(np.linalg.norm(u))
        
        for A in A_list:
            u = A @ u
        t += dt
    
    # save final state
    saved.append(u.copy())
    times.append(t)
    norms.append(np.linalg.norm(u))
    
    return np.array(times), np.array(saved), np.array(norms)



# ==================
# MPS/MPO GENERATION
# ==================

# these functions permit the following conversions:
# 1. vector <-> MPS
# 2. matrix  -> MPO

def vec_to_qtt_mps(u, n, mps_cutoff=1e-10, max_bond=64):
    T = np.asarray(u).reshape((2,) * n)
    return qtn.MatrixProductState.from_dense(T, cutoff=mps_cutoff, max_bond=max_bond)


def qtt_mps_to_vec(mps):
    T = np.asarray(mps.to_dense())
    return T.reshape(-1)


def mat_to_qtt_mpo(A, n, mpo_cutoff=1e-12, max_bond=256):
    return qtn.MatrixProductOperator.from_dense(
        A, dims=[2] * n, cutoff=mpo_cutoff, max_bond=max_bond
    )

def mats_to_qtt_mpos(A_list, n, mpo_cutoff=1e-12, max_bond=256):
    return [
        mat_to_qtt_mpo(A, n, mpo_cutoff=mpo_cutoff, max_bond=max_bond)
        for A in A_list
    ]



# ====================
# TIME EVOLUTION IN TN
# ====================

def step_mps(mps, mpo, mps_cutoff=1e-10, max_bond=64):
    mps_new = mpo.apply(mps)
    mps_new.compress(cutoff=mps_cutoff, max_bond=max_bond)
    return mps_new

def evolve_mps(mps0, mpoA_list, steps, save_every=50, mps_cutoff=1e-10, max_bond=64):
    mps = mps0.copy()
    saved = []
    bonds = []
    
    for i in range(steps):
        if i % save_every == 0:
            saved.append(mps.copy())
            bonds.append(max(mps.bond_sizes()))
    
        for mpoA in mpoA_list:
            mps = step_mps(mps, mpoA, mps_cutoff, max_bond)
    
    # save final state
    saved.append(mps.copy())
    bonds.append(max(mps.bond_sizes()))
    return saved, bonds


# the following functions are for time evolution but measure and print time taken and bond dimension at each step

def step_mps_profiled(mps, mpo, mps_cutoff=1e-10, max_bond=64):
    t0 = time.perf_counter()
    mps_new = mpo.apply(
        mps,
        compress=True,
        cutoff=mps_cutoff,
        max_bond=max_bond
    )
    t_apply = time.perf_counter() - t0

    t1 = time.perf_counter()
    t_compress = time.perf_counter() - t1

    return mps_new, t_apply, t_compress

def evolve_mps_timed(mps0, mpoA_list, steps, save_every=50, mps_cutoff=1e-10, max_bond=64):
    mps = mps0.copy()
    saved = []
    bonds = []
    times = []
    
    for i in range(steps):
        if i % save_every == 0:
            saved.append(mps.copy())
            bonds.append(max(mps.bond_sizes()))

        step_apply = 0.0
        step_compress = 0.0

        for mpoA in mpoA_list:
            mps, t_apply, t_compress = step_mps_profiled(mps, mpoA, mps_cutoff, max_bond)
            step_apply += t_apply
            step_compress += t_compress

        times.append({
        "step": i,
        "apply": step_apply,
        "compress": step_compress,
        "total": step_apply + step_compress,
        "bond": max(mps.bond_sizes())
        })

    print("st | apply  |compress| total  | bond")
    print("----------------------------------------")
    for t in times:
        print(
            f"{t['step']:2d} | "
            f"{t['apply']:.4f} | "
            f"{t['compress']:.4f} | "
            f"{t['total']:.4f} | "
            f"{t['bond']:3d}"
        )

    # save final state
    saved.append(mps.copy())
    bonds.append(max(mps.bond_sizes()))
    return saved, bonds