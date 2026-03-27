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


def evolve_mps_timed(mps0, mpoA_list, steps, save_every=50, mps_cutoff=1e-10, max_bond=64):
    mps = mps0.copy()
    saved = []
    bonds = []
    
    for i in range(steps):
        if i % save_every == 0:
            saved.append(mps.copy())
            bonds.append(max(mps.bond_sizes()))

        t0 = time.perf_counter()

        for mpoA in mpoA_list:
            mps = step_mps(mps, mpoA, mps_cutoff, max_bond)

        dt_step = time.perf_counter() - t0
        print(f"step {i:2d}: {dt_step:.6f} s, max bond = {max(mps.bond_sizes())}")

    # save final state
    saved.append(mps.copy())
    bonds.append(max(mps.bond_sizes()))
    return saved, bonds




# ==========================
# MANUAL CONSTRUCTION OF MPO
# ==========================


def qtt_identity_mpo(n):

    W = np.zeros((1, 1, 2, 2))
    W[0, 0] = np.eye(2)
    arrays = [W.copy() for _ in range(n)] 
    return qtn.MatrixProductOperator(arrays, shape='lrud') # l=left, r=right, u=upper/output, d=lower/input


def qtt_shift_plus_mpo(n):
    arrays = []

    # leftmost site: enforce carry_out = 0 (no overflow past the MSB)
    WL = np.zeros((1, 2, 2, 2))   # [carry_out, carry_in, bit_out, bit_in]
    for carry_in in (0, 1):
        for bit_in in (0, 1):
            if carry_in == 0:
                bit_out = bit_in
                carry_out = 0
            else:
                if bit_in == 0:
                    bit_out = 1
                    carry_out = 0
                else:
                    bit_out = 0
                    carry_out = 1

            if carry_out == 0:
                WL[0, carry_in, bit_out, bit_in] = 1.0

    arrays.append(WL)

    # middle sites: propagate carry from right to left
    W = np.zeros((2, 2, 2, 2))    # [carry_out, carry_in, bit_out, bit_in]
    for carry_in in (0, 1):
        for bit_in in (0, 1):
            if carry_in == 0:
                bit_out = bit_in
                carry_out = 0
            else:
                if bit_in == 0:
                    bit_out = 1
                    carry_out = 0
                else:
                    bit_out = 0
                    carry_out = 1

            W[carry_out, carry_in, bit_out, bit_in] = 1.0

    for _ in range(n - 2):
        arrays.append(W.copy())

    # rightmost site: inject carry_in = 1 (add one at the LSB)
    WR = np.zeros((2, 1, 2, 2))   # [carry_out, carry_in, bit_out, bit_in]
    for bit_in in (0, 1):
        if bit_in == 0:
            bit_out = 1
            carry_out = 0
        else:
            bit_out = 0
            carry_out = 1

        WR[carry_out, 0, bit_out, bit_in] = 1.0

    arrays.append(WR)

    return qtn.MatrixProductOperator(arrays, shape='lrud')



def qtt_shift_minus_mpo(n):
    arrays = []

    # leftmost site: enforce borrow_out = 0 (no underflow past the MSB)
    WL = np.zeros((1, 2, 2, 2))   # [borrow_out, borrow_in, bit_out, bit_in]
    for borrow_in in (0, 1):
        for bit_in in (0, 1):
            if borrow_in == 0:
                bit_out = bit_in
                borrow_out = 0
            else:
                if bit_in == 0:
                    bit_out = 1
                    borrow_out = 1
                else:
                    bit_out = 0
                    borrow_out = 0

            if borrow_out == 0:
                WL[0, borrow_in, bit_out, bit_in] = 1.0

    arrays.append(WL)

    # middle sites: propagate borrow from right to left
    W = np.zeros((2, 2, 2, 2))    # [borrow_out, borrow_in, bit_out, bit_in]
    for borrow_in in (0, 1):
        for bit_in in (0, 1):
            if borrow_in == 0:
                bit_out = bit_in
                borrow_out = 0
            else:
                if bit_in == 0:
                    bit_out = 1
                    borrow_out = 1
                else:
                    bit_out = 0
                    borrow_out = 0

            W[borrow_out, borrow_in, bit_out, bit_in] = 1.0

    for _ in range(n - 2):
        arrays.append(W.copy())

    # rightmost site: inject borrow_in = 1 (subtract one at the LSB)
    WR = np.zeros((2, 1, 2, 2))   # [borrow_out, borrow_in, bit_out, bit_in]
    for bit_in in (0, 1):
        if bit_in == 0:
            bit_out = 1
            borrow_out = 1
        else:
            bit_out = 0
            borrow_out = 0

        WR[borrow_out, 0, bit_out, bit_in] = 1.0

    arrays.append(WR)

    return qtn.MatrixProductOperator(arrays, shape='lrud')



def qtt_diffusion_mpo(n, cfl, cutoff=1e-12, max_bond=64):
    I  = qtt_identity_mpo(n)
    Sp = qtt_shift_plus_mpo(n)
    Sm = qtt_shift_minus_mpo(n)

    A = (1.0 - 2.0 * cfl) * I + cfl * Sp + cfl * Sm
    A.compress(cutoff=cutoff, max_bond=max_bond)
    return A