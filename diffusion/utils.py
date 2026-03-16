import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import quimb.tensor as qtn
import time

# ========================
# OPERATOR-BASED EVOLUTION
# ========================

# laplacian_1d returns a laplacian matrix, calling one of the following 4 helper functions
# depending on whether boundary condition is dirichlet or periodic, and if the format required
# is a dense matrix (ie all 0s are stored) or sparse matrix

def laplacian_1d(N, dx, bc="dirichlet", fmt="dense"):
    if bc == "periodic" and fmt == "dense":
        return laplacian_1d_dense_periodic(N, dx)
    elif bc == "dirichlet" and fmt == "dense":
        return laplacian_1d_dense_dirichlet(N, dx)
    elif bc == "periodic" and fmt == "sparse":
        return laplacian_1d_sparse_periodic(N, dx)
    elif bc == "dirichlet" and fmt == "sparse":
        return laplacian_1d_sparse_dirichlet(N, dx)
    else:
        raise ValueError("bc must be 'periodic' or 'dirichlet', fmt must be 'dense' or 'sparse'")

def laplacian_1d_dense_periodic(N, dx):
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = -2.0
        L[i, (i - 1) % N] = 1.0
        L[i, (i + 1) % N] = 1.0
    return L / (dx * dx)

def laplacian_1d_dense_dirichlet(N, dx):
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = -2.0
        if i > 0:
            L[i, i-1] = 1.0
        if i < N-1:
            L[i, i+1] = 1.0
    return L / (dx * dx)

def laplacian_1d_sparse_periodic(N, dx):
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

def laplacian_1d_sparse_dirichlet(N, dx):
    main = -2.0 * np.ones(N)
    off  = 1.0 * np.ones(N - 1)
    L = sp.diags(
        [off, main, off],
        offsets=[-1, 0, 1],
        shape=(N, N),
        format="csr"
    )
    return L / (dx * dx)




# the following two functions, evolve_operator_lpe2 and evolve_operator_euler, both execute time evolution
# save_every defines how frequently we want to save snapshots of the evolution (eg every 50 timestep advancements)
# based on this, both functions return three numpy arrays:
# 1. np.array(times): the time = steps * dt at each snapshot
# 2. np.array(saved): the function at the time of the snapshot, discretised into a vector of N elements separated by width dx
# 3. np.array(norms): the euclidean norm of the vector at each snapshot (helps to compare error) 

def evolve_operator_lpe2(u0, steps, A1, A2, dt, save_every=50):
    u = u0.copy()

    saved = []
    times = []
    norms = []

    t = 0.0
    
    for i in range(steps):
        # save current state
        if i % save_every == 0:
            saved.append(u.copy())
            times.append(t)
            norms.append(np.linalg.norm(u))
    
        # advance one timestep
        u = A1 @ u
        u = A2 @ u
        t += dt
    
    # save final state
    saved.append(u.copy())
    times.append(t)
    norms.append(np.linalg.norm(u))
    
    return np.array(times), np.array(saved), np.array(norms)


def evolve_operator_euler(u0, steps, A, dt, save_every=50):
    u = u0.copy()

    saved = []
    times = []
    norms = []

    t = 0.0
    
    for i in range(steps):
        # save current state
        if i % save_every == 0:
            saved.append(u.copy())
            times.append(t)
            norms.append(np.linalg.norm(u))
    
        # advance one timestep
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

# these helper functions convert from vectors to MPS and vice versa,
# as well as from matrices to MPOs

def vec_to_qtt_mps(u, n, cutoff=1e-10, max_bond=64):
    T = u.reshape((2,) * n)
    mps = qtn.MatrixProductState.from_dense(T, cutoff=cutoff, max_bond=max_bond)
    return mps

def qtt_mps_to_vec(mps):
    T = mps.to_dense()
    return np.asarray(T).reshape(-1)

def mat_to_qtt_mpo(A, n, cutoff=1e-12, max_bond=256):
    return qtn.MatrixProductOperator.from_dense(
        A, dims=[2] * n, cutoff=cutoff, max_bond=max_bond
    )


# these functions are for the case where we create the MPO manually

def vec_to_qtt_mps_reverse(u, n, cutoff=1e-10, max_bond=64):
    T = np.asarray(u).reshape((2,) * n)
    T = T.transpose(tuple(range(n - 1, -1, -1)))
    # reverses MPS direction such that least significant bit is first: easier for shift MPOs later
    return qtn.MatrixProductState.from_dense(T, cutoff=cutoff, max_bond=max_bond)

def qtt_mps_to_vec_unreverse(mps):
    T = np.asarray(mps.to_dense())
    n = T.ndim
    T = T.transpose(tuple(range(n - 1, -1, -1)))
    # undo MPS direction reversal
    return T.reshape(-1)

def qtt_identity_mpo(n):

    W = np.zeros((1, 1, 2, 2))
    # creates a 4-dimensional tensor. \chi_left = \chi_right = 1, and input and output physical dims = 2
    # bond dims = 0 since each site acts independently

    W[0, 0] = np.eye(2)
    # fills the slice W[0,0,:,:] with an identity matrix, so now,
    # W[0,0,0,0] = 1
    # W[0,0,0,1] = 0
    # W[0,0,1,0] = 0
    # W[0,0,1,1] = 1
    # ie 0 goes to 0, 1 goes to 1

    arrays = [W.copy() for _ in range(n)] # duplicate n such tensors
    return qtn.MatrixProductOperator(arrays, shape='lrud') # l=left, r=right, u=upper/output, d=lower/input

def qtt_shift_plus_mpo(n):
    # we want an MPO that represents |i> -> |i+1>, except at the ends of course
    # this means the MPO looks like |x0 x1 x2 ... xn> -> |x0 x1 x2 ... xn> + 1, where each xk is 0 or 1, and the + is binary addition


    # so for example if we have 0101 + 1, we iterate through each bit starting from the least significant bit, ie rightmost
    # first bit: 1 + 1 = 0 with carry = 1

    # second bit: because carry = 1, we add 0 + 1 = 1
    # since there is no overflow, the carry resets to 0

    # third bit: carry = 0 so the bit remains unchanged
    # therefore all higher bits also remain unchanged


    # in MPS language, each site represents one bit, so 0101 corresponds to the state |0101>

    # the MPO implements this binary addition locally at each site
    # each tensor maps (input bit, carry_in) -> (output bit, carry_out)

    # the bond index stores the carry bit, so the bond dimension is 2
    # corresponding to carry = 0 or carry = 1

    # because vec_to_qtt_mps() reverses the ordering of bits,
    # the least significant bit is placed at the leftmost site

    # therefore the left boundary injects carry = 1 (adding one),
    # while the right boundary enforces carry = 0 (preventing overflow)


    arrays = []

    # first site: incoming carry fixed to 1
    W0 = np.zeros((1, 2, 2, 2))
    # [carry_in, carry_out, bit_out, bit_in]
    # leftmost site (corresponding to least significant bit) has left bond dim = 1 (carry_in must = 1) and right = 2 (one for carry=0, the other for carry=1)
    # input has two dimensions to receive the original bit and output has two dimensions to represent the result bit
    for bit_in in (0, 1):
        if bit_in == 0:
            # 0 + 1 = 1, carry = 0
            bit_out = 1
            carry = 0
        else:
            # 1 + 1 = 0 with carry
            bit_out = 0
            carry = 1
        W0[0, carry, bit_out, bit_in] = 1.0
    arrays.append(W0)

    # middle sites: propagate carry
    W = np.zeros((2, 2, 2, 2))  # [carry_in, carry_out, bit_out, bit_in]

    for carry_in in (0, 1):
        for bit_in in (0, 1):
            if carry_in == 0:
                # no carry coming in
                bit_out = bit_in
                carry_out = 0
            else:
                # carry_in = 1
                if bit_in == 0:
                    # 0 + 1 = 1
                    bit_out = 1
                    carry_out = 0
                else:
                    # 1 + 1 = 0 with carry
                    bit_out = 0
                    carry_out = 1
            W[carry_in, carry_out, bit_out, bit_in] = 1.0
    
    # actually this part can be expressed far more concisely as
    # bit_out   = bit_in ^ carry_in   (^ represents XOR in python)
    # carry_out = bit_in & carry_in   (& represents AND in python)

    for _ in range(n - 2):
        arrays.append(W.copy())

    # last site
    WN = np.zeros((2, 1, 2, 2))  # [carry_in, carry_out, bit_out, bit_in]

    for carry_in in (0, 1):
        for bit_in in (0, 1):
            if carry_in == 0:
                # no carry coming in
                bit_out = bit_in
                carry_out = 0
            else:
                # carry_in = 1
                if bit_in == 0:
                    # 0 + 1 = 1
                    bit_out = 1
                    carry_out = 0
                else:
                    # 1 + 1 = 0 with carry
                    bit_out = 0
                    carry_out = 1
            # only allow transitions where no overflow occurs
            if carry_out == 0:
                WN[carry_in, 0, bit_out, bit_in] = 1.0

    arrays.append(WN)

    return qtn.MatrixProductOperator(arrays, shape='lrud')


def qtt_shift_minus_mpo(n):
    arrays = []

    # first site
    W0 = np.zeros((1, 2, 2, 2)) # [borrow_in, borrow_out, bit_out, bit_in]
    # at the first site, borrow_in is fixed to 1 because we are subtracting 1
    for bit_in in (0, 1):
        if bit_in == 0:
            # 0 - 1 = 1 with borrow
            bit_out = 1
            borrow_out = 1
        else:
            # 1 - 1 = 0, no borrow
            bit_out = 0
            borrow_out = 0
        W0[0, borrow_out, bit_out, bit_in] = 1.0

    arrays.append(W0)

    # middle sites
    W = np.zeros((2, 2, 2, 2)) # [borrow_in, borrow_out, bit_out, bit_in]

    for borrow_in in (0, 1):
        for bit_in in (0, 1):

            if borrow_in == 0:
                # no borrow coming in, so the bit stays unchanged
                bit_out = bit_in
                borrow_out = 0

            else:
                # borrow_in = 1
                if bit_in == 0:
                    # 0 - 1 = 1 with borrow
                    bit_out = 1
                    borrow_out = 1
                else:
                    # 1 - 1 = 0, no further borrow
                    bit_out = 0
                    borrow_out = 0

            W[borrow_in, borrow_out, bit_out, bit_in] = 1.0

    for _ in range(n - 2):
        arrays.append(W.copy())

    # last site
    # at the final site we only allow borrow_out = 0
    WN = np.zeros((2, 1, 2, 2)) # [borrow_in, borrow_out, bit_out, bit_in]

    for borrow_in in (0, 1):
        for bit_in in (0, 1):

            if borrow_in == 0:
                # no borrow coming in, so the bit stays unchanged
                bit_out = bit_in
                borrow_out = 0

            else:
                # borrow_in = 1
                if bit_in == 0:
                    # 0 - 1 = 1 with borrow
                    bit_out = 1
                    borrow_out = 1
                else:
                    # 1 - 1 = 0, no further borrow
                    bit_out = 0
                    borrow_out = 0

            # only allow transitions with no underflow
            if borrow_out == 0:
                WN[borrow_in, 0, bit_out, bit_in] = 1.0
    
    arrays.append(WN)
    return qtn.MatrixProductOperator(arrays, shape='lrud')

def qtt_diffusion_mpo(n, alpha, cutoff=1e-12, max_bond=64):
    I  = qtt_identity_mpo(n)
    Sp = qtt_shift_plus_mpo(n)
    Sm = qtt_shift_minus_mpo(n)

    A = (1.0 - 2.0 * alpha) * I + alpha * Sp + alpha * Sm
    A.compress(cutoff=cutoff, max_bond=max_bond)
    return A

# ====================
# TIME EVOLUTION IN TN
# ====================

def step_mps(mps, mpo, cutoff=1e-10, max_bond=64):
    mps_new = mpo.apply(mps)
    mps_new.compress(cutoff=cutoff, max_bond=max_bond)
    return mps_new

def evolve_mps(mps0, mpoA, steps, save_every=50, cutoff=1e-10, max_bond=64):
    mps = mps0.copy()
    saved = []
    bonds = []
    
    for i in range(steps):
        if i % save_every == 0:
            saved.append(mps.copy())
            bonds.append(max(mps.bond_sizes()))
    
        mps = step_mps(mps, mpoA, cutoff, max_bond)
    
    # save final state
    saved.append(mps.copy())
    bonds.append(max(mps.bond_sizes()))
    return saved, bonds



# =======================================
# WRAPPER FUNCTIONS TO MEASURE TIME TAKEN
# =======================================

def time_initialisation_complex(N, dx, dt, nu):
    t0 = time.perf_counter()
    L = laplacian_1d_dense_periodic(N, dx).astype(np.complex128)

    a1 = (1 - 1j) / 2
    a2 = (1 + 1j) / 2
    A1 = np.eye(N, dtype=np.complex128) + a1 * dt * nu * L
    A2 = np.eye(N, dtype=np.complex128) + a2 * dt * nu * L

    t1 = time.perf_counter()
    return A1, A2, t1 - t0

def time_initialisation(N, dx, dt, nu):
    t0 = time.perf_counter()
    L = laplacian_1d_dense_periodic(N, dx)
    A = np.eye(N, dtype=np.complex128) + dt * nu * L
    t1 = time.perf_counter()
    return L, A, t1 - t0

def time_mps_construction(u0, n, cutoff, max_bond):
    t0 = time.perf_counter()
    mps0 = vec_to_qtt_mps(u0, n, cutoff, max_bond)
    t1 = time.perf_counter()
    return mps0, t1 - t0

def time_mps_construction(u0, n, cutoff, max_bond):
    t0 = time.perf_counter()
    mps0 = vec_to_qtt_mps_reverse(u0, n, cutoff, max_bond)
    t1 = time.perf_counter()
    return mps0, t1 - t0

def time_mpo_construction(A, n, mpo_cutoff, mpo_max_bond):
    t0 = time.perf_counter()
    mpoA = mat_to_qtt_mpo(A, n, mpo_cutoff, mpo_max_bond)
    t1 = time.perf_counter()
    return mpoA, t1 - t0

def time_mpo_construction(n, alpha, mpo_cutoff, mpo_max_bond):
    t0 = time.perf_counter()
    mpoA = qtt_diffusion_mpo(n, alpha, cutoff=mpo_cutoff, max_bond=mpo_max_bond)
    t1 = time.perf_counter()
    return mpoA, t1 - t0

def time_evolve_dense(u0, steps, A, dt, save_every=50):
    t0 = time.perf_counter()
    times, us, norms = evolve_dense(u0, steps, A, dt, save_every)
    t1 = time.perf_counter()
    return times, us, norms, t1 - t0

def time_evolve_dense_complex(u0, steps, A1, A2, dt, save_every=50):
    t0 = time.perf_counter()
    times, us, norms = evolve_dense_complex(u0, steps, A1, A2, dt, save_every)
    t1 = time.perf_counter()
    return times, us, norms, t1 - t0

def time_evolve_mps(mps0, mpoA, steps, save_every=50, cutoff=1e-10, max_bond=64):
    t0 = time.perf_counter()
    mps_saved, bond_track = evolve_mps(mps0, mpoA, steps, save_every, cutoff, max_bond)
    t1 = time.perf_counter()
    return mps_saved, bond_track, t1 - t0