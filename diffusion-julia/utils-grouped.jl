using ITensors, ITensorMPS
using LinearAlgebra, Printf

# =============
# PDE OPERATORS
# =============

function laplacian_1d(N::Int, bc::Symbol=:dirichlet)
    v = ones(N)
    L = diagm(0 => 2*v, -1 => -v[1:N-1], 1 => -v[1:N-1])    
    
    if bc == :dirichlet
        L[1, 1] = 2.0
        L[N, N] = 2.0
    elseif bc == :neumann
        L[1, 1] = 1.0
        L[N, N] = 1.0
    elseif bc == :periodic
        L[1, N] = -1.0
        L[N, 1] = -1.0
    else
        throw(ArgumentError("Invalid boundary condition: $bc"))
    end
    
    return L
end

function laplacian_2d(Nx::Int, Ny::Int; bcx::Symbol=:dirichlet, bcy::Symbol=:dirichlet)
    Lx = laplacian_1d(Nx, bcx)
    Ly = laplacian_1d(Ny, bcy)

    Ix = Matrix(I, Nx, Nx)
    Iy = Matrix(I, Ny, Ny)

    return kron(Lx, Iy) + kron(Ix, Ly)
end



function timestep_operator_1d(N::Int, cfl::Float64, bc::Symbol=:dirichlet)
    return I - cfl * laplacian_1d(N, bc)
end

function timestep_operator_2d(Nx::Int, Ny::Int, cflx::Float64, cfly::Float64;
                    bcx::Symbol=:dirichlet, bcy::Symbol=:dirichlet)
    Lx = laplacian_1d(Nx, bcx)
    Ly = laplacian_1d(Ny, bcy)

    Ix = Matrix(I, Nx, Nx)
    Iy = Matrix(I, Ny, Ny)

    return Matrix(I, Nx*Ny, Nx*Ny) - cflx * kron(Lx, Iy) - cfly * kron(Ix, Ly)
end



# ================================
# GENERIC DENSE <-> TN CONVERSIONS
# ================================

function mps_to_dense_vector(mps::MPS, sites::Vector{<:Index})
    T = prod(mps) # contract entire mps
    C = combiner(reverse(sites)...) # reverse site ordering, then combine all into one index
    Tc = T * C
    return Array(Tc, combinedind(C))
end

function dense_vector_to_mps(v::AbstractVector, sites::Vector{<:Index}; cutoff=1e-10)
    n = length(sites)
    length(v) == 2^n || throw(ArgumentError("Expected vector of length $(2^n), got $(length(v))"))

    T = reshape(v, ntuple(_ -> 2, n)...)
    IT = ITensor(T, reverse(sites)...)
    return MPS(IT, sites; cutoff=cutoff)
end


function mpo_to_dense_matrix(M::MPO, sites::Vector{<:Index})
    T = prod(M)
    
    # REVERSE the sites so sites[n] varies fastest (Julia follows column-major convention unlike Python)
    C_row = combiner(reverse(prime.(sites))...)
    C_col = combiner(reverse(sites)...)
    
    Tc = T * C_row * C_col
    return Array(Tc, combinedind(C_row), combinedind(C_col))
end


function dense_matrix_to_mpo(A::AbstractMatrix, sites::Vector{<:Index}; cutoff=1e-12)
    n = length(sites)
    N = 2^n
    size(A) == (N, N) || throw(ArgumentError("Expected $(N)×$(N) matrix, got $(size(A))"))

    # reshape matrix into tensor with row bits and column bits
    A_tensor = reshape(A, ntuple(_ -> 2, 2 * n)...)

    row_inds = reverse(prime.(sites))
    col_inds = reverse(sites)

    T = ITensor(A_tensor, row_inds..., col_inds...)
    return MPO(T, sites; cutoff=cutoff)
end

# ============================
# INTERLEAVED 2D BASIS HELPERS
# ============================

# -------------- GRID --> INTERLEAVED -------------- 

function int_to_bits_msb(k::Int, nbits::Int)
    # converts numbers to binary digits with most significant bit coming first
    # k should run from 0 to 2^nbits - 1

    # digits(2, base=2) == [0,1], but digits(2, base=2, pad=4) = [0,1,0,0] (ensures that all bitvectors are of same length)
    # note that julia returns least significant bit first, ie [0,1,0,0] instead of [0,0,1,0]. we have to reverse this

    ds = digits(k, base=2, pad=nbits) 
    return reverse(ds) 
end

function interleave_bits(xbits::Vector{Int}, ybits::Vector{Int})
    # takes in an input x bitvector and y bitvector and interleaves them
    # by creating an output vector of length 2n and filling odd (2k-1) indices with x and even (2k) with y

    n = length(xbits)
    length(ybits) == n || throw(ArgumentError("xbits and ybits must have same length"))
    out = Vector{Int}(undef, 2 * n)
    for k in 1:n
        out[2k - 1] = xbits[k]
        out[2k]     = ybits[k]
    end
    return out
end

function grid_to_interleaved_tensor_2d(u::AbstractMatrix, n::Int)
    Nx, Ny = size(u)

    Nx == 2^n || throw(ArgumentError("Expected Nx = 2^n = $(2^n), got Nx = $Nx"))
    Ny == 2^n || throw(ArgumentError("Expected Ny = 2^n = $(2^n), got Ny = $Ny"))

    T = zeros(eltype(u), ntuple(_ -> 2, 2 * n)) # rank-2n tensor, each dimension has size 2. each element is 0 of same type as u

    # index through all the x and y indices, converting both to binary and interweaving them
    for ix in 0:Nx-1
        xbits = int_to_bits_msb(ix, n)
        for iy in 0:Ny-1
            ybits = int_to_bits_msb(iy, n)
            bits = interleave_bits(xbits, ybits)

            inds = Tuple(b + 1 for b in bits) # increment everything by 1 since julia is 1-indexed
            T[inds...] = u[ix + 1, iy + 1]
        end
    end

    return T
end

function grid_to_interleaved_mps_2d(u::AbstractMatrix, sites::Vector{<:Index}; cutoff=1e-10)
    nsites = length(sites) # nsites is total number of bits, ie num x bits + num y bits
    iseven(nsites) || throw(ArgumentError("Need an even number of sites for interleaved 2D QTT"))
    n = nsites ÷ 2 # ÷ returns integer, / returns float. this is the number of bits per dimension

    T = grid_to_interleaved_tensor_2d(u, n)
    IT = ITensor(T, reverse(sites)...)
    return MPS(IT, sites; cutoff=cutoff)
end

# -------------- INTERLEAVED --> GRID -------------- 

# function bits_msb_to_int(bits::AbstractVector{<:Integer})
#     x = 0
#     for b in bits
#         x = 2 * x + b
#     end
#     return x
# end

function interleaved_tensor_to_grid_2d(T::AbstractArray, n::Int)
    ndims(T) == 2 * n || throw(ArgumentError("Tensor must have 2n dimensions"))
    all(size(T, k) == 2 for k in 1:2*n) || throw(ArgumentError("Each tensor dimension must be 2"))

    Nx = 2^n
    Ny = 2^n
    u = zeros(eltype(T), Nx, Ny)

    for ix in 0:Nx-1
        xbits = int_to_bits_msb(ix, n)
        for iy in 0:Ny-1
            ybits = int_to_bits_msb(iy, n)
            bits = interleave_bits(xbits, ybits)

            inds = Tuple(b + 1 for b in bits)
            u[ix + 1, iy + 1] = T[inds...]
        end
    end

    return u
end

function interleaved_mps_to_grid_2d(mps::MPS, sites::Vector{<:Index})
    nsites = length(sites)
    iseven(nsites) || throw(ArgumentError("Need an even number of sites for interleaved 2D QTT"))
    n = nsites ÷ 2

    Tvec = mps_to_dense_vector(mps, sites)
    T = reshape(Tvec, ntuple(_ -> 2, 2 * n)...)

    return interleaved_tensor_to_grid_2d(T, n)
end

# -------------- STANDARD VECTORS <-> INTERLEAVED VECTORS -------------- 
function grid_to_standard_vector_2d(u::AbstractMatrix)
    return reshape(u, :)
end

function grid_to_interleaved_vector_2d(u::AbstractMatrix, n::Int)
    T = grid_to_interleaved_tensor_2d(u, n)
    return reshape(T, :)
end

function standard_vector_to_grid_2d(v::AbstractVector, n::Int)
    N = 2^n
    length(v) == N^2 || throw(ArgumentError("Expected vector of length $(N^2)"))
    return reshape(v, N, N)
end


# ==============================
# 1D ANALYTICAL MPO CONSTRUCTION
# ==============================

# The Shift Plus (S_+) operator in QTT format using block-matrix notation
function shift_plus_mpo_1d(sites::Vector{<:Index})
    n = length(sites)
    M = MPO(sites)
    
    # Create the internal bond (link) indices, dimension 2
    links = [Index(2, "Link,l=$i") for i in 1:n-1]
    
    # One-hot encoding: creates an ITensor with indices b (bL, bR) with a single 1 at position k (r, c)
    onehot(b::Index, k::Int) = (T = ITensor(b); T[b=>k] = 1; T)
    onehot(bL::Index, r::Int, bR::Index, c::Int) = (T = ITensor(bL,bR); T[bL=>r,bR=>c] = 1; T)

    for i in 1:n
        s = sites[i] # the physical dimension of the i-th site
        
        # Identity matrix
        I_mat = op("Id", s) 
        
        # J = [[0, 1], [0, 0]] with the appropriate physical indices
        J_mat = ITensor(s', s)
        J_mat[s'=>1, s=>2] = 1.0 
        
        # J^T = [[0, 0], [1, 0]] with the appropriate physical indices
        JT_mat = ITensor(s', s)
        JT_mat[s'=>2, s=>1] = 1.0 
        
        # Place the physical indexed matrices at the corresponding link spots of our MPO
        if i == 1 # First core: row vector [I, J]
            r = links[1]
            M[i] = I_mat * onehot(r, 1) + 
                   J_mat * onehot(r, 2)
        elseif i == n # Last core: column vector [J; J^T]
            l = links[i-1]
            M[i] = J_mat  * onehot(l, 1) + 
                   JT_mat * onehot(l, 2)
        else # Middle cores: 2x2 matrix [[I, J], [0, J^T]]
            l = links[i-1]
            r = links[i]
            M[i] = I_mat  * onehot(l, 1, r, 1) + 
                   J_mat  * onehot(l, 1, r, 2) + 
                   # Note: The [2,1] element is 0, so we just omit it
                   JT_mat * onehot(l, 2, r, 2)
        end
    end
    return M
end

# The Shift Minus (S_-) operator in QTT format using block-matrix notation
function shift_minus_mpo_1d(sites::Vector{<:Index})
    # Swap the primed and unprimed indices to interchange the input and output legs (equivalent of transposing the matrix)
    return swapprime(shift_plus_mpo_1d(sites), 0, 1)
end

# The MPO A = (1 - 2*cfl)I + cfl * (S_+ + S_-)
function diffusion_mpo_1d(sites::Vector{<:Index}, cfl::Float64)
    I_mat = MPO(sites, "Id")
    return (1 - 2*cfl) * I_mat + cfl * (shift_plus_mpo_1d(sites) + shift_minus_mpo_1d(sites))
end

function laplacian_mpo_1d(sites::Vector{<:Index})
    n = length(sites)
    M = MPO(sites)

    n >= 2 || throw(ArgumentError("Need at least 2 sites"))
    
    # Create the internal bond (link) indices, dimension 3
    links = [Index(3, "Link,l=$i") for i in 1:n-1]
    
    # One-hot encoding: creates an ITensor with indices b (bL, bR) with a single 1 at position k (r, c)
    onehot(b::Index, k::Int) = (T = ITensor(b); T[b=>k] = 1.0; T)
    onehot(bL::Index, r::Int, bR::Index, c::Int) = (T = ITensor(bL,bR); T[bL=>r,bR=>c] = 1.0; T)

    for i in 1:n
        s = sites[i] # the physical dimension of the i-th site
        
        # Identity matrix
        I_mat = op("Id", s) 
        
        # J = [[0, 1], [0, 0]] with the appropriate physical indices
        J_mat = ITensor(s', s)
        J_mat[s'=>1, s=>2] = 1.0 
        
        # J^T = [[0, 0], [1, 0]] with the appropriate physical indices
        JT_mat = ITensor(s', s)
        JT_mat[s'=>2, s=>1] = 1.0 
        
        # Place the physical indexed matrices at the corresponding link spots of our MPO
        if i == 1 
            r = links[1]
            M[i] = I_mat * onehot(r, 1) + 
                   JT_mat * onehot(r, 2) +
                   J_mat * onehot(r, 3)
        elseif i == n 
            l = links[n-1]
            M[i] = (2 * I_mat - J_mat - JT_mat) * onehot(l, 1) +
                   (-J_mat)  * onehot(l, 2) +
                   (-JT_mat) * onehot(l, 3)
        else 
            l = links[i-1]
            r = links[i]
            M[i] = I_mat  * onehot(l, 1, r, 1) +
                   JT_mat * onehot(l, 1, r, 2) +
                   J_mat  * onehot(l, 1, r, 3) +
                   J_mat  * onehot(l, 2, r, 2) +
                   JT_mat * onehot(l, 3, r, 3)
        end
    end
    return M
end

function timestep_mpo_1d(sites::Vector{<:Index}, cfl::Float64)
    I_mpo = MPO(sites, "Id")
    L_mpo = laplacian_mpo_1d(sites)
    return I_mpo - cfl * L_mpo
end


# ==============
# TIME EVOLUTION
# ==============

function evolve_mps_with_mpo(mps0::MPS, A::MPO, steps::Int; cutoff=1e-10, maxdim=64, verbose=false)
    mps = copy(mps0)
    
    # Pretty printing
    if verbose
        println("="^65)
        @printf("%-10s | %-20s | %-15s\n", "Step", "Max link dim", "Time (s)")
        println("-"^65)
    end

    # smoke-run to warm up the JIT compiler (discards the result)
    apply(A, mps; alg="naive", cutoff=cutoff, maxdim=maxdim)

    total_time = 0.0
    for step in 1:steps
        # Contract the MPO and MPS, then compress based on the given cutoff
        t = @elapsed mps = apply(A, mps; alg="naive", cutoff=cutoff, maxdim=maxdim)
        total_time += t
        if verbose
            @printf("%-10d | %-20d | %-15.6f\n", step, maxlinkdim(mps), t)
        end
    end
    if verbose
        println("-"^65)
        @printf("%-10s   %-20s   Total Time: %.6fs\n", "Total", "", total_time)
        println("="^65)
    end
    return mps
end

