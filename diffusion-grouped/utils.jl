using ITensors, ITensorMPS
using LinearAlgebra, Printf

# ===========
# DEFINITIONS
# ===========


# n: determines the system size, Nx = Ny = 2^n

# GRID: a matrix u indexed simply by u[i_x, i_y]
# STANDARD VECTOR: 1D vector of length Nx * Ny = 2^(2n) (u[1,1], u[2,1], ... u[Nx, 1], u[1,2], u[2,2], ... u[Nx, 2], ... u[Nx, Ny])

# GROUPED TENSOR: rank-2n tensor T where each dimension has size 2. represented by T[x1, x2, ... xn, y1, ... yn], where each index is 0 or 1
# GROUPED MPS: an MPS with 2n sites of physical dimension 2, ordered like [x1] [x2] ... [xn] [y1] ... [yn]
# GROUPED VECTOR: 1D vector of length Nx * Ny obtained from flattening grouped tensor or converting grouped MPS


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

# Note: these functions return a vector in the ordering defined by `sites`
#       so a group-ordered MPS will be converted to a group-ordered vector

function mps_to_vector(mps::MPS, sites::Vector{<:Index})
    T = prod(mps) # contract entire mps
    C = combiner(reverse(sites)...) # reverse site ordering, then combine all into one index
    Tc = T * C
    return Array(Tc, combinedind(C))
end

function vector_to_mps(v::AbstractVector, sites::Vector{<:Index}; cutoff=1e-10)
    n = length(sites)
    length(v) == 2^n || throw(ArgumentError("Expected vector of length $(2^n), got $(length(v))"))

    T = reshape(v, ntuple(_ -> 2, n)...)
    IT = ITensor(T, reverse(sites)...)
    return MPS(IT, sites; cutoff=cutoff)
end


function mpo_to_matrix(M::MPO, sites::Vector{<:Index})
    T = prod(M)
    
    # REVERSE the sites so sites[n] varies fastest (Julia follows column-major convention unlike Python)
    C_row = combiner(reverse(prime.(sites))...)
    C_col = combiner(reverse(sites)...)
    
    Tc = T * C_row * C_col
    return Array(Tc, combinedind(C_row), combinedind(C_col))
end


function matrix_to_mpo(A::AbstractMatrix, sites::Vector{<:Index}; cutoff=1e-12)
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
# GROUPED 2D BASIS HELPERS
# ============================

# -------------- GRID --> GROUPED -------------- 

function int_to_bits_msb(k::Int, nbits::Int)
    # converts numbers to binary digits with most significant bit coming first
    # k should run from 0 to 2^nbits - 1

    # digits(2, base=2) == [0,1], but digits(2, base=2, pad=4) = [0,1,0,0] (ensures that all bitvectors are of same length)
    # note that julia returns least significant bit first, ie [0,1,0,0] instead of [0,0,1,0]. we have to reverse this

    ds = digits(k, base=2, pad=nbits) 
    return reverse(ds) 
end

function group_bits(xbits::Vector{Int}, ybits::Vector{Int})
    length(xbits) == length(ybits) || throw(ArgumentError("xbits and ybits must have same length"))
    return vcat(xbits, ybits)
end

function grid_to_grouped_tensor_2d(u::AbstractMatrix, n::Int)
    Nx, Ny = size(u)

    Nx == 2^n || throw(ArgumentError("Expected Nx = 2^n = $(2^n), got Nx = $Nx"))
    Ny == 2^n || throw(ArgumentError("Expected Ny = 2^n = $(2^n), got Ny = $Ny"))

    T = zeros(eltype(u), ntuple(_ -> 2, 2 * n)) # rank-2n tensor, each dimension has size 2. each element is 0 of same type as u

    # index through all the x and y indices, converting both to binary and concatenating
    for ix in 0:Nx-1
        xbits = int_to_bits_msb(ix, n)
        for iy in 0:Ny-1
            ybits = int_to_bits_msb(iy, n)
            bits = group_bits(xbits, ybits)

            inds = Tuple(b + 1 for b in bits) # increment everything by 1 since julia is 1-indexed
            T[inds...] = u[ix + 1, iy + 1]
        end
    end

    return T
end

function grid_to_grouped_mps_2d(u::AbstractMatrix, sites::Vector{<:Index}; cutoff=1e-10)
    nsites = length(sites) # nsites is total number of bits, ie num x bits + num y bits
    iseven(nsites) || throw(ArgumentError("Need an even number of sites for grouped 2D QTT"))
    n = nsites ÷ 2 # ÷ returns integer, / returns float. this is the number of bits per dimension

    T = grid_to_grouped_tensor_2d(u, n)
    IT = ITensor(T, reverse(sites)...)
    return MPS(IT, sites; cutoff=cutoff)
end

# -------------- GROUPED --> GRID -------------- 

# function bits_msb_to_int(bits::AbstractVector{<:Integer})
#     x = 0
#     for b in bits
#         x = 2 * x + b
#     end
#     return x
# end

function grouped_tensor_to_grid_2d(T::AbstractArray, n::Int)
    ndims(T) == 2 * n || throw(ArgumentError("Tensor must have 2n dimensions"))
    all(size(T, k) == 2 for k in 1:2*n) || throw(ArgumentError("Each tensor dimension must be 2"))

    Nx = 2^n
    Ny = 2^n
    u = zeros(eltype(T), Nx, Ny)

    for ix in 0:Nx-1
        xbits = int_to_bits_msb(ix, n)
        for iy in 0:Ny-1
            ybits = int_to_bits_msb(iy, n)
            bits = group_bits(xbits, ybits)

            inds = Tuple(b + 1 for b in bits)
            u[ix + 1, iy + 1] = T[inds...]
        end
    end

    return u
end

function grouped_mps_to_grid_2d(mps::MPS, sites::Vector{<:Index})
    nsites = length(sites)
    iseven(nsites) || throw(ArgumentError("Need an even number of sites for grouped 2D QTT"))
    n = nsites ÷ 2

    Tvec = mps_to_vector(mps, sites)
    T = reshape(Tvec, ntuple(_ -> 2, 2 * n)...)

    return grouped_tensor_to_grid_2d(T, n)
end

# -------------- STANDARD VECTORS <-> GROUPED VECTORS -------------- 
function grid_to_standard_vector_2d(u::AbstractMatrix)
    return reshape(u, :)
end

function grid_to_grouped_vector_2d(u::AbstractMatrix, n::Int)
    T = grid_to_grouped_tensor_2d(u, n)
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


function laplacian_mpo_1d(sites::Vector{<:Index})
    n = length(sites)
    n >= 2 || throw(ArgumentError("Need at least 2 sites"))

    M = MPO(sites)
    
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

function tensor_product_mpo(A::MPO, B::MPO, sites::Vector{<:Index})
    NA = length(A)
    NB = length(B)

    length(sites) == NA + NB || throw(ArgumentError("Expected $(NA+NB) sites, got $(length(sites))"))

    # allocate new MPO
    M = MPO(sites)

    # copy A cores
    for i in 1:NA
        M[i] = A[i]
    end

    # copy B cores
    for j in 1:NB
        M[NA + j] = B[j]
    end

    return M
end

function laplacian_mpo_2d(sites::Vector{<:Index})
    nsites = length(sites)
    iseven(nsites) || throw(ArgumentError("Need even number of sites"))
    n = nsites ÷ 2

    x_sites = sites[1:n]
    y_sites = sites[n+1:2n]

    Lx_1d = laplacian_mpo_1d(x_sites)
    Ly_1d = laplacian_mpo_1d(y_sites)

    Ix_1d = MPO(x_sites, "Id")
    Iy_1d = MPO(y_sites, "Id")

    LxIy = tensor_product_mpo(Lx_1d, Iy_1d, sites)
    IxLy = tensor_product_mpo(Ix_1d, Ly_1d, sites)

    return LxIy + IxLy
end

function timestep_mpo_2d(sites::Vector{<:Index}, cfl::Float64)
    I_mpo = MPO(sites, "Id")
    L = laplacian_mpo_2d(sites)
    return I_mpo - cfl * L
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

