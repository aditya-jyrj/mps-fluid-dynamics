# Import the necessary modules
# using Pkg; Pkg.add("ITensors"); Pkg.add("ITensorMPS") # Run this if you don't have the packages installed
using ITensors, ITensorMPS

# Native Julia packages
using LinearAlgebra, Printf

# The matrix Laplacian
function laplacian(N::Int, bc::Symbol=:dirichlet)
    # Create a vector of ones with length N
    v = ones(N)
    
    # Create the Laplacian matrix
    L = diagm(0 => 2*v, -1 => -v[1:N-1], 1 => -v[1:N-1])    
    
    # Apply boundary conditions
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

# The exact time evolution matrix in dense format
function A_exact(N::Int, cfl::Float64, bc::Symbol=:dirichlet)
    return I - cfl * laplacian(N, bc)
end

# The Shift Plus (S_+) operator in QTT format using block-matrix notation
function qtt_shift_plus(sites::Vector{<:Index})
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
function qtt_shift_minus(sites::Vector{<:Index})
    # Swap the primed and unprimed indices to interchange the input and output legs (equivalent of transposing the matrix)
    return swapprime(qtt_shift_plus(sites), 0, 1)
end

# The MPO A = (1 - 2*cfl)I + cfl * (S_+ + S_-)
function A_mpo(sites::Vector{<:Index}, cfl::Float64)
    I_mat = MPO(sites, "Id")
    return (1 - 2*cfl) * I_mat + cfl * (qtt_shift_plus(sites) + qtt_shift_minus(sites))
end

# ---------------------- Vector to Tensor to MPS ----------------------
function dense_to_qtt_mps(u::Vector{<:Number}, sites::Vector{<:Index}; cutoff=1e-10)
    n = length(sites)
    # Reshape the vector into a 2x2x...x2 multidimensional tensor
    u_tensor = reshape(u, fill(2, n)...)
    
    # Push the dense array into a single ITensor block
    T = ITensor(u_tensor, reverse(sites)...)
    
    # Perform a sequential SVD to break the block down into an MPS
    return MPS(T, sites; cutoff=cutoff)
end

# ---------------------- MPO to Dense Matrix ----------------------
function mpo_to_matrix(M::MPO, sites::Vector{<:Index})
    T = prod(M)
    
    # REVERSE the sites so sites[n] varies fastest (Julia follows column-major convention unlike Python)
    C_row = combiner(reverse(prime.(sites))...)
    C_col = combiner(reverse(sites)...)
    
    Tc = T * C_row * C_col
    
    return Array(Tc, combinedind(C_row), combinedind(C_col))
end

# MPS parameters
n = 5
steps = 10
N = 2^n

# Mesh parameters
nu = 1e-3
cfl = 0.1
x = range(0, 1, length=N+1)[1:end-1]
dx = x[2] - x[1]
dt = cfl * dx^2 / nu

sites = siteinds("S=1/2", n)

# Initial state sampled at x
u0 = @. sin(2 * pi * 2 * x) + 0.5 * sin(2 * pi * 7 * x) # same initial condition as in the notebook

mps0 = dense_to_qtt_mps(u0, sites)

# Compare the A obtained via MPO and exact matrix
A_mpo_network = A_mpo(sites, cfl)
A_mpo_mat = mpo_to_matrix(A_mpo_network, sites)

A_exact_mat = A_exact(N, cfl, :dirichlet)

diff = norm(A_mpo_mat - A_exact_mat)
println("Max difference between A_mpo_mat and A_exact_mat: ", diff)

# ---------------------- Time Evolution ----------------------
function evolve_mps(mps0::MPS, A::MPO, steps::Int; cutoff=1e-10, maxdim=64)
    mps = copy(mps0)
    
    # Pretty printing
    println("="^65)
    @printf("%-10s | %-20s | %-15s\n", "Step", "Max link dim", "Time (s)")
    println("-"^65)

    # smoke-run to warm up the JIT compiler (discards the result)
    apply(A, mps; alg="naive", cutoff=cutoff, maxdim=maxdim)

    total_time = 0.0
    for step in 1:steps
        # Contract the MPO and MPS, then compress based on the given cutoff
        t = @elapsed mps = apply(A, mps; alg="naive", cutoff=cutoff, maxdim=maxdim)
        total_time += t

        @printf("%-10d | %-20d | %-15.6f\n", step, maxlinkdim(mps), t)
    end
    println("-"^65)
    @printf("%-10s   %-20s   Total Time: %.6fs\n", "Total", "", total_time)
    println("="^65)
    return mps
end

final_mps = evolve_mps(mps0, A_mpo_network, steps)

#=
Output: 
Max difference between A_mpo_mat and A_exact_mat: 8.092116931227675e-15
=================================================================
Step       | Max link dim         | Time (s)       
-----------------------------------------------------------------
1          | 4                    | 0.000559       
2          | 4                    | 0.000599       
3          | 4                    | 0.000483       
4          | 4                    | 0.000446       
5          | 4                    | 0.000457       
6          | 4                    | 0.000437       
7          | 4                    | 0.000456       
8          | 4                    | 0.000675       
9          | 4                    | 0.000500       
10         | 4                    | 0.000445       
-----------------------------------------------------------------
Total                               Total Time: 0.005057s
=================================================================
5-element MPS:
 ((dim=2|id=933|"S=1/2,Site,n=1"), (dim=2|id=878|"CMB,Link"))
 ((dim=2|id=403|"S=1/2,Site,n=2"), (dim=4|id=563|"CMB,Link"), (dim=2|id=878|"CMB,Link"))
 ((dim=2|id=804|"S=1/2,Site,n=3"), (dim=4|id=180|"CMB,Link"), (dim=4|id=563|"CMB,Link"))
 ((dim=2|id=596|"S=1/2,Site,n=4"), (dim=2|id=570|"CMB,Link"), (dim=4|id=180|"CMB,Link"))
 ((dim=2|id=114|"S=1/2,Site,n=5"), (dim=2|id=570|"CMB,Link"))
=#
