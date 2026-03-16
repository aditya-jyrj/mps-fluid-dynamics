# Introduction

To convert a matrix into an MPO, Quimb only allows dense matrices as input. As scipy's sparse matrices could not be used, this meant that constructing the Laplacian MPO for larger $n$ (where the system size $N=2^n$) took up a lot of time.

To circumvent this, I attempted to construct the MPOs that comprise the time-stepping operator manually. 

## Identity MPO

`W = np.zeros((1, 1, 2, 2))` creates a 4-dimensional tensor with bond dimension $\chi_{\text{left}} = \chi_{\text{right}} = 1$ and a physical dimension of 2.

`W[0, 0] = np.eye(2)` fills the slice `W[0,0,:,:]` with an identity matrix, so now
```
W[0,0,0,0] = 1
W[0,0,0,1] = 0
W[0,0,1,0] = 0
W[0,0,1,1] = 1
```

ie, if we get input 0, we output 0. If we get input 1, we output 1.

The MPO is essentially a chain of length $n$ of $2\times 2$ identity matrices multiplying together (but they're technically in 4 dimensions).

## Plus-Shift MPO

We want an MPO that represents $\ket i \to \ket{i+1}$ except at the ends, where $i\in[1,n]$ is the site number. Alternatively, our MPO executes 
$$
\ket{x_0, x_1,x_2,\dots, x_n} \to \ket{x_0, x_0, x_1,\dots, x_{n-1}}
$$