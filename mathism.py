def transpose(A: list[list[float] | list[int]])->list[list[float | int]]:
    """Returns the transpose of the provided matrix"""
    transposed = [list(i) for i in zip(*A)]
    return transposed

def qr_decomposition(A: list[list[float|int]])->tuple[list[list[float|int]], list[list[float|int]]]:
    """Performs QR decomposition using Householder reflections"""
    n = len(A)
    Q = [[0 if i != j else 1 for j in range(n)] for i in range(n)]
    R = [row[:] for row in A]
    
    for k in range(n-1):
        # Calculate the Householder vector
        sigma = sum(R[i][k] * R[i][k] for i in range(k, n))
        sigma = (sigma ** 0.5)
        
        if sigma == 0:
            continue
            
        beta = sigma * (sigma + abs(R[k][k]))
        if beta == 0:
            continue
            
        R[k][k] += (R[k][k]/abs(R[k][k])) * sigma if R[k][k] != 0 else sigma
        
        # Apply Householder transformation to R
        for j in range(k, n):
            tau = sum(R[i][k] * R[i][j] for i in range(k, n)) / beta
            for i in range(k, n):
                R[i][j] -= tau * R[i][k]
                
        # Update Q
        for j in range(n):
            tau = sum(R[i][k] * Q[i][j] for i in range(k, n)) / beta
            for i in range(k, n):
                Q[i][j] -= tau * R[i][k]
                
        # Clean up R column
        for i in range(k+1, n):
            R[i][k] = 0
            
    return Q, R

def solve_triangular(R: list[list[float|int]], b: list[float|int], upper: bool = True) -> list[float|int]:
    """Solves Rx = b where R is triangular"""
    n = len(b)
    x = [0] * n
    
    if upper:
        for i in range(n-1, -1, -1):
            x[i] = (b[i] - sum(R[i][j] * x[j] for j in range(i+1, n))) / R[i][i]
    else:
        for i in range(n):
            x[i] = (b[i] - sum(R[i][j] * x[j] for j in range(i))) / R[i][i]
            
    return x

def matrix_inverse(A: list[list[float|int]])->list[list[float|int]]:
    """Computes the inverse of matrix A using QR decomposition"""
    n = len(A)
    Q, R = qr_decomposition(A)
    
    # Transpose Q for easier column access
    Qt = [[Q[j][i] for j in range(n)] for i in range(n)]
    
    # Initialize inverse matrix
    inv_A = [[0] * n for _ in range(n)]
    
    # Solve RX = Q^T column by column
    for j in range(n):
        # Get jth column of Q^T
        qtj = [Qt[j][i] for i in range(n)]
        # Solve system
        col = solve_triangular(R, qtj)
        # Store in inverse matrix
        for i in range(n):
            inv_A[i][j] = col[i]
            
    return inv_A


def mat_mul(A: list[list[float|int]], B: list[list[float|int]])->list[list[float|int]]:
    """Matrix multiplication implementation"""
    return [[sum(a * b for a, b in zip(A_row, B_col)) 
            for B_col in zip(*B)] 
            for A_row in A]