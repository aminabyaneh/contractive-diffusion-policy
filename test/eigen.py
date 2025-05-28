import torch
import torch.nn.functional as F
import time

from src.contraction_loss import leading_eigenvalue_approx, leading_eigenvalue_exact


def generate_symmetric_matrices(batch_size=512, dim=256):
    """
    Generates a batch of symmetric matrices.

    Args:
        batch_size (int, optional): Number of matrices to generate. Defaults to 512.
        dim (int, optional): Dimension of each square matrix. Defaults to 256.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, dim, dim) containing symmetric matrices.
    """
    A = torch.randn(batch_size, dim, dim)
    return (A + A.transpose(1, 2)) / 2  # symmetric matrix


def benchmark():
    """
    Benchmarks the performance of exact and approximate leading eigenvalue computations on symmetric matrices.

    This function generates symmetric matrices, then measures and records the execution time for:
      - Computing the leading eigenvalue exactly.
      - Computing the leading eigenvalue approximately using a specified number of iterations.

    The timings are measured on the GPU (CUDA) and returned as a dictionary.

    Returns:
        dict: A dictionary containing the elapsed time (in seconds) for each method:
            - 'exact': Time for exact leading eigenvalue computation.
            - 'approx_{n}_iters': Time for approximate computation with n iterations (for n in [5, 10, 20]).
    """
    J_sym = generate_symmetric_matrices().to("cuda")
    timings = {}

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        leading_eigenvalue_exact(J_sym)
    torch.cuda.synchronize()
    timings['exact'] = time.time() - start

    for n in [5, 10, 20, 50]:
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            leading_eigenvalue_approx(J_sym, n_iters=n)
        torch.cuda.synchronize()
        timings[f'approx_{n}_iters'] = time.time() - start

    return timings


if __name__ == "__main__":
    results = benchmark()
    for method, time_taken in results.items():
        print(f"{method}: {time_taken:.6f} seconds")

    '''
    Sample output
    > exact: 36.205545 seconds
    > approx_5_iters: 0.118569 seconds
    > approx_10_iters: 0.069589 seconds
    > approx_20_iters: 0.134029 seconds
    '''