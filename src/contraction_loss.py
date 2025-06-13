"""
Contractivity loss for diffusion models.

This module computes the contractivity loss for diffusion models by calculating the Jacobian of the model output with respect to the input.
"""

import torch
import torch.nn.functional as F

from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion import BaseNNDiffusion


def compute_jacobian(model: BaseNNDiffusion, xt: torch.Tensor, t: torch.Tensor, condition=None):
    """
    Compute the Jacobian of the model output with respect to the input xt.
    This is used to compute the Jacobian of the model output with respect to the input xt.
    The Jacobian is computed using autograd.

    Args:
        model (BaseNNDiffusion): The diffusion model.
        xt (torch.Tensor): The input tensor.
        t (torch.Tensor): The time step tensor.
        condition (torch.Tensor, optional): The condition tensor. Defaults to None.

    Returns:
        torch.Tensor: The Jacobian of the model output with respect to the input xt.
    """
    xt = xt.clone().detach().requires_grad_(True)
    output = model(xt, t, condition) # TODO: replace with sample()?
    batch_size, dim = xt.shape
    jacobian = torch.zeros(batch_size, dim, dim, device=xt.device)

    # differentiate each output dimension with respect to input xt
    for i in range(dim):
        grad_outputs = torch.zeros_like(output)
        grad_outputs[:, i] = 1.0
        grad = torch.autograd.grad(
            outputs=output,
            inputs=xt,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True,
            allow_unused=True)[0]
        jacobian[:, i, :] = grad

    return jacobian


def leading_eigenvalue_exact(J_sym):
    """
    Enforce negative definiteness by penalizing the max eigenvalue of the symmetric Jacobian.

    Args:
        J_sym (torch.Tensor): The symmetric Jacobian matrix of shape [batch_size, dim, dim].
    """
    eigvals = torch.linalg.eigvalsh(J_sym)  # [batch_size, dim]
    lambda_max = eigvals[:, -1]  # largest eigenvalue

    return lambda_max.mean()


def leading_eigenvalue_approx(J_sym, n_iters=10):
    """
    Compute the leading eigenvalue of the symmetric Jacobian matrix using power iteration.
    This is a computationally efficient approximation of the leading eigenvalue. It iteratively
    applies the Jacobian to a random vector and normalizes it, making it easier to retain the
    gradient flow through the model.

    Args:
        J_sym (torch.Tensor): The symmetric Jacobian matrix of shape [batch_size, dim, dim].
        n_iters (int, optional): The number of iterations for the power iteration method. Defaults to 10.

    Returns:
        torch.Tensor: The estimated leading eigenvalue of the Jacobian matrix, *averaged* over the batch.
    """
    v = torch.randn(J_sym.shape[0], J_sym.shape[-1], 1, device=J_sym.device)
    v = F.normalize(v, dim=1)

    # power iteration
    for _ in range(n_iters):
        v = J_sym @ v
        v = F.normalize(v, dim=1)

    lambda_max = torch.sum((J_sym @ v) * v, dim=(1, 2))  # Rayleigh quotient
    return lambda_max.mean()  # shape: [batch]


def compute_contractive_loss(model: DiscreteDiffusionSDE, xt: torch.Tensor, t: torch.Tensor,
                             condition=None, lambda_contr=0.1, loss_type="jacobian",
                             num_power_iters=10):
    """
    Compute the contractivity loss for the model. The contractivity loss is defined as the squared
    Frobenius norm of the Jacobian of the model output with respect to the input xt.

    Args:
        model (DiscreteDiffusionSDE): The diffusion model.
        xt (torch.Tensor): The input tensor.
        t (torch.Tensor): The time step tensor.
        condition (torch.Tensor, optional): The condition tensor. Defaults to None.
        lambda_contr (float, optional): The weight of the contractivity loss. Defaults to 0.1.
        loss_type (str, optional): The type of loss to compute. Options are "jacobian", "eigen_max", "eigen_avg", or "all". Defaults to "jacobian".
        n_iters_eigen (int, optional): The number of iterations for the eigenvalue approximation. Defaults to 10.

    Raises:
        ValueError: If the loss_type is not one of the supported types.

    Returns:
        dict: A dictionary containing the computed losses. The keys depend on the loss_type:
            - "jacobian_loss": The contractivity loss based on the Jacobian and lambda_contr.
            - "jacobian_norm": The Frobenius norm of the Jacobian.
            - "eigen_est": The computationally efficient estimated leading eigenvalue.
            - "eigen_avg": The average eigenvalue.
            - "eigen_std": The standard deviation.
    """
    if loss_type not in ["jacobian", "eigen_max", "eigen_avg", "all", "none"]:
        raise ValueError(f"Unknown loss: {loss_type}. Supported types: "
                         f"'jacobian', 'eigen_max', 'eigen_avg', 'all', 'none'.")

    # compute symmetric Jacobian
    J = compute_jacobian(model, xt, t, condition)
    J_sym = 0.5 * (J + J.transpose(-2, -1))

    # initialize results
    results = {
        "jacobian_loss": torch.tensor(0.0, device=xt.device),
        "jacobian_norm": torch.tensor(0.0, device=xt.device),
        "eigen_max": torch.tensor(0.0, device=xt.device),
        "eigen_avg": torch.tensor(0.0, device=xt.device),
        "eigen_std": torch.tensor(0.0, device=xt.device)
    }

    # compute losses based on the specified loss_type
    if loss_type == "eigen_max" or loss_type == "all":
        eig_approx = leading_eigenvalue_approx(J_sym, n_iters=num_power_iters)
        results["eigen_max"] = eig_approx

    if loss_type == "eigen_avg" or loss_type == "all":
        eigvals = torch.linalg.eigvalsh(J_sym)  # [batch_size, dim]
        eig_avg = eigvals.mean()
        eig_std = eigvals.std()
        results["eigen_avg"] = eig_avg
        results["eigen_std"] = eig_std

    if loss_type == "jacobian" or loss_type == "all":
        batch_size, dim, _ = J_sym.shape
        identity = torch.eye(dim, device=xt.device).expand(batch_size, -1, -1)
        loss = ((J_sym + lambda_contr * identity)**2).mean()
        results["jacobian_loss"] = loss
        results["jacobian_norm"] = torch.norm(J_sym, p="fro", dim=(-2, -1)).mean()

    return results
