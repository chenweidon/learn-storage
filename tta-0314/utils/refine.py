import torch

from utils.loss1 import physics_data_loss


def unrolled_refine(tm, x0, input_speckle, num_steps=3, step_size=0.1):
    """
    固定 TM 下，对 object 本身做少步可微 refinement
    """
    x = x0
    for _ in range(num_steps):
        x.requires_grad_(True)

        loss_phy = physics_data_loss(tm, x, input_speckle)
        grad = torch.autograd.grad(loss_phy, x, create_graph=True)[0]

        x = x - step_size * grad
        x = torch.clamp(x, 0.0, 1.0)

    return x
