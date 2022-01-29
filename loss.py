import torch
import torch.nn.functional as F
import sys


def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def cluster_contrastive_Loss(view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb + 1) * torch.log(p_j) \
                      - (lamb + 1) * torch.log(p_i))

    loss = loss.sum()

    return loss


def variance(batch1, batch2):
    "Calculate VIC loss"
    d, k = batch1.size()
    if batch1.size() != batch2.size():
        return 0
    z_a, z_b = batch1, batch2
    # variance loss
    std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
    std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
    std_loss = torch.mean(F.relu(1 - std_z_a))
    std_loss += torch.mean(F.relu(1 - std_z_b))
    return std_loss


def mse(batch1, batch2, device):
    # invariance loss
    return F.mse_loss(batch1, batch2)


def infoNCE_loss(view1, view2, device, temperature=torch.tensor(0.1)):
    n, dim = view1.size()
    assert (view1.size() == view2.size())

    dot_product = torch.matmul(view1, view2.T) / temperature
    mask = torch.eye(n, dtype=torch.float32).to(device)
    mask = mask * dot_product

    exp_dot_product = torch.exp(dot_product - mask)

    positive = mask.sum()
    negative = exp_dot_product.sum()
    loss = -positive / n + torch.log(negative)

    return loss