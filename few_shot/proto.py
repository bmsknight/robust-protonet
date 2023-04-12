import torch
import copy
from torch.optim import Optimizer
from torch.nn import Module
from typing import Callable
from few_shot.utils import pairwise_distances

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def attack(x, y, model, atk, train):
    if train and atk.steps != 7:
        atk.steps = 7
    elif not train and atk.steps != 20:
        atk.steps = 20
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    atk.model = model_copied
    adv_x = atk(x, y)
    return adv_x

def proto_net_episode(model: Module,
                      optimiser: Optimizer,
                      loss_fn: Callable,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      n_shot: int,
                      k_way: int,
                      q_queries: int,
                      distance: str,
                      train: bool,
                      is_he_model: bool = False):
    """Performs a single training episode for a Prototypical Network.

    # Arguments
        model: Prototypical Network to be trained.
        optimiser: Optimiser to calculate gradient step
        loss_fn: Loss function to calculate between predictions and outputs. Should be cross-entropy
        x: Input samples of few shot classification task
        y: Input labels of few shot classification task
        n_shot: Number of examples per class in the support set
        k_way: Number of classes in the few shot classification task
        q_queries: Number of examples per class in the query set
        distance: Distance metric to use when calculating distance between class prototypes and queries
        train: Whether (True) or not (False) to perform a parameter update

    # Returns
        loss: Loss of the Prototypical Network on this task
        y_pred: Predicted class probabilities for the query set on this task
    """
    if train:
        # Zero gradients
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    # Embed all samples
    xq = x[n_shot * k_way:]
    adv_x = attack(xq, y, model, atk, train)
    x = torch.concat((x[:n_shot*k_way], adv_x), 0)
    embeddings = model(x.to(device))
    # embeddings = model(x.to(device))

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot*k_way]
    queries = embeddings[n_shot*k_way:]
    # For HE model we need to rescale the prototypes to ensure they are on the manifold
    prototypes = compute_prototypes(support, k_way, n_shot, rescale=is_he_model)

    # Calculate squared distances between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = pairwise_distances(queries, prototypes, distance)

    # Calculate log p_{phi} (y = k | x)
    log_p_y = (-distances).log_softmax(dim=1)
    loss = loss_fn(log_p_y, y)

    # Prediction probabilities are softmax over distances
    y_pred = (-distances).softmax(dim=1)

    if train:
        # Take gradient step
        loss.backward()
        optimiser.step()
    else:
        pass

    return loss, y_pred


def compute_prototypes(support: torch.Tensor, k: int, n: int, rescale: bool = False) -> torch.Tensor:
    """Compute class prototypes from support samples.

    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task

    # Returns
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    class_prototypes = support.reshape(k, n, -1).mean(dim=1)

    # rescale the prototype embedding to be in the manifold for Hypersphere Embedding
    if rescale:
        class_prototypes = torch.nn.functional.normalize(class_prototypes, p=2, dim=1)
    return class_prototypes
