import torch
from torch.optim import Optimizer
from torch.nn import Module
from typing import Callable, Union

from few_shot.utils import pairwise_distances


def proto_net_episode(model: Module,
                      optimiser: Optimizer,
                      loss_fn: Callable,
                      x: Union[torch.Tensor,list],
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

    if type(x) == list:
        is_joint_adv_training = True
        x_adv = x[1]
        x = x[0]
    else:
        is_joint_adv_training = False
        x_adv = None

    # Embed all samples
    embeddings = model(x)

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
    loss_clean = loss_fn(log_p_y, y)

    if is_joint_adv_training:
        adv_embeddings = model(x_adv)
        adv_queries = adv_embeddings[n_shot * k_way:]
        adv_distance = pairwise_distances(adv_queries, prototypes, distance)
        adv_log_p_y = (-adv_distance).log_softmax(dim=1)
        loss_adv = loss_fn(adv_log_p_y, y)

        loss = loss_clean + 0.5 * loss_adv
        # print("joint train is called")
    else:
        loss = loss_clean
        # print("non joint train")

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
