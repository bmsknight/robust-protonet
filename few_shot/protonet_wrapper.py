from torch.nn import Module

from few_shot.proto import compute_prototypes
from few_shot.utils import pairwise_distances


class ProtoNetWrapper(Module):

    def __init__(self, embedding_model, distance, n_shot, k_way, is_he_model=False):
        super().__init__()
        self.embedding_model = embedding_model
        self.distance = distance
        self.n_shot = n_shot
        self.k_way = k_way
        # self.embedding_model.eval()
        self.he_model = is_he_model

    def _classification_head(self, embeddings):
        support = embeddings[:self.n_shot * self.k_way]
        queries = embeddings[self.n_shot * self.k_way:]
        prototypes = compute_prototypes(support, self.k_way, self.n_shot, rescale=self.he_model)
        distances = pairwise_distances(queries, prototypes, self.distance)
        y_pred = (-distances).softmax(dim=1)
        return y_pred

    def forward(self, x):
        emb = self.embedding_model(x)
        y_pred = self._classification_head(emb)
        return y_pred
