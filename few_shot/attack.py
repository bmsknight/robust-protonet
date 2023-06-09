import copy

import torch
from torchattacks import PGD

from few_shot.protonet_wrapper import ProtoNetWrapper


class PGDAttackWrapperForTraining:
    def __init__(self, emb_model, distance, n_shot, k_way, is_he_model=False,
                 eps=8 / 255, alpha=2 / 255, steps=10, random_start=True, attack_type="query"):
        self.n_shot = n_shot
        self.k_way = k_way
        self.model = ProtoNetWrapper(emb_model, distance, n_shot, k_way, is_he_model=is_he_model)
        self.pgd = PGD(self.model, eps=eps, alpha=alpha, steps=steps, random_start=random_start)
        self.pgd.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.attack_type = attack_type

    def __call__(self, x, y, model):
        support = x[:self.n_shot * self.k_way]
        queries = x[self.n_shot * self.k_way:]

        model_copy = copy.deepcopy(model)
        self.model.embedding_model = model_copy
        self.model.eval()
        self.model.set_embeddings(support)

        self.pgd.set_model(self.model)
        if self.attack_type == "all":
            temp_y = list(range(self.k_way))
            temp_y = torch.Tensor(temp_y).type(torch.int64)
            temp_y = temp_y.repeat_interleave(self.n_shot).cuda()
            temp_y = torch.concat([temp_y,y], dim=0)
            adv_x = self.pgd(x, temp_y)
        else:
            adv_queries = self.pgd(queries, y)
            adv_x = torch.concat([support, adv_queries], dim=0)

        return adv_x
