import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchattacks import PGD
from tqdm import tqdm

from config import PATH
from few_shot.core import NShotTaskSampler, prepare_nshot_task
from few_shot.datasets import MiniImageNet
from few_shot.metrics import categorical_accuracy
from few_shot.resnet_12 import get_few_shot_he_encoder
from few_shot.resnet_12 import get_few_shot_encoder
from few_shot.protonet_wrapper import ProtoNetWrapper
from few_shot.utils import setup_dirs

setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="miniImageNet")
parser.add_argument('--model', default="baseline")
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=5, type=int)
parser.add_argument('--n-test', default=5, type=int)
parser.add_argument('--k-train', default=20, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=15, type=int)
parser.add_argument('--q-test', default=1, type=int)
args = parser.parse_args()

evaluation_episodes = 2000

dataset_class = MiniImageNet
num_input_channels = 3

param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
            f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

print(param_str)

###################
# Create datasets #
###################


evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, evaluation_episodes, args.n_test, args.k_test, args.q_test),
    num_workers=4
)
prepare_batch = prepare_nshot_task(args.n_test, args.k_test, args.q_test)

#########
# Model #
#########
if ("baseline" in args.model) or ("contrast" in args.model):
    emb_model = get_few_shot_encoder(num_input_channels, avg_pool=False, drop_rate=0.1)
    emb_model = torch.nn.DataParallel(emb_model)
    args.distance = "l2"
    model_str = args.model
    is_he_model = False
elif ("he" in args.model) or ("arc" in args.model):
    emb_model = get_few_shot_he_encoder(num_input_channels,final_layer_size=16000, avg_pool=False, drop_rate=0.1)
    emb_model = torch.nn.DataParallel(emb_model)
    args.distance = "cosine"
    model_str = args.model
    is_he_model = True
elif "fc" in args.model:
    emb_model = get_few_shot_he_encoder(num_input_channels, final_layer_size=16000, is_he=False,
                                        avg_pool=False, drop_rate=0.1)
    emb_model = torch.nn.DataParallel(emb_model)
    args.distance = "l2"
    model_str = args.model
    is_he_model = False
else:
    raise ValueError("Unknown Model type")
emb_model.load_state_dict(torch.load(PATH + f'/models/proto_nets/{model_str}.pth'))
model = ProtoNetWrapper(embedding_model=emb_model, distance=args.distance, n_shot=args.n_test, k_way=args.k_test,
                        is_he_model=is_he_model)
model.to(device, dtype=torch.float)

# attack
atk = PGD(model, eps=8 / 255, alpha=2 / 255, steps=20, random_start=True)
atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print(atk)

# evaluate
model.eval()
total_clean_acc = 0
total_adv_acc = 0
count = 0
for batch_index, batch in enumerate(tqdm(evaluation_taskloader)):
    x, y = prepare_batch(batch)
    support = x[:args.n_test * args.k_test]
    queries = x[args.n_test * args.k_test:]
    model.set_embeddings(support)

    y_pred = model(queries)
    count += y_pred.shape[0]
    total_clean_acc += categorical_accuracy(y, y_pred) * y_pred.shape[0]
    adv_query = atk(queries, y)
    y_pred_adv = model(adv_query)
    total_adv_acc += categorical_accuracy(y, y_pred_adv) * y_pred_adv.shape[0]

clean_acc = total_clean_acc / count
print(f"Clean Accuracy of the model : {clean_acc}")

adv_acc = total_adv_acc / count
print(f"Adversarial Accuracy of the model : {adv_acc}")
