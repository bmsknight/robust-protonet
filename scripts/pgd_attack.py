import sys
sys.path.append('.')
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
from few_shot.models import get_few_shot_encoder
from few_shot.utils import setup_dirs

# assert torch.cuda.is_available()
# device = torch.device('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="miniImageNet")
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=5, type=int)
parser.add_argument('--n-test', default=5, type=int)
parser.add_argument('--k-train', default=20, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=15, type=int)
parser.add_argument('--q-test', default=1, type=int)
parser.add_argument('--weights_path', default=".\\few_shot\\models\\proto_nets\\baseline\\baseline.pth", type=str)
# parser.add_argument('--weights_path', default=".\\few_shot\\models\\proto_nets\\contrastive_results\contrast_miniImageNet_nt=5_kt=20_qt=15_nv=5_kv=5_qv=1.pth", type=str)
args = parser.parse_args()

param_str = f'{args.dataset}_nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

setup_dirs(param_str)

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
    num_workers=1
)
prepare_batch = prepare_nshot_task(args.n_test, args.k_test, args.q_test)

#########
# Model #
#########
evaluation_episodes = 2000
episodes_per_epoch = 100

#########
# Model #
#########
model = get_few_shot_encoder(num_input_channels)
model.to(device, dtype=torch.float)
if args.weights_path is not None:
    model.load_state_dict(torch.load(args.weights_path, map_location=torch.device(device)))
    print(f'Loaded weights from {args.weights_path}')

# attack
atk = PGD(model, eps=8 / 255, alpha=2 / 225, steps=10, random_start=True)
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