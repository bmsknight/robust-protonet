import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import PATH
from few_shot.core import NShotTaskSampler, prepare_nshot_task
from few_shot.datasets import MiniImageNet
from few_shot.metrics import categorical_accuracy
from few_shot.models import get_few_shot_encoder
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
emb_model = get_few_shot_encoder(num_input_channels)
emb_model.load_state_dict(torch.load(PATH + f'/models/proto_nets/baseline.pth'))
model = ProtoNetWrapper(embedding_model=emb_model, distance=args.distance, n_shot=args.n_test, k_way=args.k_test,
                        is_he_model=False)
model.to(device, dtype=torch.double)

# evaluate
model.eval()
total_clean_acc = 0
count = 0
for batch_index, batch in enumerate(tqdm(evaluation_taskloader)):
    x, y = prepare_batch(batch)
    y_pred = model(x)
    count += y_pred.shape[0]
    total_clean_acc += categorical_accuracy(y, y_pred) * y_pred.shape[0]

clean_acc = total_clean_acc / count
print(clean_acc)
