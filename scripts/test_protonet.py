"""
Protonet test script
"""
import sys
sys.path.append('.')
import argparse
import torch
import tqdm
from typing import List, Iterable, Callable, Tuple
from few_shot.callbacks import *
from few_shot.core import NShotTaskSampler, prepare_nshot_task
from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.models import get_few_shot_encoder
from few_shot.proto import proto_net_episode
from few_shot.utils import setup_dirs
from few_shot.metrics import categorical_accuracy
from torch.utils.data import DataLoader
  
from config import PATH

# assert torch.cuda.is_available()
# device = torch.device('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='miniImageNet')
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-test', default=5, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-test', default=1, type=int)
parser.add_argument('--weights_path', default=".\\few_shot\\models\\proto_nets\\baseline\\baseline.pth", type=str)
# parser.add_argument('--weights_path', default=".\\few_shot\\models\\proto_nets\\contrastive_results\contrast_miniImageNet_nt=5_kt=20_qt=15_nv=5_kv=5_qv=1.pth", type=str)
args = parser.parse_args()

param_str = f'{args.dataset}_nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

setup_dirs(param_str)

if args.dataset == 'omniglot':
    n_epochs = 40
    dataset_class = OmniglotDataset
    num_input_channels = 1
    drop_lr_every = 20
elif args.dataset == 'miniImageNet':
    n_epochs = 80
    dataset_class = MiniImageNet
    num_input_channels = 3
    drop_lr_every = 40
else:
    raise (ValueError, 'Unsupported dataset')

print(param_str)

class EvaluateFewShot:
    """Evaluate a network on  an n-shot, k-way classification tasks after every epoch.

    # Arguments
        eval_fn: Callable to perform few-shot classification. Examples include `proto_net_episode`,
            `matching_net_episode` and `meta_gradient_step` (MAML).
        num_tasks: int. Number of n-shot classification tasks to evaluate the model with.
        n_shot: int. Number of samples for each class in the n-shot classification tasks.
        k_way: int. Number of classes in the n-shot classification tasks.
        q_queries: int. Number query samples for each class in the n-shot classification tasks.
        task_loader: Instance of NShotWrapper class
        prepare_batch: function. The preprocessing function to apply to samples from the dataset.
        prefix: str. Prefix to identify dataset.
    """

    def __init__(self,
                 eval_fn: Callable,
                 num_tasks: int,
                 n_shot: int,
                 k_way: int,
                 q_queries: int,
                 model,
                 taskloader: torch.utils.data.DataLoader,
                 prepare_batch: Callable,
                 loss_fn: Callable,
                 prefix: str = 'test_',
                 **kwargs):
        self.eval_fn = eval_fn
        self.num_tasks = num_tasks
        self.n_shot = n_shot
        self.k_way = k_way
        self.q_queries = q_queries
        self.model = model
        self.taskloader = taskloader
        self.prepare_batch = prepare_batch
        self.prefix = prefix
        self.kwargs = kwargs
        self.metric_name = f'{self.prefix}{self.n_shot}-shot_{self.k_way}-way_acc'
        self.loss_fn = loss_fn
        self.optimiser = None

    def evaluate(self, logs=None):
        logs = logs or {}
        seen = 0
        totals = {'loss': 0, self.metric_name: 0}
        loop = tqdm(enumerate(self.taskloader), desc = 'Evaluation Iteration')
        for _, batch in loop:
            x, y = self.prepare_batch(batch)

            loss, y_pred = self.eval_fn(
                self.model,
                self.optimiser,
                self.loss_fn,
                x,
                y,
                n_shot=self.n_shot,
                k_way=self.k_way,
                q_queries=self.q_queries,
                train=False,
                **self.kwargs
            )

            seen += y_pred.shape[0]

            totals['loss'] += loss.item() * y_pred.shape[0]
            totals[self.metric_name] += categorical_accuracy(y, y_pred) * y_pred.shape[0]

        logs[self.prefix + 'loss'] = totals['loss'] / seen
        logs[self.metric_name] = totals[self.metric_name] / seen
        print('Test Logs:', logs)
        return logs

if __name__=='__main__':
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

    loss_fn = torch.nn.NLLLoss().to(device)

    ###################
    # Create test dataset #
    ###################
    evaluation = dataset_class('evaluation')
    evaluation_taskloader = DataLoader(
        evaluation,
        batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
        num_workers=4
    )

    print(f'\n{"*"*10}  EVAULATION STARTED  {"*"*10}\n')
    eval = EvaluateFewShot(
        eval_fn=proto_net_episode,
        num_tasks=evaluation_episodes,
        n_shot=args.n_test,
        k_way=args.k_test,
        q_queries=args.q_test,
        model=model,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
        distance=args.distance,
        loss_fn=loss_fn
        )
    print(f'\n{"*"*10}  Created Evaluation Object  {"*"*10}\n')
    logs = eval.evaluate()
    print(f'\n{"*"*10}  Obtained Evaluation Logs  {"*"*10}\n')
    print(f'\n{"*"*10}  EVAULATION ENDED  {"*"*10}\n')
    sys.exit()
    