"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
import sys
sys.path.append('.')
import argparse

from few_shot.callbacks import *
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.models import get_few_shot_encoder, SupConProjHead
from few_shot.proto import proto_net_sup_contrast_episode, proto_net_episode
from few_shot.train import fit_contrast
from few_shot.utils import setup_dirs, SupConLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
  
from config import PATH

setup_dirs()
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
parser.add_argument('--n-train', default=1, type=int)
parser.add_argument('--n-test', default=1, type=int)
parser.add_argument('--k-train', default=60, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=5, type=int)
parser.add_argument('--q-test', default=1, type=int)
parser.add_argument('--contrast_feature_dim', default=128)
parser.add_argument('--contrast_head', default='mlp')
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--weights_path', type=str)
parser.add_argument('--resume', default=False, type=bool)
args = parser.parse_args()

evaluation_episodes = 1000
episodes_per_epoch = 100

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

param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
            f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

print(param_str)

###################
# Create datasets #
###################
background = dataset_class('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, episodes_per_epoch, args.n_train, args.k_train, args.q_train),
    num_workers=4
)
evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
    num_workers=4
)

#########
# Model #
#########
model = get_few_shot_encoder(num_input_channels)
model.to(device, dtype=torch.double)
model.load_state_dict(torch.load(args.weights_path))
print(f'Loaded weights from {args.weights_path}')

proj_head = SupConProjHead(dim_in=1600, feat_dim=args.contrast_feature_dim, head=args.contrast_head)
proj_head.to(device, dtype=torch.double)

############
# Training #
############
print(f'Training Prototypical network on {args.dataset}...')
optimiser = Adam(list(model.parameters()) + list(proj_head.parameters()), lr=1e-3)
# loss_fn = torch.nn.NLLLoss().cuda()
loss_fn = torch.nn.NLLLoss().to(device)
contrast_loss_fn = SupConLoss().to(device)


def lr_schedule(epoch, lr):
    # Drop lr every 2000 episodes
    if epoch % drop_lr_every == 0:
        return lr / 2
    else:
        return lr


callbacks = [
    EvaluateFewShot(
        eval_fn=proto_net_episode,
        num_tasks=evaluation_episodes,
        n_shot=args.n_test,
        k_way=args.k_test,
        q_queries=args.q_test,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
        distance=args.distance
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/proto_nets/contrast_{param_str}.pth',
        monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'
    ),
    LearningRateScheduler(schedule=lr_schedule),
    CSVLogger(PATH + f'/models/proto_nets/contrast_logs_{param_str}.csv'),
]

fit_contrast(
    model,
    optimiser,
    loss_fn,
    epochs=n_epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=proto_net_sup_contrast_episode,
    start_epoch=args.start_epoch,
    fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                         'distance': args.distance, 'proj_head': proj_head, 'contrast_loss_fn': contrast_loss_fn},
)
