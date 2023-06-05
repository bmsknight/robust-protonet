"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
import argparse

from torch.optim import Adam
from torch.utils.data import DataLoader

from config import PATH
from few_shot.attack import PGDAttackWrapperForTraining
from few_shot.callbacks import *
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.models import get_few_shot_he_encoder
from few_shot.proto import proto_net_episode
from few_shot.train import fit
from few_shot.utils import setup_dirs
from few_shot.arc_margin import ArcFace

setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="miniImageNet")
parser.add_argument('--attack-type', default="query")
parser.add_argument('--adv-train-type', default="")
parser.add_argument('--distance', default='cosine')
parser.add_argument('--scale', default=1, type=float)
parser.add_argument('--n-train', default=5, type=int)
parser.add_argument('--n-test', default=5, type=int)
parser.add_argument('--k-train', default=20, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=15, type=int)
parser.add_argument('--q-test', default=1, type=int)
args = parser.parse_args()

evaluation_episodes = 1000
episodes_per_epoch = 100

if args.dataset == 'omniglot':
    n_epochs = 40
    dataset_class = OmniglotDataset
    num_input_channels = 1
    final_layer_size = 64
    drop_lr_every = 20
elif args.dataset == 'miniImageNet':
    n_epochs = 80
    dataset_class = MiniImageNet
    num_input_channels = 3
    final_layer_size = 1600
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
model = get_few_shot_he_encoder(num_input_channels, final_layer_size)
model.to(device, dtype=torch.float)
metric = ArcFace(s=args.scale, margin=0.5)
metric.to(device, dtype=torch.float)
pgd_attack = PGDAttackWrapperForTraining(model, distance=args.distance, n_shot=args.n_train, k_way=args.k_train,
                                         is_he_model=True, eps=8 / 255, alpha=2 / 255, steps=7, random_start=True,
                                         attack_type=args.attack_type)

############
# Training #
############
print(f'Training Prototypical network on {args.dataset}...')
optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.NLLLoss().cuda()


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
        distance=args.distance,
        is_he_model=True
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/proto_nets/adv{args.adv_train_type}_lin_arc_{args.scale}_{param_str}.pth',
        monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'
    ),
    LearningRateScheduler(schedule=lr_schedule),
    CSVLogger(PATH + f'/logs/proto_nets/adv{args.adv_train_type}_lin_arc_{args.scale}_{param_str}.csv'),
    ArcFaceMarginScheduler(arc_head=metric, max_epoch=n_epochs)
]

fit(
    model,
    optimiser,
    loss_fn,
    epochs=n_epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=proto_net_episode,
    attack_fn=pgd_attack,
    adv_train_type=args.adv_train_type,
    fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                         'distance': args.distance, 'is_he_model': True, 'arc_head': metric},
)
