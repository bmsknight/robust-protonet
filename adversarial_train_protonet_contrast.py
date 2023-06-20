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
from few_shot.models import get_few_shot_encoder, SupConProjHead
from few_shot.proto import proto_net_episode_contrast
from few_shot.train import fit
from few_shot.utils import setup_dirs, SupConLoss

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
parser.add_argument('--adv-train-type', default="joint")
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=5, type=int)
parser.add_argument('--n-test', default=5, type=int)
parser.add_argument('--k-train', default=20, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=15, type=int)
parser.add_argument('--q-test', default=1, type=int)
args = parser.parse_args()
print('Arguments:', args)
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
    final_layer_size = 16000 #1600
    drop_lr_every = 40
else:
    raise (ValueError, 'Unsupported dataset')

param_str = f'{args.adv_train_type}_{args.k_test}way_{args.n_test}shot'

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
model.to(device, dtype=torch.float)

proj_head = SupConProjHead(dim_in=final_layer_size)
proj_head.to(device, dtype=torch.float)

pgd_attack = PGDAttackWrapperForTraining(model, distance=args.distance, n_shot=args.n_train, k_way=args.k_train,
                                         is_he_model=False, eps=8 / 255, alpha=2 / 255, steps=7, random_start=True,
                                         attack_type=args.attack_type)

############
# Training #
############
print(f'Training Prototypical network on {args.dataset}...')
optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.NLLLoss().cuda()
contrast_loss_fn = SupConLoss().cuda()


def lr_schedule(epoch, lr):
    # Drop lr every 2000 episodes
    if epoch % drop_lr_every == 0:
        return lr / 2
    else:
        return lr


callbacks = [
    EvaluateFewShot(
        eval_fn=proto_net_episode_contrast,
        num_tasks=evaluation_episodes,
        n_shot=args.n_test,
        k_way=args.k_test,
        q_queries=args.q_test,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
        distance=args.distance
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/proto_nets/adv_contrast_{param_str}.pth',
        monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'
    ),
    LearningRateScheduler(schedule=lr_schedule),
    CSVLogger(PATH + f'/logs/proto_nets/adv_contrast_{param_str}.csv'),
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
    fit_function=proto_net_episode_contrast,
    attack_fn=pgd_attack,
    adv_train_type=args.adv_train_type,
    fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                         'distance': args.distance, 'is_contrast_model': True, 'proj_head': proj_head, 'contrast_loss_fn': contrast_loss_fn}
)

