"""
Train MNIST
"""
from comet_ml import Experiment

import torch
from torchvision.datasets import mnist
from torchvision import transforms
from torch.utils.data import DataLoader
from ep_mlp import EPMLP
from fp_solver import FixedStepSolver, MaxGradNormSolver
from torch.utils.tensorboard import SummaryWriter
from time import time

# ARGS
BATCH_SIZE = 128
HIDDEN_SIZES = [500]
STEP_SIZE = 0.0
MAX_STEPS = 100
LR = 0.0
LOGGING_STEPS = 5
DEVICE = 'cuda'
EPOCHS = 35
MAX_GRAD_NORM = 10

# # GLOBAL stuff
# WRITER = SummaryWriter('./logs')


class RunningAvg:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def record(self, val, num):
        self.sum += val * num
        self.count += num

    def get_avg(self):
        return self.sum / self.count if self.count > 0 else 0.


class OneHot(object):
    def __init__(self, num_class):
        self.num_class = num_class

    def __call__(self, label):
        oh_vec = torch.Tensor(self.num_class).zero_()
        oh_vec[label] = 1.
        return oh_vec


class Logger:
    def __init__(self, tb_writer, comet_exp):
        self.tb_writer = tb_writer
        self.comet_exp = comet_exp

    def log_scalar(self, scalar_name, scalar, step):
        self.tb_writer.add_scalar(scalar_name, scalar, step)
        self.comet_exp.log_metric(scalar_name, scalar, step)


def get_data_loaders():
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.view(-1))])
    train_dset = mnist.MNIST(root='./mnist_data', train=True,
                             download=True,
                             transform=img_transform,
                             target_transform=OneHot(10))
    val_dset = mnist.MNIST(root='./mnist_data', train=False,
                           download=True,
                           transform=img_transform,
                           target_transform=OneHot(10))
    train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=BATCH_SIZE,
                            shuffle=False)
    return train_loader, val_loader


def get_model():
    model = EPMLP(784, 10, HIDDEN_SIZES, device=torch.device(DEVICE))
    solver = MaxGradNormSolver(step_size=STEP_SIZE, max_steps=MAX_STEPS, max_grad_norm=MAX_GRAD_NORM)
    return model, solver


def get_opt(model):
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    return opt


def get_avg_cost_and_corrects(free_states, labels, model):
    avg_costs = torch.mean(model.get_cost(free_states, labels))
    out = free_states[-1]
    preds = out.max(1)[1]
    trus = labels.max(1)[1]
    avg_corrects = (preds == trus).float().mean()
    return avg_costs.item(), avg_corrects.item()


def train(solver, model, opt, dataloader, global_step):
    acc_stats = RunningAvg()
    cost_stats = RunningAvg()
    device = model.device
    for imgs, labels in dataloader:
        imgs = imgs.to(device=device)
        labels = labels.to(device=device)
        start = time()

        init_states = None
        if USE_PREDICTORS:
            init_states = model.predict_initial_states(imgs)

        free_states = model.free_phase(imgs, solver, init_states=init_states)
        # TODO: Add init_states[-1] to clapm_phase?
        clamp_states = model.clamp_phase(imgs, labels, solver, 1,
                                         out=free_states[-1],
                                         hidden_units=free_states[:-1])

        LOGGER.log_scalar('solver_steps', solver._logs['steps_made'], global_step)
        LOGGER.log_scalar('solver_grad_norm', solver._logs['grad_norm'], global_step)
        if USE_PREDICTORS:
            # we do not use init_states here, but rather predict them again for simplicity
            # there are some gradient computation issues using init_states
            mean_predictor_loss = model.update_predictors(imgs, free_states)

            WRITER.add_scalar('train/mean_predictor_loss', mean_predictor_loss, global_step=global_step)
            EXP.log_metric('mean_predictor_loss', mean_predictor_loss, step=global_step)

        fp_time = time() - start

        opt.zero_grad()
        start = time()
        model.set_gradients(imgs, free_states, clamp_states)
        grad_time = time() - start
        opt.step()
        global_step += 1

        # Record stats and report
        with torch.no_grad():
            avg_cost, avg_corrects = get_avg_cost_and_corrects(free_states, labels, model)
        acc_stats.record(avg_corrects, imgs.size(0))
        cost_stats.record(avg_cost, imgs.size(0))
        if global_step % LOGGING_STEPS == 0:
            print('At step {}, cost: {:.4f}, acc: {:.2f}, '
                  'fp time: {:.3f}, grad time: {:.3f}'.format(global_step,
                                                              cost_stats.get_avg(),
                                                              acc_stats.get_avg() * 100, fp_time, grad_time))
            WRITER.add_scalar('train/cost', cost_stats.get_avg(), global_step=global_step)
            WRITER.add_scalar('train/acc', acc_stats.get_avg() * 100, global_step=global_step)
            WRITER.add_scalar('fp_time', fp_time, global_step=global_step)
            WRITER.add_scalar('grad_time', grad_time, global_step=global_step)

            EXP.log_metric('cost', cost_stats.get_avg(), step=global_step)
            EXP.log_metric('acc', acc_stats.get_avg() * 100, step=global_step)
            EXP.log_metric('fp_time', fp_time, step=global_step)
            EXP.log_metric('grad_time', grad_time, step=global_step)

            acc_stats.reset()
            cost_stats.reset()
    return global_step


def validate(solver, model, dataloader, global_step):
    acc_stats = RunningAvg()
    cost_stats = RunningAvg()
    device = model.device
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        init_states = None
        if USE_PREDICTORS:
            init_states = model.predict_initial_states(imgs)
            init_states = [s.detach().requires_grad_() for s in init_states]

        free_states = model.free_phase(imgs, solver, init_states=init_states)

        # Record stats and report
        with torch.no_grad():
            avg_cost, avg_corrects = get_avg_cost_and_corrects(free_states, labels, model)
        acc_stats.record(avg_corrects, imgs.size(0))
        cost_stats.record(avg_cost, imgs.size(0))
    print('At step {}, '
          'validation cost： {:.4f}, '
          'validation acc: {:.2f}'.format(global_step,
                                          cost_stats.get_avg(),
                                          acc_stats.get_avg() * 100))
    WRITER.add_scalar('valid/cost', cost_stats.get_avg(), global_step=global_step)
    WRITER.add_scalar('valid/acc', acc_stats.get_avg() * 100, global_step=global_step)

    EXP.log_metric('cost', cost_stats.get_avg(), step=global_step)
    EXP.log_metric('acc', acc_stats.get_avg() * 100, step=global_step)


def main():
    train_loader, val_loader = get_data_loaders()
    model, solver = get_model()
    opt = get_opt(model)
    print('Train on {}'.format(model.device))
    if USE_PREDICTORS: print('Using predictors')  # noqa E701
    global_step = 0
    epoch = 0
    while epoch < EPOCHS:
        EXP.set_epoch(epoch)
        with EXP.train():
            global_step = train(solver, model, opt, val_loader, global_step)
        with EXP.validate():
            validate(solver, model, val_loader, global_step)
        EXP.log_epoch_end(epoch)

        epoch += 1


if __name__ == '__main__':
    """ Main loop """

    for use_predictors in [True, False]:
        hparams = {
            'max_steps': MAX_STEPS,
            'use_predictors': use_predictors,
            'batch_size': BATCH_SIZE,
            'hidden_sizes': str(HIDDEN_SIZES),
            'step_size': STEP_SIZE,
            'lr': LR,
            'epochs': EPOCHS,
            'max_grad_norm': MAX_GRAD_NORM,
            'solver': 'MAX_GRAD_NORM'
        }

        EXP = Experiment(project_name='EqProp', auto_metric_logging=False)
        EXP.log_parameters(hparams)

        USE_PREDICTORS = use_predictors

        comment = f'{MAX_STEPS}_steps'
        if USE_PREDICTORS:
            comment += '_predictors'

        WRITER = SummaryWriter(comment=comment)

        LOGGER = Logger(WRITER, EXP)

        main()
        EXP.end()
