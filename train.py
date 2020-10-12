import numpy as np
import os
from torch.utils import data
import torch
import torch.optim as optim
from torch.autograd import Variable
from time import gmtime, strftime
import torch.nn as nn
from tqdm import tqdm
from model import OurNet
from torch.utils.data import DataLoader
from data import HicDataset

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping
from ignite.engine import Engine

epochs = 100
HiC_max_value = 100
batch_size = 512

log_interval = 1000

def train(lowres, highres, val_lowres, val_hires, outModel, startmodel=None,startepoch=0, down_sample_ratio=16):
    low_resolution_samples = lowres.astype(np.float32) * down_sample_ratio
    high_resolution_samples = highres.astype(np.float32)

    val_lowres = val_lowres.astype(np.float32) * down_sample_ratio

    low_resolution_samples = np.minimum(HiC_max_value, low_resolution_samples)
    high_resolution_samples = np.minimum(HiC_max_value, high_resolution_samples)

    val_lowres = np.minimum(HiC_max_value, val_lowres)
    val_hires = np.minimum(HiC_max_value, val_hires)

    model = OurNet()

    sample_size = low_resolution_samples.shape[-1]
    padding = model.padding
    half_padding = padding // 2
    output_length = sample_size - padding
    Y = []
    for i in range(high_resolution_samples.shape[0]):
        no_padding_sample = high_resolution_samples[i][0][half_padding:(sample_size-half_padding) , half_padding:(sample_size - half_padding)]
        Y.append(no_padding_sample)
    Y = np.array(Y).astype(np.float32)
    Y = Y[:,np.newaxis,:,:]

    val_Y = []
    for i in range(val_hires.shape[0]):
        no_padding_sample = val_hires[i][0][half_padding:(sample_size-half_padding) , half_padding:(sample_size - half_padding)]
        val_Y.append(no_padding_sample)
    val_Y = np.array(val_Y).astype(np.float32)
    val_Y = val_Y[:,np.newaxis,:,:]

    train_loader = DataLoader(HicDataset(torch.from_numpy(low_resolution_samples), torch.from_numpy(Y)), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(HicDataset(torch.from_numpy(val_lowres), torch.from_numpy(val_Y)), batch_size=batch_size, shuffle=True, drop_last=True)

    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    model.to(device)

    if startmodel is not None:
        print('loading state dict from {}...'.format(startmodel))
        model.load_state_dict(torch.load(startmodel))
        print('finish loading state dict')

    optimizer = optim.SGD(model.parameters(), lr = 0.00001, momentum=0.9, weight_decay=0.0001)
    criterion = nn.MSELoss()
    model.train()

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    val_metrics = {
        "nll": Loss(criterion)
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    desc = "EPOCH - val loss: {:.2f}"
    pbar = tqdm(initial=0, leave=False, total=epochs, desc=desc.format(0))

    def score_function(engine):
        val_loss = engine.state.metrics['nll']
        return -val_loss

    handler = EarlyStopping(patience=15, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    @trainer.on(Events.COMPLETED)
    def save_model(trainer):
        epoch = trainer.state.epoch + startepoch
        torch.save(model.state_dict(), outModel + str(epoch+1) + str('.model'))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        epoch = trainer.state.epoch + startepoch
        tqdm.write(
            "Training Results - Epoch: {} Avg loss: {:.2f}".format(
                epoch, metrics['nll']
            )
        )
        if trainer.state.epoch % 99 == 0 and epoch > 0:
            torch.save(model.state_dict(), outModel + str(epoch+1) + str('.model'))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        epoch = trainer.state.epoch + startepoch
        epoch = trainer.state.epoch + startepoch
        pbar.desc = desc.format(metrics['nll'])
        pbar.update(1)
        tqdm.write(
            "Validating Results - Epoch: {} Avg loss: {:.2f}".format(
                epoch, metrics['nll']
            )
        )

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()
