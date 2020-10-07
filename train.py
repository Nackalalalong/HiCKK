import numpy as np
import os
from torch.utils import data
import torch
import torch.optim as optim
from torch.autograd import Variable
from time import gmtime, strftime
import torch.nn as nn
from tqdm import tqdm
from model import OurNetV2
from torch.utils.data import DataLoader
from data import HicDataset

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping

use_gpu = 1

epochs = 100
HiC_max_value = 100
batch_size = 512

log_interval = 1000

def split_train_val(features, targets, train_size=0.7):
    targets = targets[:,np.newaxis,:,:]
    divide_point = int(features.shape[0] * train_size)
    feature_train, feature_val = features[:divide_point], features[divide_point:]
    target_train, target_val = targets[:divide_point], targets[divide_point:]

    train_loader = DataLoader(HicDataset(torch.from_numpy(feature_train),torch.from_numpy(target_train)), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(HicDataset(torch.from_numpy(feature_val),torch.from_numpy(target_val)), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def train(lowres,highres, outModel, startmodel=None,startepoch=0, down_sample_ratio=16):
    low_resolution_samples = lowres.astype(np.float32) * down_sample_ratio

    high_resolution_samples = highres.astype(np.float32)

    low_resolution_samples = np.minimum(HiC_max_value, low_resolution_samples)
    high_resolution_samples = np.minimum(HiC_max_value, high_resolution_samples)

    model = OurNetV2()

    sample_size = low_resolution_samples.shape[-1]
    padding = model.padding
    half_padding = padding // 2
    output_length = sample_size - padding
    Y = []
    for i in range(high_resolution_samples.shape[0]):
        no_padding_sample = high_resolution_samples[i][0][half_padding:(sample_size-half_padding) , half_padding:(sample_size - half_padding)]
        Y.append(no_padding_sample)
    Y = np.array(Y).astype(np.float32)

    train_loader, val_loader = split_train_val(low_resolution_samples, Y)

    if startmodel is not None:
        print('loading state dict from {}...'.format(startmodel))
        model.load_state_dict(torch.load(startmodel))
        print('finish loading state dict')

    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr = 0.00001)
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

    handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

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
            "Training Results - Epoch: {} Avg loss: {:.2f}".format(
                epoch, metrics['nll']
            )
        )

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()
