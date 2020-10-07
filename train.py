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

    # lowres_set = data.TensorDataset(torch.from_numpy(low_resolution_samples), torch.from_numpy(np.zeros(low_resolution_samples.shape[0])))
    # lowres_loader = DataLoader(lowres_set, batch_size=batch_size, shuffle=False)

    # hires_set = data.TensorDataset(torch.from_numpy(Y), torch.from_numpy(np.zeros(Y.shape[0])))
    # hires_loader = DataLoader(hires_set, batch_size=batch_size, shuffle=False)

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

    # running_loss = 0.0
    # running_loss_validate = 0.0
    # reg_loss = 0.0

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
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    evaluator.add_event_handler(Events.COMPLETED, handler)

    # @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    # def log_training_loss(trainer):
    #     print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

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

    # # write the log file to record the training process
    # with open('HindIII_train.txt', 'w') as log:
    #     for epoch in tqdm.tqdm(range(1+startepoch, epochs+1+startepoch)):
    #         for i, (v1, v2) in enumerate(zip(lowres_loader, hires_loader)):
    #             if (i == len(lowres_loader) - 1):
    #                 continue
    #             _lowRes, _ = v1
    #             _highRes, _ = v2
                
    #             _lowRes = Variable(_lowRes)
    #             _highRes = Variable(_highRes).unsqueeze(1)

    #             if use_gpu:
    #                 _lowRes = _lowRes.cuda()
    #                 _highRes = _highRes.cuda()
    #             optimizer.zero_grad()
    #             y_prediction = model(_lowRes)
		
    #             loss = _loss(y_prediction, _highRes)
    #             loss.backward()
    #             optimizer.step()
		
    #             running_loss += loss.item()

    #         if (epoch % 100 == 0 or epochs == 1):
    #             torch.save(model.state_dict(), outModel + str(epoch) + str('.model'))
        
    #     print('-------', i, epoch, running_loss/i, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
	
    #     log.write(str(epoch) + ', ' + str(running_loss/i,) +', '+ strftime("%Y-%m-%d %H:%M:%S", gmtime())+ '\n')
    #     running_loss = 0.0
    #     running_loss_validate = 0.0
            
