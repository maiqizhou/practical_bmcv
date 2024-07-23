import os
import time
import tqdm
import argparse
import torch
from torch.utils.data import random_split, DataLoader
import torchvision
import torchsummary

import logging
from torch.utils.tensorboard import SummaryWriter

from utils.utils import *

torch.autograd.set_detect_anomaly(True)

def training_loop(config):
    writer = SummaryWriter(f'logs/{config.log}')

    train_loader, test_loader = build_dataloader(config.batch_size, resize=config.resize, patch_size=config.patch_size, normalize=True)
    # train_loader, test_loader = build_CIFAR10_dataloader(config.batch_size, normalize=True)

    assert config.model in ['diffusion', 'diff_ae']
    if config.model == 'diffusion':
        from diffusion_model.diffusion import build_model
    elif config.model == 'diff_ae':
        from diff_ae.diffusion import build_model
    model = build_model(config.device, config.cond, config.timesteps)

    debug = config.debug
    if debug:
        info = f'Model: {model.model}\n Encoder: {model.encoder}\n'
        logging.info(info)
        # TODO: Add model summary
    if config.resume:
        model.load(f'{config.resume}.pt')

    currrent_best_loss = 1e9

    for epoch in range(config.epochs):
        start_time = time.time()

        if config.visualize:
            data = next(iter(test_loader))
            data = (data[0][0].unsqueeze(0), data[1][0].unsqueeze(0))
            data_img, x_recon = model.visualize(data, False)
            writer.add_image('Sample/data', torchvision.utils.make_grid(denormalize(data_img)), epoch)
            writer.add_image('Sample/x_recon', torchvision.utils.make_grid(denormalize(x_recon)), epoch)

        train_loss = {}
        for data in tqdm.tqdm(train_loader):
            # zero = torch.zeros(1)
            # data = (zero, data[0])
            loss = model.train(data)
            for key, value in loss.items():
                if key not in train_loss:
                    train_loss[key] = 0
                train_loss[key] += value.item()
        for key in train_loss:
            train_loss[key] /= len(train_loader)
            writer.add_scalar(f'Train/{key}', train_loss[key], epoch)

        training_info = f'Epoch {epoch} | Train Loss: {train_loss} | Time: {time.time() - start_time}'
        logging.info(training_info)
        print(training_info)

        flag = True

        test_loss = {}
        for data in tqdm.tqdm(test_loader):
            # zero = torch.zeros(1)
            # data = (zero, data[0])
            if flag:
                loss, noisy_x, x_0_pred, sampled_t = model.eval(data, True)
                noisy_x = add_text_to_image(denormalize(noisy_x), f't={sampled_t}')
                x_0_pred = denormalize(x_0_pred)
                dataimg = (data[0][0].unsqueeze(0), data[1][0].unsqueeze(0))

                writer.add_image(f'Visualization/data', torchvision.utils.make_grid(denormalize(torch.cat(dataimg, dim=0))), epoch)
                writer.add_image(f'Visualization/noisy_x', torchvision.utils.make_grid(noisy_x), epoch)
                writer.add_image(f'Visualization/x_0_pred', torchvision.utils.make_grid(x_0_pred), epoch)
                flag = False
            else:
                loss = model.eval(data)
            for key, value in loss.items():
                if key not in test_loss:
                    test_loss[key] = 0
                test_loss[key] += value.item()
        for key in test_loss:
            test_loss[key] /= len(test_loader)
            writer.add_scalar(f'Test/{key}', test_loss[key], epoch)
        total_loss = sum(test_loss.values())

        testing_info = f'Epoch {epoch} | Test Loss: {test_loss}'
        logging.info(testing_info)
        print(testing_info)

        if epoch % 50 == 0:
            model.save(f'saved_models/{config.log}_{epoch}.pt')

        if total_loss < currrent_best_loss:
            currrent_best_loss = total_loss
            model.save(f'saved_models/{config.log}_best.pt')
            logging.info(f'Best model saved at epoch {epoch}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model', type=str, default='diff_ae')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--log', type=str, default='diffae_final_3', help='Log session name')

    parser.add_argument('--cond', type=int, default=512, help='Condition dimension')
    parser.add_argument('--timesteps', type=int, default=500)
    parser.add_argument('--resize', type=int, default=None, help='Resize image to this size')
    parser.add_argument('--patch_size', type=int, default=128, help='Image patch size for training')

    config = parser.parse_args()

    if not os.path.isdir('saved_models'):
        os.makedirs('saved_models')

    if not os.path.exists('logs'):
        os.makedirs('logs')

    log = f'logs/{config.log}.log'
    logging.basicConfig(filename=log, level=logging.INFO)
    with open(log, 'w') as f:
        f.write(f'Config: {config}\n')

    if config.resume:
        print(f'Resuming training from {config.resume}')
    else:
        print('Training from scratch')

    training_loop(config)