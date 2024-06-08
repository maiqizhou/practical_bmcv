import os
import time
import tqdm
import argparse
import torch
import torchvision

import logging
from torch.utils.tensorboard import SummaryWriter

from diffusion.diffusion import build_model
from data.paired_dataset import PairedDataset
from utils.utils import *

def build_dataloader(batch_size, resize=128, normalize=True):
    transform = get_transforms(resize, normalize)

    train_dataset = PairedDataset('data/BCI_dataset/HE/train', 'data/BCI_dataset/IHC/train', transform)
    test_dataset = PairedDataset('data/BCI_dataset/HE/test', 'data/BCI_dataset/IHC/test', transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def training_loop(config):
    writer = SummaryWriter(f'logs/{config.log}')

    train_loader, test_loader = build_dataloader(config.batch_size)
    model = build_model(config.model, config.device)

    debug = config.debug

    if config.resume:
        model.load(f'{config.resume}.pt')

    for epoch in range(config.epochs):
        start_time = time.time()
        train_loss = 0
        test_loss = 0

        for data in tqdm.tqdm(train_loader):
            loss = model.train(data)
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        training_info = f'Epoch {epoch} | Train Loss: {avg_train_loss} | Time: {time.time() - start_time}'
        logging.info(training_info)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        print(training_info)

        flag = True

        for data in tqdm.tqdm(test_loader):
            if flag:
                loss, noisy_x, noisy_x_pred, sampled_t = model.eval(data, True)
                noisy_x = add_text_to_image(denormalize(noisy_x), f't={sampled_t}')
                writer.add_image(f'Visualization/noisy_x', torchvision.utils.make_grid(noisy_x), epoch)
                writer.add_image(f'Visualization/noisy_x_pred', torchvision.utils.make_grid(denormalize(noisy_x_pred)), epoch)
                flag = False
            else:
                loss = model.eval(data)
            test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)

        testing_info = f'Epoch {epoch} | Test Loss: {test_loss}'
        logging.info(testing_info)
        writer.add_scalar('Loss/test', avg_test_loss, epoch)
        print(testing_info)

        if config.visualize:
            data = next(iter(test_loader))
            data = (data[0][0].unsqueeze(0), data[1][0].unsqueeze(0))
            cond, x_start, x_recon = model.visualize(data, False)
            writer.add_image('Visualization/cond', torchvision.utils.make_grid(denormalize(cond)), epoch)
            writer.add_image('Visualization/x_start', torchvision.utils.make_grid(denormalize(x_start)), epoch)
            writer.add_image('Visualization/x_recon', torchvision.utils.make_grid(denormalize(x_recon)), epoch)

        if epoch % 50 == 0:
            model.save(f'saved_models/{config.log}_{epoch}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model', type=str, default='diffusion')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--log', type=str, default='test', help='Log session name')

    config = parser.parse_args()

    if not os.path.isdir('saved_models'):
        os.makedirs('saved_models')

    if not os.path.exists('logs'):
        os.makedirs('logs')

    log = f'logs/{config.log}.log'
    logging.basicConfig(filename=log, level=logging.INFO)
    with open(log, 'w') as f:
        f.write(f'Config: {config}\n')

    training_loop(config)