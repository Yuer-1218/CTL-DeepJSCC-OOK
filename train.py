#train.py
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  17:00:00 2023

@author: chun
"""
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model import DeepJSCC, ratio2filtersize
from torch.nn.parallel import DataParallel
from utils import image_normalization, set_seed, save_model, view_model_param
from fractions import Fraction
from dataset import Vanilla
import numpy as np
import time
from tensorboardX import SummaryWriter
import glob
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def train_epoch(model, optimizer, param, data_loader):
    model.train()
    epoch_loss = 0

    for iter, (images, _) in enumerate(data_loader):
        images = images.cuda() if param['parallel'] and torch.cuda.device_count(
        ) > 1 else images.to(param['device'])
        optimizer.zero_grad()
        outputs = model.forward(images)
        
        # 修正：直接在 [0, 1] 范围内计算损失，移除了反归一化步骤
        loss = model.loss(images, outputs) if not param['parallel'] else model.module.loss(
            images, outputs)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)

    return epoch_loss, optimizer


def evaluate_epoch(model, param, data_loader):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for iter, (images, _) in enumerate(data_loader):
            images = images.cuda() if param['parallel'] and torch.cuda.device_count(
            ) > 1 else images.to(param['device'])
            outputs = model.forward(images)

            # 修正：直接在 [0, 1] 范围内计算损失，移除了反归一化步骤
            loss = model.loss(images, outputs) if not param['parallel'] else model.module.loss(
                images, outputs)

            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)

    return epoch_loss


def config_parser_pipeline():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'imagenet'], help='dataset')
    parser.add_argument('--out', default='./out', type=str, help='out_path')
    parser.add_argument('--disable_tqdm', default=False, type=bool, help='disable_tqdm')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--parallel', default=False, type=bool, help='parallel')
    parser.add_argument('--snr_list', default=['19', '13', '7', '4', '1'], nargs='+', help='snr_list')
    parser.add_argument('--ratio_list', default=['1/6', '1/12'], nargs='+', help='ratio_list')
    parser.add_argument('--channel', default='AWGN', type=str,
                        choices=['AWGN', 'Rayleigh', 'BSC'], help='channel')
    parser.add_argument('--ber', default=0.01, type=float, help='BER for BSC channel')
    return parser.parse_args()


def main_pipeline():
    args = config_parser_pipeline()
    print("Training Start")
    dataset_name = args.dataset
    out_dir = args.out
    args.snr_list = list(map(float, args.snr_list))
    args.ratio_list = list(map(lambda x: float(Fraction(x)), args.ratio_list))
    params = {}
    params['disable_tqdm'] = args.disable_tqdm
    params['dataset'] = dataset_name
    params['out_dir'] = out_dir
    params['device'] = args.device
    params['ratio_list'] = args.ratio_list
    params['channel'] = args.channel
    if dataset_name == 'cifar10':
        params['batch_size'] = 64
        params['num_workers'] = 4
        params['epochs'] = 1000
        params['init_lr'] = 1e-3
        params['weight_decay'] = 5e-4
        params['parallel'] = False
        params['if_scheduler'] = True
        params['step_size'] = 640
        params['gamma'] = 0.1
        params['seed'] = 42
        params['ReduceLROnPlateau'] = False
        params['lr_reduce_factor'] = 0.5
        params['lr_schedule_patience'] = 15
        params['max_time'] = 12
        params['min_lr'] = 1e-5
    elif dataset_name == 'imagenet':
        params['batch_size'] = 32
        params['num_workers'] = 4
        params['epochs'] = 300
        params['init_lr'] = 1e-4
        params['weight_decay'] = 5e-4
        params['parallel'] = True
        params['if_scheduler'] = True
        params['gamma'] = 0.1
        params['seed'] = 42
        params['ReduceLROnPlateau'] = True
        params['lr_reduce_factor'] = 0.5
        params['lr_schedule_patience'] = 15
        params['max_time'] = 12
        params['min_lr'] = 1e-5
    else:
        raise Exception('Unknown dataset')

    set_seed(params['seed'])

    if args.channel == 'BSC':
        # 对于 BSC，我们不遍历 SNR 列表，只使用指定的 BER
        print(f"Channel is BSC. Running experiments for BER: {args.ber}")
        params['ber'] = args.ber
        for ratio in params['ratio_list']:
            current_params = params.copy()
            current_params['ratio'] = ratio
            # 为文件名和日志设置一个占位符，避免混淆
            current_params['snr'] = 'NA_BSC' 
            train_pipeline(current_params)
    else:
        # 对于 AWGN 或 Rayleigh，我们遍历 SNR 列表
        print(f"Channel is {args.channel}. Running experiments for SNRs: {args.snr_list}")
        params['snr_list'] = args.snr_list
        for ratio in params['ratio_list']:
            for snr in params['snr_list']:
                current_params = params.copy()
                current_params['ratio'] = ratio
                current_params['snr'] = snr
                train_pipeline(current_params)


def train_pipeline(params):

    # --- 早停参数 ---
    patience = 25  # 连续 25 个 epoch 验证损失没有改善就停止
    best_val_loss = float('inf')
    patience_counter = 0
    # --- 早停参数结束 ---

    dataset_name = params['dataset']
    # load data
    if dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), ])
        train_dataset = datasets.CIFAR10(root='../dataset/', train=True,
                                         download=True, transform=transform)

        train_loader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=params['batch_size'], num_workers=params['num_workers'])
        test_dataset = datasets.CIFAR10(root='../dataset/', train=False,
                                        download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=params['batch_size'], num_workers=params['num_workers'])

    elif dataset_name == 'imagenet':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((128, 128))])
        print("loading data of imagenet")
        train_dataset = datasets.ImageFolder(root='../dataset/ImageNet/train', transform=transform)

        train_loader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=params['batch_size'], num_workers=params['num_workers'])
        test_dataset = Vanilla(root='../dataset/ImageNet/val', transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=params['batch_size'], num_workers=params['num_workers'])
    else:
        raise Exception('Unknown dataset')

    # create model
    image_fisrt = train_dataset.__getitem__(0)[0]
    c = ratio2filtersize(image_fisrt, params['ratio'])
    
    if params['channel'] == 'BSC':
        print("The BER is {}, the inner channel is {}, the ratio is {:.2f}".format(
            params['ber'], c, params['ratio']))
        model = DeepJSCC(c=c, channel_type=params['channel'], ber=params['ber'])
    else:
        print("The snr is {}, the inner channel is {}, the ratio is {:.2f}".format(
            params['snr'], c, params['ratio']))
        model = DeepJSCC(c=c, channel_type=params['channel'], snr=params['snr'])

    # init exp dir
    out_dir = params['out_dir']
    
    if params['channel'] == 'BSC':
        channel_param_str = f"ber{params['ber']}"
    else:
        channel_param_str = f"snr{params['snr']}"

    phaser = f"{dataset_name.upper()}_{c}_{channel_param_str}_{params['ratio']:.2f}_{params['channel']}_{time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')}"

    root_log_dir = out_dir + '/' + 'logs/' + phaser
    root_ckpt_dir = out_dir + '/' + 'checkpoints/' + phaser
    root_config_dir = out_dir + '/' + 'configs/' + phaser
    writer = SummaryWriter(log_dir=root_log_dir)

    # model init
    device = torch.device(params['device'] if torch.cuda.is_available() else 'cpu')
    if params['parallel'] and torch.cuda.device_count() > 1:
        model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model = model.cuda()
    else:
        model = model.to(device)

    # opt
    optimizer = optim.Adam(
        model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    if params['if_scheduler'] and not params['ReduceLROnPlateau']:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=params['step_size'], gamma=params['gamma'])
    elif params['ReduceLROnPlateau']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=params['lr_reduce_factor'],
                                                         patience=params['lr_schedule_patience'],
                                                         verbose=False)
    else:
        print("No scheduler")
        scheduler = None

    writer.add_text('config', str(params))
    t0 = time.time()
    epoch_train_losses, epoch_val_losses = [], []
    per_epoch_time = []

    # train
    try:
        with tqdm(range(params['epochs']), disable=params['disable_tqdm']) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, optimizer = train_epoch(
                    model, optimizer, params, train_loader)

                epoch_val_loss = evaluate_epoch(model, params, test_loader)

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss)

                per_epoch_time.append(time.time() - start)

                # --- 早停逻辑开始 ---
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    patience_counter = 0
                    # 保存性能最好的模型
                    if not os.path.exists(root_ckpt_dir):
                        os.makedirs(root_ckpt_dir)
                    torch.save(model.state_dict(), os.path.join(root_ckpt_dir, "best_model.pkl"))
                    t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                                  train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                  best_val_loss=best_val_loss) # 更新tqdm显示
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
                    break
                # --- 早停逻辑结束 ---

                # (可选) 您可以保留原来的保存最新模型的逻辑，或者为了简洁而删除它
                # 如果保留，您将同时拥有 best_model.pkl 和最新的 epoch_N.pkl
                # 这里我暂时注释掉它，因为早停逻辑已经保存了最好的模型
                """
                if not os.path.exists(root_ckpt_dir):
                    os.makedirs(root_ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(
                    root_ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(root_ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch - 1:
                        os.remove(file)
                """

                if params['ReduceLROnPlateau'] and scheduler is not None:
                    scheduler.step(epoch_val_loss)
                elif params['if_scheduler'] and not params['ReduceLROnPlateau']:
                    scheduler.step()

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                if time.time() - t0 > params['max_time'] * 3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(
                        params['max_time']))
                    break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    # --- 重要：加载最好的模型进行最终评估 ---
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(root_ckpt_dir, "best_model.pkl")))
    # ---

    test_loss = evaluate_epoch(model, params, test_loader)
    train_loss = evaluate_epoch(model, params, train_loader)
    print("Test Loss (from best model): {:.4f}".format(test_loss))
    print("Train Loss (from best model): {:.4f}".format(train_loss))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    """
        Write the results in out_dir/results folder
    """

    writer.add_text(tag='result', text_string="""Dataset: {}\nparams={}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS (from best model)\nTEST Loss: {:.4f}\nTRAIN Loss: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""
                    .format(dataset_name, params, view_model_param(model), test_loss,
                            train_loss, epoch, (time.time() - t0) / 3600, np.mean(per_epoch_time)))
    writer.close()
    if not os.path.exists(os.path.dirname(root_config_dir)):
        os.makedirs(os.path.dirname(root_config_dir))
    with open(root_config_dir + '.yaml', 'w') as f:
        dict_yaml = {'dataset_name': dataset_name, 'params': params,
                     'inner_channel': c, 'total_parameters': view_model_param(model)}
        import yaml
        yaml.dump(dict_yaml, f)

    del model, optimizer, scheduler, train_loader, test_loader
    del writer



def train(args, ratio: float, snr: float):  # deprecated
    dataset_name = args.dataset
    # load data
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.CIFAR10(root='../dataset/', train=True,
                                         download=True, transform=transform)

        train_loader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=args.batch_size, num_workers=args.num_workers)
        test_dataset = datasets.CIFAR10(root='../dataset/', train=False,
                                        download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=args.batch_size, num_workers=args.num_workers)

    elif dataset_name == 'imagenet':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128))
        ])
        print("loading data of imagenet")
        train_dataset = datasets.ImageFolder(root='../dataset/ImageNet/train', transform=transform)

        train_loader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=args.batch_size, num_workers=args.num_workers)
        test_dataset = Vanilla(root='../dataset/ImageNet/val', transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise Exception('Unknown dataset')

    # create model
    image_fisrt = train_dataset.__getitem__(0)[0]
    c = ratio2filtersize(image_fisrt, ratio)
    print("The snr is {}, the inner channel is {}, the ratio is {:.2f}".format(snr, c, ratio))
    model = DeepJSCC(c=c, channel_type='AWGN', snr=snr)
    # init exp dir
    phaser = "{}_{}_{}_{}_{}".format(dataset_name.upper(), c, snr, ratio, time.strftime(
        '%Hh%Mm%Ss_on_%b_%d_%Y'))
    root_log_dir = args.out_dir + '/' + 'logs/' + phaser
    root_ckpt_dir = args.out_dir + '/' + 'checkpoints/' + phaser
    root_config_dir = args.out_dir + '/' + 'configs/' + phaser
    writer = SummaryWriter(log_dir=root_log_dir)

    # model init
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.parallel and torch.cuda.device_count() > 1:
        model = DataParallel(model, device_ids=[0, 1])
        model = model.cuda()
    else:
        model = model.to(device)

    # opt
    optimizer = optim.Adam(
        model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    if args.if_scheduler:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None

    writer.add_text('config', str(args))
    t0 = time.time()
    epoch_train_losses, epoch_val_losses = [], []
    per_epoch_time = []

    # train
    for epoch in range(args.epochs):

        start = time.time()

        epoch_train_loss, optimizer = train_epoch(
            model, optimizer, args, train_loader)

        epoch_val_loss = evaluate_epoch(model, args, test_loader)

        epoch_train_losses.append(epoch_train_loss)
        epoch_val_losses.append(epoch_val_loss)

        writer.add_scalar('train/_loss', epoch_train_loss, epoch)
        writer.add_scalar('val/_loss', epoch_val_loss, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print('epoch %d, lr %.6f, train_loss %.5f, val_loss %.5f, time %.2fs' %
              (epoch, optimizer.param_groups[0]['lr'], epoch_train_loss, epoch_val_loss, time.time() - start))
        per_epoch_time.append(time.time() - start)

        # Saving checkpoint
        save_model(epoch, epoch_val_losses, model, root_ckpt_dir)

        if args.if_scheduler:
            scheduler.step()

    test_loss = evaluate_epoch(model, args, test_loader)
    train_loss = evaluate_epoch(model, args, train_loader)
    print("Test Accuracy: {:.4f}".format(test_loss))
    print("Train Accuracy: {:.4f}".format(train_loss))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    """
        Write the results in out_dir/results folder
    """

    writer.add_text(tag='result', text_string="""Dataset: {}\nArgs={}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST Loss: {:.4f}\nTRAIN Loss: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""
                    .format(dataset_name, args, view_model_param(model), np.mean(np.array(train_loss)),
                            np.mean(np.array(test_loss)), epoch, (time.time() - t0) / 3600, np.mean(per_epoch_time)))
    writer.close()
    if not os.path.exists(os.path.dirname(root_config_dir)):
        os.makedirs(os.path.dirname(root_config_dir))
    with open(root_config_dir + '.yaml', 'w') as f:
        dict_yaml = {'dataset_name': dataset_name, 'args': args,
                     'inner_channel': c, 'total_parameters': view_model_param(model)}
        import yaml
        yaml.dump(dict_yaml, f)

    del model, optimizer, scheduler, train_loader, test_loader
    del writer


def config_parser():  # deprecated
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10',
                        choices=['cifar10', 'imagenet'])
    parser.add_argument('--out_dir', default='./out')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--parallel', default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--init_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--if_scheduler', default=True)
    parser.add_argument('--step_size', type=int, default=640)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():  # deprecated
    args = config_parser()
    print("Training Start")
    snr_list = [19, 13, 7, 4, 1]
    ratio_list = [1 / 6, 1 / 12]
    for ratio in ratio_list:
        for snr in snr_list:
            train(args, ratio, snr)


if __name__ == '__main__':
    main_pipeline()
