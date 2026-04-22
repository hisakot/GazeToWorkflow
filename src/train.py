import argparse
import glob
import os
import time
from tqdm import tqdm

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataloader
from torch.utils.tensorboard import SummaryWriter

import config
import dataset

def train(train_dataloader, model, loss_function, optimizer, device):
    model.train()
    train_loss = 0
    
    with tqdm(total=len(train_dataloader)) as pbar:
        for data in train_dataloader:
            inputs = data[0].to(device)
            tagets = data[1].to(device)
            if inputs is None or targets is None:
                continue
            pred = model(inputs)
            loss = lossfunction(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.update()

    return total_loss / len(train_dataloader)

def evaluate(val_dataloader, model, loss_function, device):
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        with tqdm(total=len(val_dataloader)) as pbar:
            for data in val_dataloader:
                inputs = data[0].to(device)
                tagets = data[1].to(device)
                if inputs is None or targets is None:
                    continue
                pred = model(inputs)
                loss = lossfunction(pred, targets)
            val_loss += loss.item()
            pbar.update()

    return val_loss / len(val_dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", required=False, default=1, type=int)
    parser.add_argument("--checkpoint", required=False,
                        help="if you want to retry training, write model path")
    args = parser.parse_args()
    batch_size = atgs.batch_size

    cfg = config.Config()
    lr = cfg.lr
    epochs = cfg.epochs
    img_height = cfg.img_height
    img_width = cfg.img_width
    train_data_dir = cfg.train_data_dir
    val_data_dir = cfg.val_data_dir

    # model = 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() >= 2:
        print("---------- Use multiple GPUs ----------")
    else:
        print(f"---------- Use {device} ----------")
    model.to(device)

    # loss_function = 
    # optimizer = 
    # scheduler = 

    writer = SummaryWriter(log_dir="logs")

    num_cpu = os.cpu_count()
    num_cpu = num_cpu // 4
    print(f"number of cpu: {num_cpu}")

    train_loss_list = list()
    val_loss_list = list()

    train_data = dataset.Dataset(train_data_dir, img_height=img_height, img_width= img_width,
                                 transform=None, is_train=True, inf_rotate=None)
    val_data = dataset.Dataset(val_data_dir, img_height=img_height, img_width= img_width,
                               transform=None, is_train=False, inf_rotate=None)

    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=num_cpu, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size,
                                shuffle=False, num_workers=num_cpu, pin_memory=True)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_loss_list = checkpoint["train_loss_list"]
        val_loss_list = checkpoint["val_loss_list"]
        for i, train_loss in enumerate(train_loss_list):
            writer.add_scalar("Train Loss", train_loss, i+1)
        for i, val_loss in enumerate(val_loss_list):
            writer.add_scalar("Validation Loss", val_loss, i+1)
        print(f"Reload model: epoch {start_epoch} and restart training")
    else:
        start_epoch = 0

    early_stopping = [np.inf, 50, 0]
    for epoch in range(epochs):
        epoch += start_epoch
        print(f"Epoch: {epoch}")
        print(f"Early stopping: {early_stopping}")
        print(f"lr: {scheduler.get_latest_lr()[0]}")

        try:
        # train
        train_loss = train(train_loader, model, loss_function, optimizer, device)
        train_loss_list.append(train_loss)

        # val
        with torch.no_grad():
            val_loss = val(val_loader, model, loss_function, optimizer, device)
            val_loss_list.append(val_loss)

        print("Epoch %d : train_loss &.3f" % (epoch, train_loss))
        print("Epoch %d : valid_loss &.3f" % (epoch, val_loss))

        # lr_scheduler
        scheduler.step()

        # save_model
        if val_loss < early_stopping[0]:
            early_stopping[0] = val_loss
            early_stopping[2] = 0
            torch.save({"epoch" : epoch,
                        "model_state_dict" : model.state_dict(),
                        "optimizer_state_dict" : optimizer.state_dict(),
                        "train_loss_list" : train_loss_list,
                        "val_loss_list" : val_loss_list,
                        }, "models/newest_model.pth")
        else:
            early_stopping[2] += 1
            if early_stopping[2] == early_stopping[1]:
                break

        # tensorboard
        writer.add_scalar("Train Loss", train_loss, epoch)
        writer.add_scalar("Valid Loss", val_loss, epoch)
        print("log updated")
        
    except ValueError:
        continue

if __name__ == "__main__":
    main()
