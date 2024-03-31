
import easydict
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.models import LSTMAE, LSTMVAE
from utils.MovingMNIST import MovingMNIST
from utils.dataset import SinusoidalDataset, MultiSinusoidalDataset
writer = SummaryWriter()

from s4torch import S4Model

def train(args, model, train_loader, test_loader):
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    ## interation setup
    epochs = tqdm(range(args.max_iter // len(train_loader) + 1))

    ## training
    count = 0
    for epoch in epochs:
        print(f'Epoch {epoch+1}/{len(epochs)}')
        model.train()
        optimizer.zero_grad()
        train_iterator = tqdm(
            enumerate(train_loader), total=len(train_loader), desc="training"
        )

        for i, batch_data in train_iterator:

            if count > args.max_iter:
                return model
            count += 1

            #future_data, past_data = batch_data
            inputs, labels = batch_data['input'].to(args.device), batch_data['label'].to(args.device)
            
            ## reshape
            batch_size = inputs.size(0)
            example_size = inputs.size(1)

            mloss, recon_x, info = model(inputs)

            # Backward and optimize
            optimizer.zero_grad()
            mloss.mean().backward()
            optimizer.step()

            train_iterator.set_postfix({"train_loss": float(mloss.mean())})
        writer.add_scalar("train_loss", float(mloss.mean()), epoch)

        model.eval()　
        eval_loss = 0
        test_iterator = tqdm(
            enumerate(test_loader), total=len(test_loader), desc="testing"
        )

        with torch.no_grad():
            for i, batch_data in test_iterator:
                inputs, labels = batch_data['input'].to(args.device), batch_data['label'].to(args.device)

                mloss, recon_x, info = model(inputs)

                eval_loss += mloss.mean().item()

                test_iterator.set_postfix({"eval_loss": float(mloss.mean())})


        eval_loss = eval_loss / len(test_loader)
        writer.add_scalar("eval_loss", float(eval_loss), epoch)
        print("Evaluation Score : [{:.3g}]".format(eval_loss))

    return model