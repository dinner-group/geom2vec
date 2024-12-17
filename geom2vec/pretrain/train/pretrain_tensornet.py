import argparse
import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ...pretrain.datasets.denali import DenaliDataset
from ...representation_models.torchmd.main_model import create_model, get_args

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--num_rbf', type=int, default=64)
parser.add_argument('--cutoff', type=float, default=7.5)
parser.add_argument('--rep_model', type=str, default='tensornet')
parser.add_argument('--batchsize', type=int, default=200)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('noise_level', type=float, default=0.2)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_dir', type=str, default='./data/denali')
parser.add_argument('--save_dir', type=str, default='.')
args = parser.parse_args()


class NoiseTransfer(object):
    def __init__(self, noise_level):
        self.noise_level = noise_level

    def __call__(self, data):
        data.noise = self.noise_level * torch.randn_like(data.pos)
        data.pos_noise = data.pos + data.noise
        return data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


path = args.data_dir
dataset = DenaliDataset(root=path, transform=NoiseTransfer(args.noise_level)).shuffle()

device = torch.device(args.device)

hidden_channels = args.hidden_channels
num_layers = args.num_layers
num_heads = args.num_heads
num_rbf = args.num_rbf
cutoff = args.cutoff
batchsize = args.batchsize

model_args = get_args(hidden_channels=hidden_channels,
 num_layers=num_layers,
 num_rbf=num_rbf,
 num_heads=num_heads,
 cutoff=cutoff,
 rep_model='tensornet')

model = create_model(args=model_args).to(device)

# Current timestamp for unique log directory
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_dir_name = f"tensor_{hidden_channels}hc_{num_layers}l_{current_time}_cutoff_{cutoff}_rbf_{num_rbf}"
log_dir = args.save_dir + f'/logs_models_denali_{args.noise_level}/tensorboard_logs/{log_dir_name}'

print(f"Logging to {log_dir}")
print(model)
print(f"Total trainable parameters: {count_parameters(model)}")

os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# Log model details and parameters
model_details = f"{model}\n\nTotal trainable parameters: {count_parameters(model)}"
writer.add_text("Model/Details", model_details, 0)


valid_dataset = dataset[:10000]
train_dataset = dataset[10000:]
train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=int(2*args.batchsize), shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,amsgrad=True)
noise_loss = torch.nn.MSELoss()

best_valid_loss = float('inf')


def train(train_loader, valid_loader, epoch):
    global best_valid_loss
    model.train()
    total_loss = 0
    step_global = (epoch - 1) * len(train_loader)

    # Initialize the tqdm progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training")

    for batch_idx, data in enumerate(progress_bar):
        optimizer.zero_grad()
        data = data.to(device)
        out, _, vec_out = model(pos=data.pos_noise, z=data.z, batch=data.batch)
        noise = model.pos_normalizer(data.noise.to(device))
        loss = noise_loss(vec_out, noise)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # Calculate the average loss
        avg_loss = total_loss / (batch_idx + 1)

        # Update the progress bar description with the current average loss
        progress_bar.set_description(f"Epoch {epoch} Training, Avg Loss: {avg_loss:.4f}")

        writer.add_scalar('Loss/Train_Batch', loss.item(), step_global + batch_idx)

        if (batch_idx + 1) % 1000 == 0:
            valid_loss = validate(valid_loader, step_global + batch_idx, show_progress=False)
            writer.add_scalar('Loss/Validation_Steps', valid_loss, step_global + batch_idx)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(args.save_dir, f'tensor_best_epoch{epoch}.pth'))
                tqdm.write(
                    f'Model saved as tensor_best.pth at step {step_global + batch_idx}, Valid Loss: {valid_loss:.4f}')

    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
    return avg_loss


def validate(loader, global_step, show_progress=True):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Validating", disable=not show_progress):
            data = data.to(device)
            out, _, vec_out = model(pos=data.pos_noise, z=data.z, batch=data.batch)
            noise = model.pos_normalizer(data.noise.to(device))
            loss = noise_loss(vec_out, noise)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    writer.add_scalar('Loss/Validation_Epoch', avg_loss, global_step)
    return avg_loss


for epoch in range(1, 11):
    train_loss = train(train_loader, valid_loader, epoch)
    valid_loss = validate(valid_loader, epoch * len(train_loader), epoch)
    print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

writer.close()
