import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from option import opt
from model import SFCCBAM
from data_utils import TrainsetFromFolder, ValsetFromFolder
from eval import PSNR
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import scipy.io as scio

psnr = []
loss_values = []
psnr_values = []

base_dir = os.path.abspath(os.path.dirname(__file__))  
out_path = os.path.join(base_dir, 'Data', 'result_x4', 'CAVE', '4')
checkpoint_dir = os.path.join(base_dir, 'checkpoint')
os.makedirs(out_path, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

def main():
    if opt.show:
        global writer
        writer = SummaryWriter(log_dir='logs')

    # Establecer el dispositivo como GPU (si está disponible)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo: {device}")

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No se encontró una GPU disponible, verifica tu configuración.")

    torch.manual_seed(opt.seed)
    cudnn.benchmark = True

    # Cargar datasets
    train_set = TrainsetFromFolder(os.path.join(base_dir, 'Data', 'train', opt.datasetName, str(opt.upscale_factor)))
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    val_set = ValsetFromFolder(os.path.join(base_dir, 'Data', 'test', opt.datasetName, str(opt.upscale_factor)))
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    # Crear el modelo y moverlo a la GPU
    model = SFCCBAM(opt).to(device)
    criterion = nn.L1Loss().to(device)

    # Usar múltiples GPUs si están disponibles
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    print('# parameters:', sum(param.numel() for param in model.parameters()))

    # Configurar el optimizador
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)

    # Configurar el scheduler de la tasa de aprendizaje
    scheduler = MultiStepLR(optimizer, milestones=[35, 70, 105, 140, 175], gamma=0.5)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = opt.lr

    # Cargar desde el último checkpoint disponible
    start_epoch = opt.start_epoch
    checkpoint = get_latest_checkpoint(checkpoint_dir)
    
    if checkpoint:
        print(f"=> Cargando checkpoint '{checkpoint}'")
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("=> No se encontró ningún checkpoint, comenzando desde la época 1")

    # Entrenamiento
    for epoch in range(start_epoch, opt.nEpochs + 1):
        scheduler.step()
        print(f"Epoch = {epoch}, lr = {optimizer.param_groups[0]['lr']}")
        train_loss = train(train_loader, optimizer, model, criterion, epoch, device)
        val_psnr = val(val_loader, model, epoch, device)
        
        loss_values.append(train_loss)
        psnr_values.append(val_psnr)

        save_checkpoint(model, epoch, optimizer)
        print(f"Epoch [{epoch}/{opt.nEpochs}] - PSNR: {val_psnr:.3f}")

        if epoch % 10 == 0:
            save_plots(epoch)

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if len(checkpoints) == 0:
        return None
    checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, checkpoints[-1])

def train(train_loader, optimizer, model, criterion, epoch, device):
    model.train()
    total_loss = 0

    with tqdm(total=len(train_loader), desc=f"Entrenando Época {epoch}/{opt.nEpochs}", unit="batch", leave=False) as pbar:
        for iteration, batch in enumerate(train_loader, 1):
            input, label = batch[0].to(device), batch[1].to(device)

            localFeats = []
            for i in range(input.shape[1]):
                if i == 0:
                    x = input[:, 0:3, :, :]
                    y = input[:, 0, :, :]
                    new_label = label[:, 0, :, :]

                elif i == input.shape[1] - 1:
                    x = input[:, i-2:i+1, :, :]
                    y = input[:, i, :, :]
                    new_label = label[:, i, :, :]
                else:
                    x = input[:, i-1:i+2, :, :]
                    y = input[:, i, :, :]
                    new_label = label[:, i, :, :]

                SR, localFeats = model(x, y, localFeats, i)
                localFeats = localFeats.detach()

                loss = criterion(SR, new_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pbar.update(1)

            if opt.show:
                writer.add_scalar('Train/Loss', loss.item())

    return total_loss / len(train_loader)

def val(val_loader, model, epoch, device):
    model.eval()
    val_psnr = 0

    with tqdm(total=len(val_loader), desc=f"Validando Época {epoch}", unit="batch", leave=False) as pbar:
        for iteration, batch in enumerate(val_loader, 1):
            input, label = batch[0].to(device), batch[1].to(device)
            SR = np.ones((label.shape[1], label.shape[2], label.shape[3])).astype(np.float32)

            localFeats = []
            for i in range(input.shape[1]):
                if i == 0:
                    x = input[:, 0:3, :, :]
                    y = input[:, 0, :, :]
                    new_label = label[:, 0, :, :]

                elif i == input.shape[1] - 1:
                    x = input[:, i-2:i+1, :, :]
                    y = input[:, i, :, :]
                    new_label = label[:, i, :, :]
                else:
                    x = input[:, i-1:i+2, :, :]
                    y = input[:, i, :, :]
                    new_label = label[:, i, :, :]

                output, localFeats = model(x, y, localFeats, i)
                SR[i, :, :] = output.cpu().data[0].numpy()

            val_psnr += PSNR(SR, label.cpu().data[0].numpy())
            pbar.update(1)

    val_psnr = val_psnr / len(val_loader)
    if opt.show:
        writer.add_scalar('Val/PSNR', val_psnr, epoch)

    return val_psnr

def save_plots(epoch):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    plt.figure()
    plt.plot(range(1, len(loss_values) + 1), loss_values, label='Pérdida (Loss)')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Pérdida por Época')
    plt.legend()
    plt.savefig(out_path + f'loss_plot_epoch_{epoch}.png')
    plt.close()

    plt.figure()
    plt.plot(range(1, len(psnr_values) + 1), psnr_values, label='PSNR')
    plt.xlabel('Época')
    plt.ylabel('PSNR')
    plt.title('PSNR por Época')
    plt.legend()
    plt.savefig(out_path + f'psnr_plot_epoch_{epoch}.png')
    plt.close()

def save_checkpoint(model, epoch, optimizer):
    model_out_path = os.path.join(checkpoint_dir, f"model_{opt.upscale_factor}_epoch_{epoch}.pth")
    state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, model_out_path)

if __name__ == "__main__":
    main()
