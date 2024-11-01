import os
import torch
import matplotlib.pyplot as plt

from option import opt

checkpoint_dir = 'F:/HyperSSR/SFCSR_Modificado/checkpoint/'
out_path = 'F:/HyperSSR/SFCSR_Modificado/out/' + opt.datasetName + '/'

loss_values = []
psnr_values = []

def load_checkpoint(checkpoint_path):
    """Función para cargar un checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        return checkpoint
    return None

def plot_loss_psnr():
    """Función para cargar todos los checkpoints y crear las gráficas de pérdida y PSNR."""
    global loss_values, psnr_values

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if len(checkpoints) == 0:
        print("No se encontraron checkpoints.")
        return

    # Ordenar los checkpoints por el número de época
    checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))

    for checkpoint_file in checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        checkpoint = load_checkpoint(checkpoint_path)
        
        if checkpoint:
            epoch = checkpoint['epoch']
            model_state = checkpoint['model']

            # Asumimos que las pérdidas y PSNR están almacenadas en los checkpoints
            # O si tienes que calcularlos a partir del modelo, necesitarías agregar esa lógica aquí
            loss = checkpoint.get('loss', None)
            psnr = checkpoint.get('psnr', None)

            if loss is not None:
                loss_values.append(loss)
            if psnr is not None:
                psnr_values.append(psnr)

    # Graficar pérdidas y PSNR
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Graficar pérdida (Loss)
    plt.figure()
    plt.plot(range(1, len(loss_values) + 1), loss_values, label='Pérdida (Loss)')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Pérdida por Época')
    plt.legend()
    plt.savefig(out_path + 'loss_plot_all_epochs.png')
    plt.close()

    # Graficar PSNR
    plt.figure()
    plt.plot(range(1, len(psnr_values) + 1), psnr_values, label='PSNR')
    plt.xlabel('Época')
    plt.ylabel('PSNR')
    plt.title('PSNR por Época')
    plt.legend()
    plt.savefig(out_path + 'psnr_plot_all_epochs.png')
    plt.close()

if __name__ == "__main__":
    plot_loss_psnr()
