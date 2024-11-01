
import os
import numpy as np
import torch
from os import listdir
import torch.nn as nn
from torch.autograd import Variable
from option import opt
from data_utils import is_image_file
from model import SFCCBAM
import scipy.io as scio
from eval import PSNR, SSIM, SAM, SAM_pixeles  
import time
import matplotlib.pyplot as plt

def plot_pixel_values(HR, SR_sfccbam, pixel_position_1, pixel_position_2, image_name, out_path):
    #Función para graficar los valores de los píxeles en las posiciones dadas
    plt.figure(figsize=(12, 6))
    
    # Extraer los valores de los píxeles para las posiciones especificadas
    hr_pixel_10_70 = HR[:, pixel_position_1[0], pixel_position_1[1]]
    sr_sfccbam_pixel_10_70 = SR_sfccbam[:, pixel_position_1[0], pixel_position_1[1]]
    
    hr_pixel_350_320 = HR[:, pixel_position_2[0], pixel_position_2[1]]
    sr_sfccbam_pixel_350_320 = SR_sfccbam[:, pixel_position_2[0], pixel_position_2[1]]

    # Graficar los valores de los píxeles en las posiciones (10,70)
    plt.subplot(1, 2, 1)
    plt.plot(hr_pixel_10_70, label='HR Pixel (10,70)', linestyle='--')
    plt.plot(sr_sfccbam_pixel_10_70, label='SFCCBAM Pixel (10,70)')
    plt.xlabel('Banda')
    plt.ylabel('Valor de Pixel')
    plt.title('Pixel (10,70)')
    plt.legend()
    plt.grid(True)

    # Graficar los valores de los píxeles en las posiciones (350,320)
    plt.subplot(1, 2, 2)
    plt.plot(hr_pixel_350_320, label='HR Pixel (350,320)', linestyle='--')
    plt.plot(sr_sfccbam_pixel_350_320, label='SFCCBAM Pixel (350,320)')
    plt.xlabel('Banda')
    plt.ylabel('Valor de Pixel')
    plt.title('Pixel (350,320)')
    plt.legend()
    plt.grid(True)

    # Guardar la gráfica
    plt.suptitle(f'Comparación de pixeles para {image_name}')
    plt.savefig(os.path.join(out_path, f'pixel_plot_{image_name}.png'))
    plt.close()

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))  

    input_path = os.path.join(base_dir, 'Data', 'test', 'CAVE', '4')
    out_path = os.path.join(base_dir, 'Data', 'result_x4', 'CAVE', '4')
    os.makedirs(out_path, exist_ok=True)

    print("Input path:", input_path)
    print("Output path:", out_path)

    
    PSNRs_sfccbam = []
    SSIMs_sfccbam = []
    SAMs_sfccbam = []
    
    SAMs_pixel_10_70_sfccbam = []
    SAMs_pixel_350_320_sfccbam = []
    
    # Definir las posiciones de los píxeles
    pixel_position_1 = (10, 70)
    pixel_position_2 = (350, 320)
    
    if opt.cuda:
        print(f"=> Usando GPU ID: {opt.gpus}")
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No se encontró una GPU o el ID de la GPU es incorrecto. Ejecuta sin --cuda")
    
    # Cargar el modelo SFCCBAM
    model_sfccbam = SFCCBAM(opt)
    
    if opt.cuda:
        model_sfccbam = nn.DataParallel(model_sfccbam).cuda()
        
    # Cargar el checkpoint del modelo SFCCBAM
    checkpoint_sfccbam = torch.load(opt.model_name)
    model_sfccbam.load_state_dict(checkpoint_sfccbam['model'])
    
    model_sfccbam.eval()  # Poner el modelo en modo evaluación
    
    images_name = [x for x in listdir(input_path) if is_image_file(x)]
    total_time_sfccbam = 0

    for index, image_name in enumerate(images_name):
        # Cargar las imágenes de entrada y HR (alta resolución)
        mat = scio.loadmat(os.path.join(input_path, image_name))
        hyperLR = mat['LR'].astype(np.float32).transpose(2, 0, 1)
        HR = mat['HR'].astype(np.float32).transpose(2, 0, 1)

        # Preparar el input para el modelo
        input_var = Variable(torch.from_numpy(hyperLR).float()).view(1, -1, hyperLR.shape[1], hyperLR.shape[2])
        if opt.cuda:
            input_var = input_var.cuda()

        # Inicialización de SR para SFCCBAM
        SR_sfccbam = np.array(HR).astype(np.float32)
        localFeats = []
        
        # Proceso de SFCCBAM
        start_time_sfccbam = time.time()
        for i in range(input_var.shape[1]):
            if i == 0:
                x = input_var[:, 0:3, :, :]
                y = input_var[:, 0, :, :]
            elif i == input_var.shape[1] - 1:
                x = input_var[:, i-2:i+1, :, :]
                y = input_var[:, i, :, :]
            else:
                x = input_var[:, i-1:i+2, :, :]
                y = input_var[:, i, :, :]
                
            output_sfccbam, localFeats = model_sfccbam(x, y, localFeats, i)
            SR_sfccbam[i, :, :] = output_sfccbam.cpu().data[0].numpy()

        end_time_sfccbam = time.time()
        elapsed_time_sfccbam = end_time_sfccbam - start_time_sfccbam
        total_time_sfccbam += elapsed_time_sfccbam
        print(f"Tiempo de procesamiento para la imagen {index+1} con SFCSR: {elapsed_time_sfccbam:.4f} segundos")

        # Limitar los valores de SR entre 0 y 1
        SR_sfccbam = np.clip(SR_sfccbam, 0, 1)

        # Calcular métricas de evaluación para la imagen completa (SFCCBAM)
        psnr_sfccbam = PSNR(SR_sfccbam, HR)
        ssim_sfccbam = SSIM(SR_sfccbam, HR)
        sam_sfccbam = SAM(SR_sfccbam, HR)

        PSNRs_sfccbam.append(psnr_sfccbam)
        SSIMs_sfccbam.append(ssim_sfccbam)
        SAMs_sfccbam.append(sam_sfccbam)
        
       

        # Graficar los valores de los píxeles para la imagen y guardar la gráfica
        plot_pixel_values(HR, SR_sfccbam, pixel_position_1, pixel_position_2, image_name, out_path)

        # Calcular SAM en las posiciones (10,70) y (350,320) para SFCCBAM
        sam_pixel_10_70_sfccbam = SAM_pixeles(SR_sfccbam[:, pixel_position_1[0], pixel_position_1[1]], HR[:, pixel_position_1[0], pixel_position_1[1]])
        SAMs_pixel_10_70_sfccbam.append(sam_pixel_10_70_sfccbam)

        sam_pixel_350_320_sfccbam = SAM_pixeles(SR_sfccbam[:, pixel_position_2[0], pixel_position_2[1]], HR[:, pixel_position_2[0], pixel_position_2[1]])
        SAMs_pixel_350_320_sfccbam.append(sam_pixel_350_320_sfccbam)

        # Imprimir las métricas para SFCCBAM en esta imagen
        print(f"Imagen {index+1}: PSNR SFCSR: {psnr_sfccbam:.3f}, SSIM SFCSR: {ssim_sfccbam:.4f}, SAM SFCSR: {sam_sfccbam:.3f}, Nombre: {image_name}")
        print(f"SAM (10,70) SFCSR: {sam_pixel_10_70_sfccbam:.3f}, SAM (350,320) SFCSR: {sam_pixel_350_320_sfccbam:.3f}")

         # Guardar el resultado
        SR = SR_sfccbam.transpose(1, 2, 0)
        HR = HR.transpose(1, 2, 0)
        scio.savemat(os.path.join(out_path, image_name), {'HR': HR, 'SR': SR})
        
    # Imprimir los promedios de todas las métricas para SFCCBAM
    print(f"Promedio PSNR SFCSR: {np.mean(PSNRs_sfccbam):.3f}, Promedio SSIM SFCSR: {np.mean(SSIMs_sfccbam):.4f}, Promedio SAM SFCSR: {np.mean(SAMs_sfccbam):.3f}")
    print(f"Promedio SAM (10,70) SFCSR: {np.mean(SAMs_pixel_10_70_sfccbam):.3f}, Promedio SAM (350,320) SFCSR: {np.mean(SAMs_pixel_350_320_sfccbam):.3f}")
    print(f"Tiempo promedio por imagen con SFCSR: {total_time_sfccbam / len(images_name):.4f} segundos")

if __name__ == "__main__":
    main()
