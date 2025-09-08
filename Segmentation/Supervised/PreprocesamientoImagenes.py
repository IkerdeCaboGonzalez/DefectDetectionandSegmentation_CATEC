import os
from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import math

def calcular_frecuencias_pixeles(data_loader):
    """
    Calcula la frecuencia de píxeles de defectos (clase 1) vs no defectos (clase 0)
    """
    total_pixels = 0
    defect_pixels = 0
    
    for batch_data in tqdm(data_loader):
        labels = batch_data["label"] 
        labels_np = labels.squeeze().numpy()  
        batch_total = labels_np.size
        total_pixels += batch_total
        
       
        batch_defects = np.sum(labels_np == 1)
        defect_pixels += batch_defects
    
   
    no_defect_pixels = total_pixels - defect_pixels
    
    freq_defect = defect_pixels / total_pixels
    freq_no_defect = no_defect_pixels / total_pixels
    
    return freq_defect, freq_no_defect

def procesar_imagenes(ruta, image_size, margen):
    extensiones_validas = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    if not ruta.lower().endswith(extensiones_validas):
        raise ValueError(f"El archivo {ruta} no tiene una extensión válida.")

    # Abrir la imagen
    imagen = Image.open(ruta)
    imagen = imagen.resize(image_size, Image.Resampling.LANCZOS)  # Redimensionar la imagen
    imagen_np = np.array(imagen, dtype=np.float32)

    # Normalizar la imagen 
    imagen_np = (imagen_np - np.min(imagen_np)) / (np.max(imagen_np) - np.min(imagen_np))

    # Agregar margen (eliminar marcadores de tomografía si es necesario)
    if margen > 0:
        imagen_np = np.pad(imagen_np, ((margen, margen), (margen, margen)), mode='constant', constant_values=0)

    return imagen_np

def cargar_masks(ruta_masks, image_size):
    extensiones_validas = (".png", ".tif", ".tiff")
    if not ruta_masks.lower().endswith(extensiones_validas):
        raise ValueError(f"El archivo {ruta_masks} no tiene una extensión válida.")

    # Abrir la máscara
    mask = Image.open(ruta_masks)
    mask = mask.resize(image_size, Image.NEAREST)  # Redimensionar la máscara
    mask_np = np.array(mask, dtype=np.int32)

    # Binarizar la máscara 
    mask_np = (mask_np > 0).astype(np.int32)

    return mask_np




def pad_to_divisible(vol, div=16):
    """Pads (Z, H, W) or (C, Z, H, W) array to be divisible by `div`"""
    shape = vol.shape[-3:]  # Z, H, W
    pad = [(0, (div - s % div) % div) for s in shape]
    return np.pad(vol, [(0, 0)] * (vol.ndim - 3) + pad, mode='constant')

def _labels_to_index(y: torch.Tensor) -> torch.Tensor:
    """
    Convierte distintas formas de label a índices [B, H, W] (o [B, H, W, D] si 3D).
    
    """
    t = y
    # Si hay dimensión Z/D al final:
    if t.ndim == 5:  # [B, C_or_1, H, W, D]
        if t.shape[-1] == 1:
            t = t[..., 0]                 # -> [B, C_or_1, H, W]
        else:
            mid = t.shape[-1] // 2
            t = t[..., mid]               # -> corte central [B, C_or_1, H, W]

    # Ahora 2D: [B, C_or_1, H, W] o [B, H, W]
    if t.ndim == 4 and t.size(1) == 2:     # one-hot -> índices
        t = t.argmax(dim=1)
    elif t.ndim == 4 and t.size(1) == 1:   # [B,1,H,W] -> [B,H,W]
        t = t[:, 0]
    elif t.ndim == 3:
        pass
    else:
        raise ValueError(f"Forma de label no soportada: {tuple(y.shape)} -> {tuple(t.shape)}")

    return t.long()

def print_stats(loader, name="LOADER", n_batches=2):
    print(f"\n=== {name} ===")
    tot_pixels = 0
    tot_fg = 0
    for bi, batch in enumerate(loader):
        if bi >= n_batches: break
        y = batch["label"]
        y_idx = _labels_to_index(y)              
        if y_idx.ndim > 3:
            y_idx = y_idx.view(y_idx.size(0), -1)

       
        ymin = int(y.min().item())
        ymax = int(y.max().item())
        uniq = torch.unique(y).detach().cpu().tolist()
        total = y_idx.numel()
        fg = int(y_idx.sum().item())
        bg = total - fg
        tot_pixels += total
        tot_fg += fg
        print(f"Batch {bi}: label.shape={tuple(y.shape)}, dtype={y.dtype}, min={ymin}, max={ymax}, únicos={uniq}")
        print(f" -> total={total:,} | bg={bg:,} | fg={fg:,} | fg%={100*fg/total:.3f}%")
    if tot_pixels > 0:
        print(f"TOTAL ({name}): fg%={100*tot_fg/tot_pixels:.3f}%  (fg={tot_fg:,} / {tot_pixels:,})")

def show_masks(loader, name="LOADER", n=12, cols=4):
    """
    Muestra un grid con n máscaras (tal cual están cargadas) y su fg%.
    """
    rows = math.ceil(n / cols)
    i_shown = 0
    fig = plt.figure(figsize=(3.2*cols, 3.2*rows))
    for batch in loader:
        y = batch["label"]
        y_idx = _labels_to_index(y).cpu().numpy()  # [B,H,W]
        B = y_idx.shape[0]
        for i in range(B):
            if i_shown >= n: break
            m = y_idx[i]
            total = m.size
            fg = int(m.sum())
            pct = 100.0 * fg / total
            ax = plt.subplot(rows, cols, i_shown + 1)
            ax.imshow(m, vmin=0, vmax=1, cmap="gray")
            ax.set_title(f"{name} #{i_shown} | fg%={pct:.3f}%")
            ax.axis("off")
            i_shown += 1
        if i_shown >= n:
            break
    plt.tight_layout()
    plt.show()