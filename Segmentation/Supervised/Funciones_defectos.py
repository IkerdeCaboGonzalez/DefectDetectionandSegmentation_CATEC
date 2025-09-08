"""
Módulo para análisis de defectos en volúmenes 3D
Calcula volúmenes, diámetros y otras métricas de defectos detectados
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.ndimage import label, generate_binary_structure


def calcular_diametros_defectos(labeled_array, defect_indices, voxel_size_mm=0.0347):
    """
    Calcula el diámetro mayor de cada defecto en 3D
    
    Args:
        labeled_array: Array con defectos etiquetados
        defect_indices: Lista de índices de defectos a analizar
        voxel_size_mm: Tamaño del voxel en mm
    
    Returns:
        Lista de diámetros en mm
    """
    diametros = []
    
    for idx in defect_indices:
        coords = np.where(labeled_array == idx)
        z_coords, y_coords, x_coords = coords
        
        if len(z_coords) < 2:
            diametros.append(0.0)
            continue
        
        points_mm = np.column_stack([
            x_coords * voxel_size_mm,
            y_coords * voxel_size_mm,
            z_coords * voxel_size_mm
        ])
    
        distances = pdist(points_mm)

        diametro_mayor = np.max(distances)
        diametros.append(diametro_mayor)
    
    return diametros


def calcular_diametros_por_plano(labeled_array, defect_indices, voxel_size_mm=0.0347):
    """
    Calcula diámetros promedio en cada plano (XY, XZ, YZ) para cada defecto
    
    Args:
        labeled_array: Array con defectos etiquetados
        defect_indices: Lista de índices de defectos a analizar
        voxel_size_mm: Tamaño del voxel en mm
    
    Returns:
        Diccionario con diámetros por plano para cada defecto
    """
    resultados = {}
    
    for idx in defect_indices:
        coords = np.where(labeled_array == idx)
        z_coords, y_coords, x_coords = coords
        
        if len(z_coords) < 2:
            resultados[idx] = {"XY": 0.0, "XZ": 0.0, "YZ": 0.0, "3D": 0.0}
            continue
        
        # Convertir a mm
        x_mm = x_coords * voxel_size_mm
        y_mm = y_coords * voxel_size_mm
        z_mm = z_coords * voxel_size_mm
        
        # Diámetro en plano XY (proyección en Z)
        if len(np.unique(np.column_stack([x_mm, y_mm]), axis=0)) > 1:
            xy_points = np.column_stack([x_mm, y_mm])
            diam_xy = np.max(pdist(xy_points)) if len(xy_points) > 1 else 0.0
        else:
            diam_xy = 0.0
            
        # Diámetro en plano XZ (proyección en Y)
        if len(np.unique(np.column_stack([x_mm, z_mm]), axis=0)) > 1:
            xz_points = np.column_stack([x_mm, z_mm])
            diam_xz = np.max(pdist(xz_points)) if len(xz_points) > 1 else 0.0
        else:
            diam_xz = 0.0
            
        # Diámetro en plano YZ (proyección en X)
        if len(np.unique(np.column_stack([y_mm, z_mm]), axis=0)) > 1:
            yz_points = np.column_stack([y_mm, z_mm])
            diam_yz = np.max(pdist(yz_points)) if len(yz_points) > 1 else 0.0
        else:
            diam_yz = 0.0
            
        # Diámetro 3D
        points_3d = np.column_stack([x_mm, y_mm, z_mm])
        diam_3d = np.max(pdist(points_3d)) if len(points_3d) > 1 else 0.0
        
        resultados[idx] = {
            "XY": diam_xy,
            "XZ": diam_xz, 
            "YZ": diam_yz,
            "3D": diam_3d
        }
    
    return resultados


def analizar_defectos_completo(pred_np, voxel_size_mm=0.0347, vol_threshold=0.001, voxels_threshold=10):
    """
    Análisis completo de defectos: etiquetado, volumen y diámetros
    
    Args:
        pred_np: Array numpy con predicciones (0=fondo, 1=defecto)
        voxel_size_mm: Tamaño del voxel en mm
        vol_threshold: Umbral mínimo de volumen en mm³
        voxels_threshold: Umbral mínimo de vóxeles
    
    Returns:
        Diccionario con análisis completo de defectos
    """
    # Crear máscara binaria
    binary_pred = (pred_np == 1).astype(np.uint8)
    
    # Etiquetado de componentes conectados
    structure = generate_binary_structure(3, 2)
    labeled_array, num_defects = label(binary_pred, structure=structure)
    
    # Calcular volumen de cada defecto
    voxel_volume_mm3 = voxel_size_mm ** 3
    defect_list = []
    defect_indices = []
    
    for i in range(1, num_defects + 1):
        mask = (labeled_array == i)
        voxels = np.sum(mask)
        volumen = voxels * voxel_volume_mm3
        
        # Slices donde aparece el defecto
        slices = np.unique(np.where(mask)[0])
        
        # Solo incluir defectos grandes
        if volumen > vol_threshold and voxels > voxels_threshold:
            defect_list.append({
                "defecto_id": i,
                "volumen": float(round(volumen, 6)),
                "voxels": int(voxels),
                "slices": [int(s) for s in slices]
            })
            defect_indices.append(i)
    
    # Calcular diámetros
    diametros_3d = calcular_diametros_defectos(labeled_array, defect_indices, voxel_size_mm)
    diametros_por_plano = calcular_diametros_por_plano(labeled_array, defect_indices, voxel_size_mm)
    
    # Agregar diámetros a la lista de defectos
    for i, defecto in enumerate(defect_list):
        if i < len(diametros_3d):
            defecto["diametro_3d"] = round(diametros_3d[i], 4)
            idx = defecto["defecto_id"]
            if idx in diametros_por_plano:
                defecto["diametro_xy"] = round(diametros_por_plano[idx]["XY"], 4)
                defecto["diametro_xz"] = round(diametros_por_plano[idx]["XZ"], 4)
                defecto["diametro_yz"] = round(diametros_por_plano[idx]["YZ"], 4)
    
    return {
        "labeled_array": labeled_array,
        "num_defects_total": num_defects,
        "num_defects_significant": len(defect_list),
        "defect_list": defect_list,
        "defect_indices": defect_indices,
        "voxel_size_mm": voxel_size_mm
    }


def imprimir_resumen_defectos(resultado_analisis):
    """
    Imprime un resumen detallado del análisis de defectos
    
    Args:
        resultado_analisis: Resultado de analizar_defectos_completo()
    """
    print(f"\n=== ANÁLISIS COMPLETO DE DEFECTOS ===")
    print(f"Total defectos detectados: {resultado_analisis['num_defects_total']}")
    print(f"Defectos significativos: {resultado_analisis['num_defects_significant']}")
    print(f"Tamaño voxel: {resultado_analisis['voxel_size_mm']} mm")
    
    for defecto in resultado_analisis['defect_list']:
        print(f"\n• Defecto {defecto['defecto_id']}:")
        print(f"  - Volumen: {defecto['volumen']:.4f} mm³")
        print(f"  - Vóxeles: {defecto['voxels']}")
        print(f"  - Slices: {len(defecto['slices'])} ({min(defecto['slices'])}-{max(defecto['slices'])})")
        
        if 'diametro_3d' in defecto:
            print(f"  - Diámetro 3D: {defecto['diametro_3d']:.3f} mm")
            print(f"  - Diámetro plano XY: {defecto['diametro_xy']:.3f} mm")
            print(f"  - Diámetro plano XZ: {defecto['diametro_xz']:.3f} mm")
            print(f"  - Diámetro plano YZ: {defecto['diametro_yz']:.3f} mm")