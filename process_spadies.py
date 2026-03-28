"""
Script para procesar datos reales del SPADIES
Alerta Estudiantil Colombia
"""

import pandas as pd
from pathlib import Path
import json


def procesar_datos_spadies(excel_path: str, output_dir: str = 'data'):
    """
    Procesa el archivo Excel del SPADIES y genera archivos JSON
    para uso en la aplicación.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("="*60)
    print("PROCESAMIENTO DE DATOS SPADIES")
    print("="*60)
    
    xlsx = pd.ExcelFile(excel_path)
    print(f"\nArchivo cargado: {excel_path}")
    print(f"Hojas encontradas: {len(xlsx.sheet_names)}")
    
    # 1. Procesar TDA (Tasa de Deserción Anual)
    print("\n[1/4] Procesando TDA...")
    df_tda = pd.read_excel(xlsx, sheet_name='TDA IES Padre Total', header=12)
    df_tda.columns = ['codigo', 'nombre', '2019', '2020', '2021', '2022', '2023'] + list(df_tda.columns[7:])
    df_tda_clean = df_tda[['codigo', 'nombre', '2019', '2020', '2021', '2022', '2023']].dropna(subset=['codigo', '2023'])
    # Filtrar filas que no son datos
    df_tda_clean = df_tda_clean[pd.to_numeric(df_tda_clean['codigo'], errors='coerce').notna()]
    df_tda_clean['codigo'] = df_tda_clean['codigo'].astype(int)
    
    # Convertir a porcentaje
    for year in ['2019', '2020', '2021', '2022', '2023']:
        df_tda_clean[year] = (df_tda_clean[year] * 100).round(2)
    
    tda_json = df_tda_clean.to_dict(orient='records')
    with open(output_path / 'tda_ies.json', 'w', encoding='utf-8') as f:
        json.dump(tda_json, f, ensure_ascii=False, indent=2)
    print(f"  -> {len(tda_json)} IES procesadas")
    
    # 2. Procesar TDCA por nivel
    print("\n[2/4] Procesando TDCA por nivel...")
    df_tdca = pd.read_excel(xlsx, sheet_name='TDCA IES Padre Nivel Formacion', header=12)
    df_tdca.columns = ['codigo', 'nombre', 'nivel', '2019', '2020', '2021', '2022', '2023'] + list(df_tdca.columns[8:])
    # Filtrar filas que no son datos (encabezados duplicados, etc.)
    df_tdca_clean = df_tdca[['codigo', 'nombre', 'nivel', '2023']].dropna(subset=['codigo', '2023'])
    df_tdca_clean = df_tdca_clean[pd.to_numeric(df_tdca_clean['codigo'], errors='coerce').notna()]
    df_tdca_clean['codigo'] = df_tdca_clean['codigo'].astype(int)
    df_tdca_clean['2023'] = (df_tdca_clean['2023'] * 100).round(2)
    
    tdca_json = df_tdca_clean.to_dict(orient='records')
    with open(output_path / 'tdca_nivel.json', 'w', encoding='utf-8') as f:
        json.dump(tdca_json, f, ensure_ascii=False, indent=2)
    print(f"  -> {len(tdca_json)} registros procesados")
    
    # 3. Procesar TGA por nivel
    print("\n[3/4] Procesando TGA por nivel...")
    df_tga = pd.read_excel(xlsx, sheet_name='TGA IES Padre Nivel Formacion', header=12)
    df_tga.columns = ['codigo', 'nombre', 'nivel', '2019', '2020', '2021', '2022', '2023'] + list(df_tga.columns[8:])
    df_tga_clean = df_tga[['codigo', 'nombre', 'nivel', '2023']].dropna(subset=['codigo', '2023'])
    df_tga_clean = df_tga_clean[pd.to_numeric(df_tga_clean['codigo'], errors='coerce').notna()]
    df_tga_clean['codigo'] = df_tga_clean['codigo'].astype(int)
    df_tga_clean['2023'] = (df_tga_clean['2023'] * 100).round(2)
    
    tga_json = df_tga_clean.to_dict(orient='records')
    with open(output_path / 'tga_nivel.json', 'w', encoding='utf-8') as f:
        json.dump(tga_json, f, ensure_ascii=False, indent=2)
    print(f"  -> {len(tga_json)} registros procesados")
    
    # 4. Generar estadísticas nacionales
    print("\n[4/4] Generando estadísticas nacionales...")
    
    stats = {
        "anio_datos": 2023,
        "fuente": "SPADIES 3.0 - Ministerio de Educación Nacional",
        "total_ies": len(df_tda_clean),
        "tda": {
            "promedio": round(df_tda_clean['2023'].mean(), 2),
            "mediana": round(df_tda_clean['2023'].median(), 2),
            "minimo": round(df_tda_clean['2023'].min(), 2),
            "maximo": round(df_tda_clean['2023'].max(), 2),
            "desviacion": round(df_tda_clean['2023'].std(), 2)
        },
        "tdca_por_nivel": {
            "TyT": round(df_tdca_clean[df_tdca_clean['nivel'] == 'TyT']['2023'].mean(), 2),
            "Universitario": round(df_tdca_clean[df_tdca_clean['nivel'] == 'Universitario']['2023'].mean(), 2)
        },
        "tga_por_nivel": {
            "TyT": round(df_tga_clean[df_tga_clean['nivel'] == 'TyT']['2023'].mean(), 2),
            "Universitario": round(df_tga_clean[df_tga_clean['nivel'] == 'Universitario']['2023'].mean(), 2)
        },
        "top_5_menor_desercion": df_tda_clean.nsmallest(5, '2023')[['codigo', 'nombre', '2023']].to_dict(orient='records'),
        "top_5_mayor_desercion": df_tda_clean.nlargest(5, '2023')[['codigo', 'nombre', '2023']].to_dict(orient='records')
    }
    
    with open(output_path / 'estadisticas_nacionales.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("RESUMEN DE ESTADÍSTICAS NACIONALES 2023")
    print("="*60)
    print(f"Total IES: {stats['total_ies']}")
    print(f"\nTasa de Deserción Anual (TDA):")
    print(f"  Promedio: {stats['tda']['promedio']}%")
    print(f"  Mediana:  {stats['tda']['mediana']}%")
    print(f"  Rango:    {stats['tda']['minimo']}% - {stats['tda']['maximo']}%")
    print(f"\nDeserción Acumulada (TDCA):")
    print(f"  TyT:          {stats['tdca_por_nivel']['TyT']}%")
    print(f"  Universitario: {stats['tdca_por_nivel']['Universitario']}%")
    print(f"\nGraduación Acumulada (TGA):")
    print(f"  TyT:          {stats['tga_por_nivel']['TyT']}%")
    print(f"  Universitario: {stats['tga_por_nivel']['Universitario']}%")
    
    print("\n" + "="*60)
    print(f"Archivos generados en: {output_path.absolute()}")
    print("="*60)
    
    return stats


if __name__ == "__main__":
    import sys
    
    # Buscar archivo Excel
    possible_paths = [
        '../DESERCION.xlsx',
        '../../DESERCION.xlsx',
        Path.home() / 'Downloads' / 'MACHINE_LEARNING' / 'DESERCION.xlsx'
    ]
    
    excel_path = None
    for path in possible_paths:
        if Path(path).exists():
            excel_path = str(path)
            break
    
    if excel_path:
        procesar_datos_spadies(excel_path)
    else:
        print("Error: No se encontró el archivo DESERCION.xlsx")
        print("Rutas buscadas:")
        for p in possible_paths:
            print(f"  - {p}")
