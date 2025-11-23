import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mne # Necesario para cargar datos de EEG/MEG y manipularlos
import numpy as np
import scipy.stats as stats # Para las pruebas estadísticas

# --- 1. Configuración de Rutas y Datos ---
# Definir las rutas base y de archivos UPDRS
# ¡ATENCIÓN! Asegúrate de que estas rutas sean correctas en tu sistema.
# Esta ruta DEBE apuntar a donde tengas tus archivos de datos de MNE (e.g., .fif)
# Podría ser la misma carpeta principal de donde se derivaron las coherencias, pero busca los datos brutos.
DATA_BASE_PATH = Path(r"C:\Users\usuario\Desktop\PYTHON SCRIPTS\raw_data_epochs") # <--- ¡ADAPTA ESTA RUTA!
UPDRS_OFF_PATH = Path(r"C:\Users\usuario\Desktop\NICOLAS MARTINEZ- INSTITUTO CAJAL (TFM)\DATASET TFM\participants_updrs_off.tsv")
UPDRS_ON_PATH = Path(r"C:\Users\usuario\Desktop\NICOLAS MARTINEZ- INSTITUTO CAJAL (TFM)\DATASET TFM\participants_updrs_on.tsv")

# Rutas para guardar los resultados del análisis de bursts
BURSTS_SAVE_PATH = Path(r"C:\Users\usuario\Desktop\PYTHON SCRIPTS\burst_analysis_results") # <--- Nueva carpeta para bursts
BURSTS_SAVE_PATH.mkdir(parents=True, exist_ok=True)


# Leer tablas UPDRS
try:
    df_updrs_off = pd.read_csv(UPDRS_OFF_PATH, sep='\t')
    df_updrs_on = pd.read_csv(UPDRS_ON_PATH, sep='\t')
    print(f"Archivos UPDRS cargados: '{UPDRS_OFF_PATH.name}' y '{UPDRS_ON_PATH.name}'")
except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo UPDRS. Verifica la ruta: {e}")
    exit() # Sale del script si los archivos UPDRS no se encuentran

# Crear diccionarios de mapeo de sujetos a puntuaciones UPDRS para acceso rápido
updrs_off_map = dict(zip(df_updrs_off['participant_id'], df_updrs_off['SUM']))
updrs_on_map = dict(zip(df_updrs_on['participant_id'], df_updrs_on['SUM']))

# Importa tu función de preprocesamiento de epochs.
# Asegúrate de que 'technical_validation_utils.py' esté en tu PYTHONPATH o en la misma carpeta.
# Si 'preprocess_raw_to_epochs' requiere un objeto 'bids', necesitarás adaptar cómo obtienes esos objetos
# o cómo pasas los nombres de archivo a la función.
try:
    from technical_validation_utils import preprocess_raw_to_epochs
except ImportError:
    print("Error: No se pudo importar 'preprocess_raw_to_epochs' de 'technical_validation_utils'.")
    print("Asegúrate de que el archivo 'technical_validation_utils.py' existe y está accesible.")
    print("Si tu función de preprocesamiento tiene otro nombre o ruta, ajústala.")
    exit()

# --- 2. Función para Detección y Caracterización de Bursts ---
def detect_and_characterize_bursts(epochs: mne.Epochs, lfp_ch_names: list,
                                   fmin: float, fmax: float,
                                   threshold_std: float = 2.0,
                                   min_duration: float = 0.05) -> dict | None:
    """
    Detecta ráfagas (bursts) en los canales LFP de las épocas y calcula su duración media.

    Args:
        epochs (mne.Epochs): Objeto MNE Epochs que contiene los datos.
        lfp_ch_names (list): Lista de nombres de canales LFP a analizar.
        fmin (float): Frecuencia mínima para el filtro de banda.
        fmax (float): Frecuencia máxima para el filtro de banda.
        threshold_std (float): Umbral en desviaciones estándar de la envolvente de la señal.
        min_duration (float): Duración mínima de la ráfaga para ser considerada (en segundos).

    Returns:
        dict | None: Un diccionario con la 'mean_burst_duration' y 'burst_rate_per_sec' si se encuentran bursts,
                     o None si no se encuentran bursts o hay un error.
    """
    if not lfp_ch_names:
        print("Advertencia: No se proporcionaron nombres de canales LFP para el análisis de bursts.")
        return None

    epochs_lfp = epochs.copy().pick_channels(lfp_ch_names)
    if not epochs_lfp.ch_names:
        print("Advertencia: No se encontraron los canales LFP especificados en las épocas.")
        return None

    # Filtrar datos en la banda de frecuencia de interés
    epochs_filtered = epochs_lfp.filter(l_freq=fmin, h_freq=fmax, verbose=False)

    # Obtener la señal de envolvente (envelope)
    # Se recomienda usar Hilbert para una envolvente más suave y para capturar la amplitud de la oscilación.
    envelope_epochs = epochs_filtered.copy().apply_hilbert(envelope=True, verbose=False)
    # También puedes usar abs() para una envolvente más simple:
    # envelope_epochs = epochs_filtered.copy().apply_function(np.abs, verbose=False)

    sfreq = epochs.info['sfreq']
    all_burst_durations = [] # Para almacenar duraciones de ráfagas de todos los canales y épocas
    total_time_s = epochs.get_data().shape[0] * epochs.get_data().shape[2] / sfreq # Tiempo total de datos en segundos

    for epoch_idx in range(len(envelope_epochs)):
        for ch_idx in range(len(lfp_ch_names)):
            ch_data = envelope_epochs.get_data(picks=lfp_ch_names[ch_idx])[epoch_idx, 0, :]

            # Calcular el umbral (por ejemplo, N desviaciones estándar del envolvente)
            # Podrías usar un percentil o un umbral absoluto si tienes una referencia.
            envelope_std = np.std(ch_data)
            # Asegúrate de que envelope_std no sea cero para evitar división por cero
            if envelope_std == 0:
                continue

            threshold = threshold_std * envelope_std

            # Detección de ráfagas
            above_threshold = ch_data > threshold
            burst_start_indices = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1
            burst_end_indices = np.where(np.diff(above_threshold.astype(int)) == -1)[0] + 1

            # Asegurarse de que los inicios y finales coincidan
            if len(burst_start_indices) > len(burst_end_indices):
                burst_start_indices = burst_start_indices[:len(burst_end_indices)]
            elif len(burst_end_indices) > len(burst_start_indices):
                burst_end_indices = burst_end_indices[len(burst_start_indices):]

            for start, end in zip(burst_start_indices, burst_end_indices):
                duration = (end - start) / sfreq
                if duration >= min_duration:
                    all_burst_durations.append(duration)

    if not all_burst_durations:
        # print(f"Info: No se encontraron ráfagas que cumplan los criterios para {epochs.info['subject_info'] if 'subject_info' in epochs.info else 'desconocido'}")
        return None

    mean_burst_duration = np.mean(all_burst_durations)
    # Calcular la tasa de ráfagas (número total de ráfagas / tiempo total de datos)
    burst_rate_per_sec = len(all_burst_durations) / total_time_s

    return {
        'mean_burst_duration': mean_burst_duration,
        'burst_rate_per_sec': burst_rate_per_sec
    }

# --- 3. Recopilación de Datos de Bursts y UPDRS ---
results = [] # Una lista para almacenar todos los resultados

# Verificar si la ruta base de datos existe y es un directorio
if not DATA_BASE_PATH.is_dir():
    print(f"Error: La ruta base de datos '{DATA_BASE_PATH}' no es un directorio válido o no existe.")
    exit()

print(f"\nIniciando procesamiento de archivos de datos para análisis de bursts en: {DATA_BASE_PATH}")

# Esta parte asume cómo están organizados tus archivos de datos brutos o épocas.
# Tu script de coherencia usa un objeto `bids` de `subject_files`.
# Aquí estoy adaptando para un patrón de carpetas sujeto/archivo.fif.
# ¡ES POSIBLE QUE NECESITES ADAPTAR ESTA LÓGICA DE BÚSQUEDA DE ARCHIVOS!
# Por ejemplo, si tus archivos son .fif y están en subcarpetas del sujeto:
# DATA_BASE_PATH / sub-001 / sub-001_ses-PeriOp_task-HoldL_acq-MedOff_run-1_epo.fif

for subject_folder in DATA_BASE_PATH.iterdir():
    if not subject_folder.is_dir():
        continue

    subject_id_match = subject_folder.name.replace('sub-', '') # Asume formato 'sub-XYZ'
    if not subject_id_match:
        print(f"Advertencia: No se pudo extraer subject_id de '{subject_folder.name}'. Saltando.")
        continue
    
    # Busca archivos de epochs o raw dentro de la carpeta del sujeto.
    # ADAPTA ESTE PATRÓN DE BÚSQUEDA si tus archivos tienen un sufijo diferente o extensión.
    data_files = sorted(list(subject_folder.glob('*.fif'))) # Asume archivos .fif

    if not data_files:
        print(f"Info: No se encontraron archivos .fif en '{subject_folder.name}'. Saltando.")
        continue

    for data_file_path in data_files:
        # Aquí necesitas pasar el argumento correcto a tu función preprocess_raw_to_epochs.
        # Si espera un objeto BIDS, tendrás que recrearlo o adaptar la función.
        # Por ahora, asumo que puede tomar la ruta directa al archivo.
        # Si la función requiere un objeto 'bids', necesitarás adaptar esta parte.
        # Ejemplo: bids_obj = mne_bids.BIDSPath(subject=subject_id_match, ...) y pasarlo.
        # Para simplificar aquí, asumo que `preprocess_raw_to_epochs` puede trabajar con la ruta del archivo.
        print(f"Procesando archivo: {data_file_path.name}")
        epochs = preprocess_raw_to_epochs(data_file_path) # <--- Posible punto de adaptación

        if epochs is None:
            print(f"❌ No se pudo preprocesar las épocas para '{data_file_path.name}'. Saltando.")
            continue

        # Obtener nombres de canales LFP (asumo que se identifican como 'eeg' en info['chs'])
        # Ajusta esto si tus LFP tienen otro nombre o tipo (e.g., 'STN_L_0', 'STN_R_1')
        lfp_channels_info = mne.pick_info(epochs.info, mne.pick_types(epochs.info, eeg=True, meg=False))
        lfp_ch_names = lfp_channels_info['ch_names']
        
        if not lfp_ch_names:
            print(f"Advertencia: No se encontraron canales LFP (tipo 'eeg') en '{data_file_path.name}'. Saltando.")
            continue

        # Definir la banda de frecuencia para los bursts (ejemplo: beta)
        # Puedes cambiar a gamma (fmin=30, fmax=100) si lo necesitas.
        fmin_burst = 13
        fmax_burst = 30 # Beta band

        burst_metrics = detect_and_characterize_bursts(epochs, lfp_ch_names,
                                                        fmin=fmin_burst, fmax=fmax_burst,
                                                        threshold_std=2.0, # Umbral de 2 desviaciones estándar
                                                        min_duration=0.05) # Mínimo 50 ms de duración

        if burst_metrics is None:
            # detect_and_characterize_bursts ya habrá impreso un mensaje de error/advertencia
            continue # Saltar este punto de datos si hubo un problema

        # Decide la condición (MedOff/MedOn) y obtener la puntuación UPDRS
        updrs_score = None
        condition = None

        # El nombre del archivo DEBE contener 'MedOff' o 'MedOn' para identificar la condición
        if 'MedOff' in data_file_path.name:
            if subject_id_match in updrs_off_map:
                updrs_score = updrs_off_map[subject_id_match]
                condition = 'MedOff'
            else:
                print(f"⚠️ Advertencia: No se encontró puntuación UPDRS MedOff para el sujeto '{subject_id_match}' (archivo '{data_file_path.name}').")
        elif 'MedOn' in data_file_path.name:
            if subject_id_match in updrs_on_map:
                updrs_score = updrs_on_map[subject_id_match]
                condition = 'MedOn'
            else:
                print(f"⚠️ Advertencia: No se encontró puntuación UPDRS MedOn para el sujeto '{subject_id_match}' (archivo '{data_file_path.name}').")
        else:
            print(f"⚠️ Advertencia: No se pudo determinar la condición (MedOff/MedOn) para el archivo '{data_file_path.name}'.")

        # Si tenemos una puntuación UPDRS y una condición válidas, añadirlas a los resultados
        if updrs_score is not None and condition is not None:
            results.append({
                'subject_id': subject_id_match,
                'mean_burst_duration': burst_metrics['mean_burst_duration'],
                'burst_rate_per_sec': burst_metrics['burst_rate_per_sec'],
                'updrs': updrs_score,
                'condition': condition
            })

# Convertir la lista de resultados a un DataFrame
df_bursts = pd.DataFrame(results)

if df_bursts.empty:
    print("\n❌ No se pudieron procesar datos válidos de bursts. El DataFrame está vacío. Revisa tus archivos, rutas y la lógica de detección.")
    exit()
else:
    print(f"\nDatos de bursts procesados con éxito. Total de registros: {len(df_bursts)}")
    print("Primeras 5 filas del DataFrame de bursts:")
    print(df_bursts.head())

# Guardar los resultados en un archivo CSV para futuras análisis sin recalcular
df_bursts.to_csv(BURSTS_SAVE_PATH / "burst_analysis_results_beta.csv", index=False) # o _gamma.csv
print(f"\nResultados de bursts guardados en: {BURSTS_SAVE_PATH / 'burst_analysis_results_beta.csv'}")


# --- 4. Análisis y Visualización de Datos de Bursts ---

# Puedes elegir analizar 'mean_burst_duration' o 'burst_rate_per_sec'
metric_to_analyze = 'mean_burst_duration' # O 'burst_rate_per_sec'
ylabel_text = f"Duración Media de Ráfagas ({fmin_burst}-{fmax_burst} Hz)" # O "Tasa de Ráfagas por Segundo"

print(f"\n--- Estadísticas Descriptivas para {ylabel_text} ---")
df_off_bursts = df_bursts[df_bursts['condition'] == 'MedOff'].copy()
df_on_bursts = df_bursts[df_bursts['condition'] == 'MedOn'].copy()

if not df_off_bursts.empty:
    print("\nEstadísticas para MedOff:")
    print(df_off_bursts[metric_to_analyze].describe())
else:
    print("\nNo hay datos disponibles para MedOff.")

if not df_on_bursts.empty:
    print("\nEstadísticas para MedOn:")
    print(df_on_bursts[metric_to_analyze].describe())
else:
    print("\nNo hay datos disponibles para MedOn.")

# 4.1. Visualización de Histogramas
plt.figure(figsize=(10, 6))
sns.histplot(data=df_off_bursts, x=metric_to_analyze, bins=20, kde=True, label="MedOff", color="red", alpha=0.6)
sns.histplot(data=df_on_bursts, x=metric_to_analyze, bins=20, kde=True, label="MedOn", color="blue", alpha=0.6)
plt.legend(title="Condición")
plt.xlabel(ylabel_text)
plt.ylabel("Frecuencia")
plt.title(f"Histograma de {ylabel_text} por Condición")
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()

# 4.2. Comparación con Boxplots
plt.figure(figsize=(8, 6))
sns.boxplot(x="condition", y=metric_to_analyze, data=df_bursts, palette={"MedOff": "red", "MedOn": "blue"})
plt.ylabel(ylabel_text)
plt.xlabel("Condición")
plt.title(f"Comparación de {ylabel_text} entre MedOff y MedOn (Boxplot)")
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()

# 4.3. Visualización con Violinplot
plt.figure(figsize=(8, 6))
sns.violinplot(x="condition", y=metric_to_analyze, data=df_bursts, palette={"MedOff": "red", "MedOn": "blue"})
plt.ylabel(ylabel_text)
plt.xlabel("Condición")
plt.title(f"Distribución de {ylabel_text} entre MedOff y MedOn (Violinplot)")
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()

# 4.4. Gráfico de dispersión con regresión lineal y correlación con UPDRS
plt.figure(figsize=(10, 7))

# Gráfico para MedOff
if not df_off_bursts.empty:
    sns.regplot(x=metric_to_analyze, y='updrs', data=df_off_bursts, label="MedOff", color="red", ci=95, scatter_kws={'alpha':0.7})
else:
    print("No hay suficientes datos para graficar la correlación MedOff.")

# Gráfico para MedOn
if not df_on_bursts.empty:
    sns.regplot(x=metric_to_analyze, y='updrs', data=df_on_bursts, label="MedOn", color="blue", ci=95, scatter_kws={'alpha':0.7})
else:
    print("No hay suficientes datos para graficar la correlación MedOn.")

plt.xlabel(ylabel_text)
plt.ylabel("Puntuación Total UPDRS")
plt.title(f"Correlación entre {ylabel_text} y Puntuación UPDRS")
plt.legend(title="Condición")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- 5. Calcular Coeficientes de Correlación de Spearman ---
print(f"\n--- Coeficientes de Correlación de Spearman para {ylabel_text} ---")

if not df_off_bursts.empty and len(df_off_bursts) >= 2: # Se necesitan al menos 2 puntos para la correlación
    corr_off, p_off = stats.spearmanr(df_off_bursts[metric_to_analyze], df_off_bursts['updrs'])
    print(f"Correlación (Spearman) MedOff: r = {corr_off:.3f}, p = {p_off:.3e}")
else:
    print("No se pudo calcular la correlación para MedOff (DataFrame vacío o menos de 2 registros).")

if not df_on_bursts.empty and len(df_on_bursts) >= 2: # Se necesitan al menos 2 puntos para la correlación
    corr_on, p_on = stats.spearmanr(df_on_bursts[metric_to_analyze], df_on_bursts['updrs'])
    print(f"Correlación (Spearman) MedOn: r = {corr_on:.3f}, p = {p_on:.3e}")
else:
    print("No se pudo calcular la correlación para MedOn (DataFrame vacío o menos de 2 registros).")

print("\nAnálisis de bursts completado. ¡Recuerda interpretar estos resultados en el contexto de tu hipótesis!")
print("Si los gráficos están vacíos o hay advertencias, asegúrate de que las rutas, el formato de los archivos y los parámetros de detección de bursts son correctos.")
