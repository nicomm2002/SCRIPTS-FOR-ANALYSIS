import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import re # Para expresiones regulares
from scipy import stats # Para pruebas estadísticas y correlaciones
import warnings
import sys
import mne_connectivity # Necesario para leer archivos _coh

# --- 1. Configuración de Rutas ---
# Ruta base donde se encuentran tus datos raw organizados en BIDS (también para UPDRS)
BIDS_ROOT_RAW_DATA = Path(r"C:\Users\usuario\Desktop\NICOLAS MARTINEZ- INSTITUTO CAJAL (TFM)\DATASET TFM")

# Ruta donde se encuentran los archivos de coherencia (subcarpetas de sujeto dentro de esta)
COHERENCE_DATA_DIR = Path(r"C:\Users\usuario\Desktop\PYTHON SCRIPTS\coherence")

# Archivos de UPDRS
UPDRS_OFF_FILE = BIDS_ROOT_RAW_DATA / "participants_updrs_off.tsv"
UPDRS_ON_FILE = BIDS_ROOT_RAW_DATA / "participants_updrs_on.tsv"

# *** MODIFICACIÓN: La ruta de salida será la especificada por el usuario ***
# Esta variable se actualizará en main_analysis
OUTPUT_PLOTS_DIR = Path("") 

# Bandas de frecuencia que estamos analizando (SOLO BETA)
BANDS_FREQS = {
    'beta': (13, 30) 
}
BANDS = list(BANDS_FREQS.keys()) 

# --- 2. Cargar Datos de UPDRS ---
def load_updrs_data(file_path, condition):
    """Carga los datos de UPDRS desde un archivo TSV."""
    if not file_path.exists():
        print(f"Error: Archivo UPDRS no encontrado en {file_path}")
        return pd.DataFrame()
    
    df_updrs = pd.read_csv(file_path, sep='\t')
    
    try:
        df_updrs = df_updrs.rename(columns={'participant_id': 'subject', 'SUM': 'updrs_score'})
    except KeyError as e:
        print(f"\nERROR: Una de las columnas esperadas ('participant_id' o 'SUM') no se encontró al renombrar. Detalles: {e}")
        print("Por favor, verifica los nombres reales de las columnas en tu archivo TSV.")
        print(f"Las columnas disponibles son: {df_updrs.columns.tolist()}")
        sys.exit(1)
        
    df_updrs['condition'] = condition
    df_updrs['subject'] = df_updrs['subject'].str.replace('sub-', '')
    print(f"Cargados {len(df_updrs)} registros UPDRS para la condición '{condition}'.")
    return df_updrs[['subject', 'updrs_score', 'condition']]

# --- 3. Cargar Datos de Coherencia por Tarea ---
def load_coherence_data(data_dir, bands_freqs, task_filter=None): 
    """
    Carga y calcula la coherencia media por banda desde archivos _coh y _ind.pkl,
    filtrando opcionalmente por una tarea específica.
    """
    coherence_data = []
    
    print(f"\nDEBUG: Buscando datos de coherencia en subcarpetas de: {data_dir}")
    
    # Nuevo patrón para extraer la tarea también
    base_name_pattern = re.compile(r'sub-(?P<subject>[a-zA-Z0-9]+)_.*?task-(?P<task_name>\w+)_acq-(?P<condition>MedOff|MedOn)_run-\d+$')

    all_subject_folders = [f for f in data_dir.iterdir() if f.is_dir()]
    print(f"DEBUG: Encontradas {len(all_subject_folders)} carpetas de sujeto.")

    if not all_subject_folders:
        print(f"DEBUG: ¡Advertencia! No se encontraron subcarpetas de sujeto en {data_dir}.")
        
    for subject_folder in all_subject_folders:
        subject_id_raw = subject_folder.name 
        subject_id = subject_id_raw.replace('sub-', '')

        coh_files = sorted([f for f in subject_folder.iterdir() if f.name.endswith('_coh') and f.is_file()])
        pkl_files = sorted([f for f in subject_folder.iterdir() if f.name.endswith('_ind.pkl') and f.is_file()])

        coh_files_dict = {f.name.replace('_coh', ''): f for f in coh_files}
        pkl_files_dict = {f.name.replace('_ind.pkl', ''): f for f in pkl_files}

        for base_name in coh_files_dict.keys():
            coh_path = coh_files_dict[base_name]
            pkl_path = pkl_files_dict.get(base_name) 

            if pkl_path is None:
                print(f"DEBUG: ❌ No se encontró archivo _ind.pkl para el base_name '{base_name}' en {subject_folder.name}. Saltando.")
                continue

            print(f"\nDEBUG: Procesando par de archivos para base_name: {base_name}")
            
            match = base_name_pattern.search(base_name)
            if not match:
                print(f"DEBUG: Base_name '{base_name}' NO COINCIDE con el patrón de extracción de sujeto/condición/tarea. Saltando.")
                continue
            
            extracted_condition = match.group('condition')
            extracted_task = match.group('task_name') 

            # Filtrar por tarea si task_filter no es None
            if task_filter is not None and extracted_task != task_filter:
                print(f"DEBUG: Tarea '{extracted_task}' no coincide con el filtro '{task_filter}'. Saltando '{base_name}'.")
                continue

            try:
                coh = mne_connectivity.read_connectivity(coh_path)
                
                with open(pkl_path, 'rb') as f:
                    coh_info = pickle.load(f)

                coh_data = coh.get_data() 
                freqs = np.array(coh.freqs)

                if coh_data.ndim != 2:
                    print(f"DEBUG: Advertencia: Formato de coh_data inesperado ({coh_data.shape}) para {coh_path.name}. Saltando.")
                    continue
                
                if 'n_lfp_sensors' not in coh_info or 'n_meg_sensors' not in coh_info:
                    print(f"DEBUG: Advertencia: _ind.pkl para '{base_name}' no contiene 'n_lfp_sensors' o 'n_meg_sensors'. Saltando.")
                    continue

                n_lfp = coh_info['n_lfp_sensors']
                n_meg = coh_info['n_meg_sensors']

                if coh_data.shape[0] != (n_lfp * n_meg):
                     print(f"DEBUG: Advertencia: coh_data.shape[0] ({coh_data.shape[0]}) no coincide con n_lfp*n_meg ({n_lfp*n_meg}) para {coh_path.name}. Saltando.")
                     continue
                
                coh_data_reshaped = coh_data.reshape(n_lfp, n_meg, len(freqs))

                for band_name, (fmin, fmax) in bands_freqs.items():
                    band_idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
                    
                    if len(band_idx) == 0:
                        print(f"DEBUG: Advertencia: No se encontraron frecuencias en la banda {band_name} ({fmin}-{fmax} Hz) para {coh_path.name}. Coherencia NaN.")
                        mean_coh_band = np.nan 
                    else:
                        coh_band_per_connection = np.mean(coh_data_reshaped[:, :, band_idx], axis=2)
                        mean_coh_band = np.mean(coh_band_per_connection) 
                    
                    coherence_data.append({
                        'subject': subject_id,
                        'condition': extracted_condition,
                        'band': band_name,
                        'task': extracted_task, 
                        'coherence_mean': mean_coh_band
                    })
                    print(f"DEBUG: Coherencia media {band_name} para {base_name}: {mean_coh_band:.4f}")

            except Exception as e:
                print(f"DEBUG: ¡ERROR! Error procesando el par de archivos para base_name '{base_name}': {e}. Saltando.")
                continue

    df_coh = pd.DataFrame(coherence_data)
    print(f"\nCargados {len(df_coh)} registros de coherencia (filtrado por tarea: {task_filter if task_filter else 'Ninguno'}).")
    return df_coh

# --- 4. Cargar y Combinar Todos los Datos (filtrando por Tarea) ---
def prepare_merged_data(task_name=None): 
    df_updrs_off = load_updrs_data(UPDRS_OFF_FILE, 'MedOff')
    df_updrs_on = load_updrs_data(UPDRS_ON_FILE, 'MedOn')
    
    if df_updrs_off.empty or df_updrs_on.empty:
        print("No se pudieron cargar los datos de UPDRS. Terminando.")
        sys.exit(1)

    df_updrs = pd.concat([df_updrs_off, df_updrs_on], ignore_index=True)
    
    # Pasa task_name a load_coherence_data
    df_coh = load_coherence_data(COHERENCE_DATA_DIR, BANDS_FREQS, task_filter=task_name) 
    
    if df_coh.empty:
        print(f"No se pudieron cargar los datos de coherencia para la tarea '{task_name}'. Terminando.")
        sys.exit(1)

    # Agrupar por sujeto, condición y banda, y promediar si hay múltiples registros (ej. por diferentes runs de la misma tarea)
    df_coh_agg = df_coh.groupby(['subject', 'condition', 'band'])['coherence_mean'].mean().reset_index()

    df_coh_pivot = df_coh_agg.pivot_table(index=['subject', 'condition'], 
                                            columns='band', 
                                            values='coherence_mean',
                                            observed=False).reset_index() 
    df_coh_pivot.columns.name = None 
    
    merged_data = pd.merge(df_updrs, df_coh_pivot, on=['subject', 'condition'], how='inner')
    
    delta_data = []
    
    for subject in merged_data['subject'].unique():
        data_sub = merged_data[merged_data['subject'] == subject]
        
        updrs_off_val = data_sub[(data_sub['condition'] == 'MedOff')]['updrs_score'].values
        updrs_on_val = data_sub[(data_sub['condition'] == 'MedOn')]['updrs_score'].values
        
        coh_vals = {
            band: {
                'MedOff': data_sub[(data_sub['condition'] == 'MedOff')][band].values,
                'MedOn': data_sub[(data_sub['condition'] == 'MedOn')][band].values
            } for band in BANDS_FREQS.keys() 
        }
        
        all_present = True
        if not (len(updrs_off_val) == 1 and len(updrs_on_val) == 1):
            all_present = False
        else:
            for band_name in BANDS_FREQS.keys():
                if not (len(coh_vals[band_name]['MedOff']) == 1 and len(coh_vals[band_name]['MedOn']) == 1):
                    all_present = False
                    break
        
        if all_present:
            delta_updrs = updrs_on_val[0] - updrs_off_val[0]
            
            delta_entry = {
                'subject': subject,
                'delta_updrs': delta_updrs,
                'updrs_off': updrs_off_val[0],
                'updrs_on': updrs_on_val[0],
            }
            
            for band_name in BANDS_FREQS.keys():
                delta_coh = coh_vals[band_name]['MedOn'][0] - coh_vals[band_name]['MedOff'][0]
                delta_entry[f'delta_{band_name}_coh'] = delta_coh
                delta_entry[f'{band_name}_off'] = coh_vals[band_name]['MedOff'][0]
                delta_entry[f'{band_name}_on'] = coh_vals[band_name]['MedOn'][0]
            
            delta_data.append(delta_entry)
        else:
            print(f"Advertencia: Datos incompletos (UPDRS y/o Coherencia) para el sujeto {subject} en ambas condiciones. Se saltará el cálculo de delta.")

    df_deltas = pd.DataFrame(delta_data)
    
    print(f"Total de sujetos con datos completos para deltas: {len(df_deltas)}")
    
    return merged_data, df_deltas

# --- 5. Funciones de Análisis Estadístico (sin cambios) ---
def perform_paired_ttest(data_df, value_col, cond1, cond2, cond_col='condition'):
    """Realiza una prueba t de Student pareada (o Wilcoxon si no es normal)."""
    pivot_df = data_df.pivot_table(index='subject', columns=cond_col, values=value_col, observed=False)
    val1 = pivot_df[cond1].dropna()
    val2 = pivot_df[cond2].dropna()
    
    if len(val1) > 1 and len(val1) == len(val2):
        t_stat, p_val = stats.ttest_rel(val1, val2, nan_policy='omit')
        print(f"  T-test pareado para {value_col} ({cond1} vs {cond2}): t={t_stat:.3f}, p={p_val:.3f}")
        if p_val < 0.05:
            print(f"  -> Diferencia significativa encontrada (p < 0.05).")
        return t_stat, p_val
    else:
        print(f"  No hay suficientes pares de datos para {value_col} ({cond1} vs {cond2}) para realizar un t-test pareado.")
        return np.nan, np.nan

def perform_correlation(data_df, x_col, y_col):
    """Realiza una correlación de Pearson."""
    temp_df = data_df[[x_col, y_col]].dropna()
    if len(temp_df) < 2:
        print(f"  No hay suficientes puntos para correlacionar {x_col} y {y_col}.")
        return np.nan, np.nan
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        r, p = stats.pearsonr(temp_df[x_col], temp_df[y_col])
    print(f"  Correlación entre {x_col} y {y_col}: R={r:.3f}, P={p:.3f}")
    if p < 0.05:
        print(f"  -> Correlación significativa encontrada (p < 0.05).")
    return r, p

# --- 6. Funciones de Graficación ---
def save_plot(fig, filename, show=True): 
    """Guarda una figura en el directorio de salida y opcionalmente la muestra."""
    filepath = OUTPUT_PLOTS_DIR / filename
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Gráfico guardado: {filepath}")
    if show:
        plt.show() 
    plt.close(fig) 

def plot_updrs_violin_boxplot(data_df):
    """Genera violinplot y boxplot para UPDRS (MedOff vs MedOn)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    df_plot = data_df.copy()
    df_plot['condition'] = pd.Categorical(df_plot['condition'], categories=['MedOff', 'MedOn'], ordered=True)
    df_plot = df_plot.sort_values('condition')

    sns.violinplot(data=df_plot, x='condition', y='updrs_score', hue='condition', 
                   palette={'MedOff': 'grey', 'MedOn': 'white'}, inner='quartile', ax=ax, 
                   linewidth=1, edgecolor='black', alpha=0.7)
    
    sns.boxplot(data=df_plot, x='condition', y='updrs_score', hue='condition', 
                palette={'MedOff': 'dimgray', 'MedOn': 'lightgray'}, width=0.2, ax=ax, 
                boxprops=dict(alpha=0.8, edgecolor='black'), 
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(color='black'), 
                capprops=dict(color='black'))
    
    sns.stripplot(data=df_plot, x='condition', y='updrs_score', color='white', 
                  edgecolor='gray', linewidth=0.5, size=7, jitter=True, ax=ax, alpha=0.9)

    ax.set_title('Distribución de Puntuación UPDRS entre MedOff y MedOn')
    ax.set_xlabel('Condición')
    ax.set_ylabel('UPDRS Score')
    ax.grid(axis='y', linestyle='-', alpha=0.7)
    
    temp_updrs_paired = df_plot.pivot_table(index='subject', columns='condition', values='updrs_score', observed=False).dropna()
    
    if len(temp_updrs_paired) > 1:
        t_stat, p_val = stats.ttest_rel(temp_updrs_paired['MedOff'], temp_updrs_paired['MedOn'])
        
        if p_val < 0.05:
            x1, x2 = 0, 1
            y, h, col = df_plot['updrs_score'].max() * 1.05, df_plot['updrs_score'].max() * 0.02, 'k'
            ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            asterisks = ''
            if p_val < 0.001: asterisks = '***'
            elif p_val < 0.01: asterisks = '**'
            elif p_val < 0.05: asterisks = '*'
            ax.text((x1+x2)*.5, y+h, f'{asterisks}', ha='center', va='bottom', color=col, fontsize=18)
            ax.text((x1+x2)*.5, y+h*0.5, f'p={p_val:.3f}', ha='center', va='top', color=col, fontsize=10)
        else:
            ax.text(0.5, 1.05, f'p={p_val:.3f} (ns)', ha='center', va='bottom', transform=ax.transAxes, fontsize=10)
    else:
        print("No hay suficientes pares de datos para UPDRS para añadir la significancia en el gráfico.")

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 2: 
        ax.legend(handles[:2], labels[:2], title='Condición')
    else: 
        if ax.legend_ is not None:
            ax.legend_.remove() 

    save_plot(fig, 'UPDRS_MedOff_MedOn_ViolinBoxplot.png', show=True) 

def plot_coherence_distributions(data_df, band, task_name=""): 
    """Genera histogramas, boxplots y violinplots para la coherencia por banda."""
    df_band = data_df.copy()
    df_band['condition'] = pd.Categorical(df_band['condition'], categories=['MedOff', 'MedOn'], ordered=True)
    
    # Nombre de la tarea para el título del gráfico
    task_title_suffix = f" ({task_name})" if task_name else ""

    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_band, x=band, hue='condition', kde=True, 
                 palette={'MedOff': 'red', 'MedOn': 'blue'}, ax=ax_hist, 
                 stat='count', linewidth=1, edgecolor='black', alpha=0.6)
    ax_hist.set_title(f'Histograma de Coherencia Media {band.capitalize()} por Condición{task_title_suffix}')
    ax_hist.set_xlabel(f'Coherencia Media {band.capitalize()} (LFP-MEG)')
    ax_hist.set_ylabel('Frecuencia')
    ax_hist.grid(axis='y', linestyle='-', alpha=0.7)
    save_plot(fig_hist, f'Histograma_Coherencia_{band.capitalize()}_{task_name}.png', show=True) 

    fig_box, ax_box = plt.subplots(figsize=(10, 7))
    sns.boxplot(data=df_band, x='condition', y=band, hue='condition', 
                palette={'MedOff': 'red', 'MedOn': 'blue'}, ax=ax_box,
                boxprops=dict(edgecolor='black'), medianprops=dict(color='black'),
                whiskerprops=dict(color='black'), capprops=dict(color='black'))
    ax_box.set_title(f'Comparación de Coherencia Media {band.capitalize()} entre MedOff y MedOn (Boxplot){task_title_suffix}')
    ax_box.set_xlabel('Condición')
    ax_box.set_ylabel(f'Coherencia Media {band.capitalize()} (LFP-MEG)')
    ax_box.grid(axis='y', linestyle='-', alpha=0.7)
    save_plot(fig_box, f'Boxplot_Coherencia_{band.capitalize()}_{task_name}.png', show=True) 

    fig_violin, ax_violin = plt.subplots(figsize=(10, 7))
    sns.violinplot(data=df_band, x='condition', y=band, hue='condition', 
                   palette={'MedOff': 'red', 'MedOn': 'blue'}, inner='quartile', ax=ax_violin,
                   linewidth=1, edgecolor='black')
    ax_violin.set_title(f'Distribución de Coherencia Media {band.capitalize()} entre MedOff y MedOn (Violinplot){task_title_suffix}')
    ax_violin.set_xlabel('Condición')
    ax_violin.set_ylabel(f'Coherencia Media {band.capitalize()} (LFP-MEG)')
    ax_violin.grid(axis='y', linestyle='-', alpha=0.7)
    save_plot(fig_violin, f'Violinplot_Coherencia_{band.capitalize()}_{task_name}.png', show=True) 

def plot_coherence_updrs_correlation(df_deltas, merged_data, band, task_name=""): 
    """Genera scatter plots de correlación entre coherencia y UPDRS."""
    task_title_suffix = f" ({task_name})" if task_name else ""

    fig_delta, ax_delta = plt.subplots(figsize=(10, 7))
    sns.regplot(data=df_deltas, x=f'delta_{band}_coh', y='delta_updrs', ax=ax_delta, 
                scatter_kws={'alpha':0.6, 'color':'purple'}, line_kws={'color':'purple'})
    
    r_delta, p_delta = perform_correlation(df_deltas, f'delta_{band}_coh', 'delta_updrs')
    ax_delta.set_title(f'Correlación entre Delta Coherencia {band.capitalize()} y Delta UPDRS{task_title_suffix}')
    ax_delta.set_xlabel(f'Delta Coherencia Media {band.capitalize()} (LFP-MEG)')
    ax_delta.set_ylabel('Delta Puntuación Total UPDRS (MedOn - MedOff)')
    ax_delta.text(0.05, 0.95, f'R={r_delta:.3f}, P={p_delta:.3f}', transform=ax_delta.transAxes, fontsize=12, verticalalignment='top')
    ax_delta.grid(True, linestyle='--', alpha=0.7)
    save_plot(fig_delta, f'Correlacion_Delta_Coherencia_{band.capitalize()}_Delta_UPDRS_{task_name}.png', show=True) 

    fig_cond, ax_cond = plt.subplots(figsize=(10, 7))
    
    sns.regplot(data=merged_data[merged_data['condition'] == 'MedOff'], x=band, y='updrs_score', 
                ax=ax_cond, scatter_kws={'color':'red', 'alpha':0.7}, line_kws={'color':'red'}, label='MedOff')
    sns.regplot(data=merged_data[merged_data['condition'] == 'MedOn'], x=band, y='updrs_score', 
                ax=ax_cond, scatter_kws={'color':'blue', 'alpha':0.7}, line_kws={'color':'blue'}, label='MedOn')

    r_off, p_off = perform_correlation(merged_data[merged_data['condition'] == 'MedOff'], band, 'updrs_score')
    r_on, p_on = perform_correlation(merged_data[merged_data['condition'] == 'MedOn'], band, 'updrs_score')

    ax_cond.set_title(f'Correlación entre Coherencia Media {band.capitalize()} y Puntuación UPDRS{task_title_suffix}')
    ax_cond.set_xlabel(f'Coherencia Media {band.capitalize()} (LFP-MEG)')
    ax_cond.set_ylabel('Puntuación Total UPDRS')
    ax_cond.grid(True, linestyle='--', alpha=0.7)
    
    ax_cond.legend(title='Condición', loc='upper right')

    ax_cond.text(0.98, 0.95, f'MedOff: R={r_off:.3f}, P={p_off:.3f}', 
                 transform=ax_cond.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', color='red')
    ax_cond.text(0.98, 0.90, f'MedOn: R={r_on:.3f}, P={p_on:.3f}', 
                 transform=ax_cond.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', color='blue')

    save_plot(fig_cond, f'Correlacion_Coherencia_{band.capitalize()}_UPDRS_Combinado_{task_name}.png', show=True) 


# --- 7. Ejecución Principal ---
def main_analysis():
    # Define la tarea a analizar aquí
    TASK_TO_ANALYZE = 'MoveR' 

    # *** MODIFICACIÓN PRINCIPAL: Ruta de salida de las figuras directamente especificada ***
    global OUTPUT_PLOTS_DIR
    OUTPUT_PLOTS_DIR = Path(r"C:\Users\usuario\Desktop\MATLAB DRIVE\Coherence vs UPDRS\FIGURAS\Coherence_Beta- MoveR")
    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Los gráficos se guardarán en: {OUTPUT_PLOTS_DIR}")

    print("Iniciando carga y preparación de datos...")
    # Pasa la tarea a prepare_merged_data
    merged_data, df_deltas = prepare_merged_data(task_name=TASK_TO_ANALYZE)
    
    if merged_data.empty:
        print("No hay datos suficientes para el análisis. Saliendo.")
        return

    print("\n--- Análisis Estadístico ---")
    
    # 1. Comparación UPDRS MedOff vs MedOn
    print("\nAnálisis de Puntuación UPDRS:")
    perform_paired_ttest(merged_data, 'updrs_score', 'MedOff', 'MedOn')
    
    # 2. Comparación de Coherencia MedOff vs MedOn
    print("\nAnálisis de Coherencia por Banda y Condición:")
    for band in BANDS: # Solo se ejecutará para 'beta'
        print(f"  Banda: {band.capitalize()}")
        perform_paired_ttest(merged_data, band, 'MedOff', 'MedOn')

    # 3. Correlaciones
    print("\nCorrelaciones Coherencia vs UPDRS:")
    for band in BANDS: # Solo se ejecutará para 'beta'
        print(f"  Banda: {band.capitalize()}")
        # MedOff
        print("    Condición MedOff:")
        perform_correlation(merged_data[merged_data['condition'] == 'MedOff'], band, 'updrs_score')
        # MedOn
        print("    Condición MedOn:")
        perform_correlation(merged_data[merged_data['condition'] == 'MedOn'], band, 'updrs_score')
        # Deltas
        if not df_deltas.empty:
            print("    Deltas (MedOn - MedOff):")
            perform_correlation(df_deltas, f'delta_{band}_coh', 'delta_updrs')
        else:
            print("    No hay suficientes datos de deltas para correlación.")

    print("\n--- Generando Gráficos ---")
    # Gráfico de UPDRS (se mantiene)
    plot_updrs_violin_boxplot(merged_data)

    # Gráficos de Coherencia por banda (solo para Beta)
    for band in BANDS: # Solo se ejecutará para 'beta'
        # Pasar la tarea a las funciones de graficación
        plot_coherence_distributions(merged_data, band, task_name=TASK_TO_ANALYZE)
        plot_coherence_updrs_correlation(df_deltas, merged_data, band, task_name=TASK_TO_ANALYZE)

    print("\nAnálisis y graficación completados. Los gráficos se guardaron en:", OUTPUT_PLOTS_DIR)

if __name__ == "__main__":
    main_analysis()
