import subprocess
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
import warnings
import mne
import mne_connectivity
import os
from pathlib import Path
from technical_validation_utils import parula_map, scaling, interp

# Reducir la salida de mensajes de MNE
mne.set_log_level(verbose='CRITICAL')

# Definir rutas
COHERENCE_DATA_DIR = Path(r"C:\Users\usuario\Desktop\PYTHON SCRIPTS\coherence")
OUTPUT_PLOTS_DIR = Path(r"C:\Users\usuario\Desktop\MATLAB DRIVE\Coherence vs UPDRS\FIGURAS\Coherence_Final")
OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

freqs_beta = {'beta': (13, 30)}
dB = False

# --- Función para cargar datos de coherencia ---
def load_coherence_data(data_dir, bands_freqs):
    coherence_data = []
    base_name_pattern = re.compile(r'sub-(?P<subject>[a-zA-Z0-9]+)_ses-PeriOp_task-(?P<task>[a-zA-Z0-9]+)_acq-(?P<condition>MedOff|MedOn)_run-\d+$')

    for subject_folder in data_dir.iterdir():
        subject_id = subject_folder.name.replace('sub-', '')
        coh_files = sorted([f for f in subject_folder.iterdir() if f.name.endswith('_coh')])
        pkl_files = sorted([f for f in subject_folder.iterdir() if f.name.endswith('_ind.pkl')])

        coh_files_dict = {f.name.replace('_coh', ''): f for f in coh_files}
        pkl_files_dict = {f.name.replace('_ind.pkl', ''): f for f in pkl_files}

        for base_name in coh_files_dict.keys():
            coh_path = coh_files_dict[base_name]
            pkl_path = pkl_files_dict.get(base_name)
            if pkl_path is None:
                continue

            match = base_name_pattern.search(base_name)
            if not match:
                continue

            condition = match.group('condition')
            task = match.group('task')

            try:
                coh = mne_connectivity.read_connectivity(coh_path)
                with open(pkl_path, 'rb') as f:
                    coh_info = pickle.load(f)

                coh_data = coh.get_data()
                coh_freqs = np.array(coh.freqs)
                meg_info = coh_info['meg_info']

                coh_data_reshaped = coh_data.reshape(coh_info['n_lfp_sensors'], coh_info['n_meg_sensors'], len(coh_freqs))

                for band_name, (fmin, fmax) in bands_freqs.items():
                    band_idx = np.where((coh_freqs >= fmin) & (coh_freqs <= fmax))[0]
                    mean_coh_band = np.mean(coh_data_reshaped[:, :, band_idx], axis=2).mean() if len(band_idx) > 0 else np.nan

                    coherence_data.append({'subject': subject_id, 'condition': condition, 'band': band_name, 'coherence_mean': mean_coh_band, 'task': task})
            except:
                continue

    return pd.DataFrame(coherence_data), meg_info

# --- Cargar datos de coherencia ---
df_coh, meg_info_max = load_coherence_data(COHERENCE_DATA_DIR, freqs_beta)

# --- Calcular coherencia general por condición y tarea ---
df_coh_agg = df_coh.groupby(['condition', 'task'])['coherence_mean'].mean().reset_index()

print("\nCoherencia promedio por condición y tarea:")
print(df_coh_agg)

# --- Crear objetos SpectrumArray ---
grand_ave_coh = df_coh.pivot_table(index=['condition', 'task'], columns='band', values='coherence_mean')
grand_ave_coh_left = interp(grand_ave_coh.loc['MedOff']) / scaling
grand_ave_coh_right = interp(grand_ave_coh.loc['MedOn']) / scaling

grand_ave_coh_left = mne.time_frequency.SpectrumArray(grand_ave_coh_left, meg_info_max, freqs=np.array([13, 30]))
grand_ave_coh_right = mne.time_frequency.SpectrumArray(grand_ave_coh_right, meg_info_max, freqs=np.array([13, 30]))

# --- Graficar topomaps ---
fig, ax = plt.subplots(figsize=(8, 6))  # MedOff
grand_ave_coh_left.plot_topomap(bands=freqs_beta, res=300, cmap=(parula_map, True), dB=dB, axes=ax, cbar_fmt='%0.2f')
ax.set_title('Coherencia Promedio - MedOff')
fig.savefig('./figures/Coherence_MedOff.jpg', dpi=300)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))  # MedOn
grand_ave_coh_right.plot_topomap(bands=freqs_beta, res=300, cmap=(parula_map, True), dB=dB, axes=ax, cbar_fmt='%0.2f')
ax.set_title('Coherencia Promedio - MedOn')
fig.savefig('./figures/Coherence_MedOn.jpg', dpi=300)
plt.show()
