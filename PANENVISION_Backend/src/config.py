# -*- coding: utf-8 -*-
"""
PANENVISION Backend - Configuration & Constants
================================================
Definisi parameter geografis, temporal, hiperparameter model,
dan konstanta optimasi distribusi logistik pangan Jawa Tengah.
"""

import os
import math
import numpy as np

# Random Seed untuk Reproduksibilitas
SEED = 42

# Path Direktori
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CSV_PATH = os.path.join(DATA_DIR, "dataset_jateng_bulanan_2019_2025.csv")

# Parameter Temporal
N_IDX = 84                     # Total 84 bulan (Januari 2019 - Desember 2025)
CUT = 66                       # Batas data latih utama (t = 0..66: Jan 2019 - Jul 2024)
TRAIN_IDX = list(range(0, CUT + 1))
TEST_IDX = list(range(CUT + 1, N_IDX))  # Data uji 17 bulan (Ags 2024 - Des 2025)

# Parameter Conformal Prediction (Split-Conformal Mondrian)
PROPER_END = 54                # Proper training set (t = 0..54: Jan 2019 - Jul 2023)
CAL_IDX = list(range(PROPER_END + 1, CUT + 1))  # Kalibrasi (t = 55..66: Ags 2023 - Jul 2024)
ALPHA_CONF = 0.20              # Cakupan nominal 80% (P10 hingga P90)

# Koordinat Sentroid 35 Kabupaten/Kota Jawa Tengah (Bujur, Lintang)
CENTROIDS = {
    "Cilacap": (109.02, -7.62), "Banyumas": (109.25, -7.45),
    "Purbalingga": (109.36, -7.39), "Banjarnegara": (109.68, -7.38),
    "Kebumen": (109.65, -7.67), "Purworejo": (110.01, -7.71),
    "Wonosobo": (109.90, -7.36), "Magelang": (110.22, -7.47),
    "Boyolali": (110.59, -7.53), "Klaten": (110.60, -7.70),
    "Sukoharjo": (110.84, -7.68), "Wonogiri": (111.00, -7.90),
    "Karanganyar": (111.05, -7.60), "Sragen": (111.02, -7.43),
    "Grobogan": (110.92, -7.10), "Blora": (111.42, -7.05),
    "Rembang": (111.35, -6.79), "Pati": (111.05, -6.80),
    "Kudus": (110.84, -6.80), "Jepara": (110.67, -6.59),
    "Demak": (110.64, -6.90), "Semarang": (110.45, -7.15),
    "Temanggung": (110.18, -7.30), "Kendal": (110.20, -6.98),
    "Batang": (109.88, -6.95), "Pekalongan": (109.62, -6.95),
    "Pemalang": (109.38, -6.95), "Tegal": (109.15, -6.92),
    "Brebes": (108.92, -6.92), "Kota Magelang": (110.22, -7.48),
    "Kota Surakarta": (110.83, -7.56), "Kota Salatiga": (110.50, -7.33),
    "Kota Semarang": (110.42, -7.00), "Kota Pekalongan": (109.67, -6.89),
    "Kota Tegal": (109.14, -6.87),
}

REG = list(CENTROIDS.keys())


def haversine(a, b):
    """Menghitung jarak lingkaran besar antara dua koordinat bola bumi (km)."""
    (lon1, lat1), (lon2, lat2) = a, b
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


DIST = {(i, j): haversine(CENTROIDS[i], CENTROIDS[j]) for i in REG for j in REG}


def build_spatial_weights():
    """Membangun matriks bobot spasial inverse-distance (radius <= 100 km, row-standardized)."""
    W = np.zeros((len(REG), len(REG)), dtype=float)
    for idx_i, i in enumerate(REG):
        for idx_j, j in enumerate(REG):
            if i != j:
                d = DIST[(i, j)]
                if d <= 100.0:
                    W[idx_i, idx_j] = 1.0 / max(d, 5.0)
        row_s = W[idx_i].sum()
        if row_s > 0:
            W[idx_i] /= row_s
    return W


# Identitas Kunci Deret
KEYS = ["kabupaten_kota", "komoditas", "time_idx"]

# Parameter MILP Distribusi Logistik
TARIF = 2500.0           # Rp 2.500 per ton-km
ALPHA = 1.0              # Pemenuhan defisit 100%
DMAX_KM = 250.0          # Batas jangkauan rute truk 250 km
IMPORT_COST = 4.0 * TARIF * DMAX_KM  # Rp 2.500.000 per ton impor luar provinsi

# Hiperparameter Model Pohon
LGBM_KW = dict(
    objective="regression_l1",
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1
)

XGB_KW = dict(
    objective="reg:absoluteerror",
    n_estimators=600,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=-1
)

CAT_KW = dict(
    iterations=800,
    depth=6,
    learning_rate=0.03,
    loss_function="MAE",
    random_seed=SEED,
    verbose=0
)
