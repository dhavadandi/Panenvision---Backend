# -*- coding: utf-8 -*-
"""
PANENVISION Backend - Feature Engineering
==========================================
Pembangunan fitur peramalan murni (zero-leakage) dengan memasukkan fitur
autoregresif lag-1 (prod_lag1, ratio_lag1, spatial_lag1), lag musiman,
dan kovariat iklim/NDVI yang dibatasi strictly pada waktu lampau (t < t_ramal).
"""

import numpy as np
import pandas as pd
from .config import REG, build_spatial_weights

# Definisi Fitur Model Pohon (22 Fitur Inti)
PFEAT = [
    "prod_lag1", "ratio_lag1",
    "prod_lag12", "prod_lag24", "prod_seasmean", "spatial_lag12",
    "ratio_lag12", "ratio_lag24", "ratio_seasmean", "ratio_spatial12",
    "month_sin", "month_cos", "flag_panenraya", "flag_paceklik",
    "rain_lag1", "rain_lag2", "rain_mean3", "extreme_lag1",
    "ndvi_lag1", "ndvi_lag2", "ndvi_anomaly_lag1",
    "populasi_jiwa",
]

# Kovariat Eksogen SARIMAX (Hanya Variabel t-1 dan Kalender)
EXOG_SARIMAX = [
    "month_sin", "month_cos", "flag_panenraya", "flag_paceklik",
    "rain_lag1", "ndvi_lag1"
]


def prepare_features(df, clim_end_idx):
    """
    Membangun fitur peramalan 1-langkah ke depan tanpa kebocoran data.
    Semua statistik klimatologi dihitung HANYA dari time_idx <= clim_end_idx.
    """
    P = df.copy()
    P["tanggal"] = pd.to_datetime(P["tanggal"])
    P["time_idx"] = (P["tanggal"].dt.year - 2019) * 12 + (P["tanggal"].dt.month - 1)
    P = P.sort_values(["kabupaten_kota", "komoditas", "time_idx"]).reset_index(drop=True)

    g = P.groupby(["kabupaten_kota", "komoditas"], group_keys=False)
    P["prod_lag1"] = g["produksi_ton"].shift(1)
    P["prod_lag2"] = g["produksi_ton"].shift(2)
    P["prod_lag12"] = g["produksi_ton"].shift(12)
    P["prod_lag24"] = g["produksi_ton"].shift(24)
    P["rain_lag1"] = g["curah_hujan_mm"].shift(1)
    P["rain_lag2"] = g["curah_hujan_mm"].shift(2)
    P["rain_lag3"] = g["curah_hujan_mm"].shift(3)
    P["rain_mean3"] = (P["rain_lag1"] + P["rain_lag2"] + P["rain_lag3"]) / 3.0
    P["extreme_lag1"] = g["cuaca_ekstrem"].shift(1)
    P["ndvi_lag1"] = g["ndvi_modis"].shift(1)
    P["ndvi_lag2"] = g["ndvi_modis"].shift(2)

    # Klimatologi dari jendela latih
    tr_sub = P[P["time_idx"] <= clim_end_idx]
    seas = (tr_sub.groupby(["kabupaten_kota", "komoditas", "bulan"])["produksi_ton"]
            .mean().rename("prod_seasmean").reset_index())
    ser_mean = (tr_sub.groupby(["kabupaten_kota", "komoditas"])["produksi_ton"]
                .mean().rename("series_mean").reset_index())
    ndvi_seas = (tr_sub.groupby(["kabupaten_kota", "bulan"])["ndvi_modis"]
                 .mean().rename("ndvi_seasmean").reset_index())

    P = P.merge(seas, on=["kabupaten_kota", "komoditas", "bulan"], how="left")
    P = P.merge(ser_mean, on=["kabupaten_kota", "komoditas"], how="left")
    P = P.merge(ndvi_seas, on=["kabupaten_kota", "bulan"], how="left")
    P = P.sort_values(["kabupaten_kota", "komoditas", "time_idx"]).reset_index(drop=True)

    # Spasial lag tahunan (lag12) dan bulanan (lag1)
    W = build_spatial_weights()
    piv = P.pivot(index=["komoditas", "time_idx"], columns="kabupaten_kota", values="prod_lag12")[REG]
    sp_mat = piv.values @ W.T
    sp_df = (pd.DataFrame(sp_mat, index=piv.index, columns=pd.Index(REG, name="kabupaten_kota"))
             .stack(future_stack=True).rename("spatial_lag12").reset_index())
    P = P.merge(sp_df, on=["komoditas", "time_idx", "kabupaten_kota"], how="left")

    piv1 = P.pivot(index=["komoditas", "time_idx"], columns="kabupaten_kota", values="prod_lag1")[REG]
    sp_mat1 = piv1.values @ W.T
    sp_df1 = (pd.DataFrame(sp_mat1, index=piv1.index, columns=pd.Index(REG, name="kabupaten_kota"))
             .stack(future_stack=True).rename("spatial_lag1").reset_index())
    P = P.merge(sp_df1, on=["komoditas", "time_idx", "kabupaten_kota"], how="left")
    P = P.sort_values(["kabupaten_kota", "komoditas", "time_idx"]).reset_index(drop=True)

    # Rasio normalisasi terhadap rata-rata deret
    sc = P["series_mean"].clip(lower=1.0)
    P["ratio_lag1"] = P["prod_lag1"] / sc
    P["ratio_lag2"] = P["prod_lag2"] / sc
    P["ratio_lag12"] = P["prod_lag12"] / sc
    P["ratio_lag24"] = P["prod_lag24"] / sc
    P["ratio_seasmean"] = P["prod_seasmean"] / sc
    P["ratio_spatial12"] = P["spatial_lag12"] / sc
    P["ratio_spatial1"] = P["spatial_lag1"] / sc
    P["ndvi_anomaly_lag1"] = P["ndvi_lag1"] / P["ndvi_seasmean"].clip(lower=0.1)

    # Kalender & Musiman
    P["month_sin"] = np.sin(2 * np.pi * P["bulan"] / 12.0)
    P["month_cos"] = np.cos(2 * np.pi * P["bulan"] / 12.0)
    P["flag_panenraya"] = P["bulan"].isin([3, 4]).astype(int)
    P["flag_paceklik"] = P["bulan"].isin([11, 12, 1]).astype(int)
    return P


def add_projected_consumption(df, train_idx):
    """Menghitung proyeksi konsumsi per kapita kalender dari data latih."""
    tr = df[df["time_idx"].isin(train_idx)].copy()
    tr["pk"] = tr["konsumsi_ton"] / tr["populasi_jiwa"].clip(lower=1.0)
    seas_pk = (tr.groupby(["kabupaten_kota", "komoditas", "bulan"])["pk"]
               .mean().rename("pk_seasmean").reset_index())
    out = df.merge(seas_pk, on=["kabupaten_kota", "komoditas", "bulan"], how="left")
    out = out.sort_values(["kabupaten_kota", "komoditas", "time_idx"]).reset_index(drop=True)
    out["konsumsi_proyeksi_ton"] = out["pk_seasmean"] * out["populasi_jiwa"]
    return out


def rend_from_train(df, train_idx):
    """Menghitung rendemen rata-rata per komoditas dari data latih."""
    tr = df[df["time_idx"].isin(train_idx)].copy()
    tr["r_eff"] = (tr["surplus_defisit_ton"] + tr["konsumsi_ton"]) / tr["produksi_ton"].clip(lower=1e-3)
    tr = tr[tr["produksi_ton"] > 10.0]
    return tr.groupby("komoditas")["r_eff"].median().to_dict()


def compute_scales(df, train_idx):
    """Menghitung penyebut skala MASE dari galat naive musiman data latih."""
    tr = df[df["time_idx"].isin(train_idx)].copy()
    g = tr.groupby(["kabupaten_kota", "komoditas"], group_keys=False)
    diff = np.abs(tr["produksi_ton"].values - g["produksi_ton"].shift(12).values)
    tr["_diff"] = diff
    v = tr.dropna(subset=["_diff"])

    series_scale_prod = v.groupby(["kabupaten_kota", "komoditas"])["_diff"].mean().to_dict()
    scale_prod_global = float(v["_diff"].mean())

    kom_scales_prod = {}
    for kom, grp in v.groupby("komoditas"):
        m = float(grp["_diff"].mean())
        kom_scales_prod[kom] = m if m > 1e-6 else 1.0

    diff_sur = np.abs(tr["surplus_defisit_ton"].values - g["surplus_defisit_ton"].shift(12).values)
    tr["_diff_sur"] = diff_sur
    vs = tr.dropna(subset=["_diff_sur"])
    series_scale_sur = vs.groupby(["kabupaten_kota", "komoditas"])["_diff_sur"].mean().to_dict()
    scale_sur_global = float(vs["_diff_sur"].mean())
    return (series_scale_prod, series_scale_sur, kom_scales_prod,
            scale_prod_global, scale_sur_global)
