# -*- coding: utf-8 -*-
"""
PANENVISION Backend - Evaluation Metrics & Statistical Tests
=============================================================
1. Metrik Akurasi: MAE, RMSE, MASE Global, MASE Macro per komoditas.
2. Uji Signifikansi Statistik: Diebold-Mariano versi HAC (Newey-West lag 3)
   atas galat bulanan agregat, serta versi iid pooled pembanding.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def dm_iid_pooled(e1, e2):
    """Uji Diebold-Mariano varians pooled iid: var(d)/T."""
    d = np.abs(e1) - np.abs(e2)
    T = len(d)
    var_d = np.var(d, ddof=1) / T
    if var_d <= 0:
        return 0.0, 1.0
    stat = float(np.mean(d) / np.sqrt(var_d))
    return stat, float(2.0 * (1.0 - norm.cdf(abs(stat))))


def dm_hac_monthly(d_monthly, lag=3):
    """
    Uji Diebold-Mariano dengan varians heteroskedasticity and autocorrelation
    consistent (HAC / Newey-West) pada selisih galat absolut bulanan (T = 17).
    """
    T = len(d_monthly)
    if T < 2:
        return 0.0, 1.0
    d_bar = np.mean(d_monthly)
    gamma_0 = np.var(d_monthly, ddof=0)
    gamma_sum = 0.0
    for l in range(1, min(lag + 1, T)):
        w_l = 1.0 - l / (lag + 1.0)
        c = float(np.mean((d_monthly[l:] - d_bar) * (d_monthly[:-l] - d_bar)))
        gamma_sum += 2.0 * w_l * c
    lr_var = gamma_0 + gamma_sum
    if lr_var <= 0:
        return 0.0, 1.0
    se = np.sqrt(lr_var / T)
    stat = float(d_bar / se)
    p_val = float(2.0 * (1.0 - norm.cdf(abs(stat))))
    return stat, p_val


def evaluate_forecast(actual, pred, scale_global, scales_by_commodity, commodities):
    """
    Menghitung MAE, RMSE, MASE Global, dan MASE Macro.
    """
    err = actual - pred
    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mase_global = float(mae / scale_global)

    kom_mase = {}
    df_eval = pd.DataFrame({"act": actual, "pred": pred, "kom": commodities, "ae": abs_err})
    for kom, grp in df_eval.groupby("kom"):
        sc = scales_by_commodity.get(kom, 1.0)
        kom_mase[kom] = float(grp["ae"].mean() / sc)

    mase_macro = float(np.mean(list(kom_mase.values())))
    return dict(MAE=mae, RMSE=rmse, MASE_Global=mase_global, MASE_Macro=mase_macro, kom_mase=kom_mase)
