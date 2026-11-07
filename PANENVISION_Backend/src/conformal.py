# -*- coding: utf-8 -*-
"""
PANENVISION Backend - Mondrian Conformal Prediction
=====================================================
Implementasi textbook Split-Conformal Mondrian:
- Penjaminan cakupan bebas-distribusi (distribution-free valid coverage)
- Dikalibrasi per kelompok komoditas (Mondrian stratification)
- Koreksi ukuran sampel hingga: ceil((n+1)(1-alpha))/n pada alpha=0,20 (cakupan 80%)
- Menghasilkan batas bawah P10 (pesimis), nilai tengah P50 (moderat), dan batas atas P90 (optimis).
"""

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor
from .config import (KEYS, PROPER_END, CAL_IDX, TEST_IDX, ALPHA_CONF,
                     LGBM_KW)
from .features import prepare_features, add_projected_consumption, rend_from_train, PFEAT
from .models import fit_predict_scaled_tree


def run_mondrian_conformal(df):
    """
    Menjalankan prosedur Split-Conformal Mondrian pada model LightGBM (+ Lag 1).
    Menghasilkan DataFrame kuantil produksi dan surplus P10, P50, P90.
    """
    proper_idx = list(range(0, PROPER_END + 1))
    Pc = prepare_features(df, PROPER_END)
    Pc = add_projected_consumption(Pc, proper_idx)
    rend_pc = rend_from_train(Pc, proper_idx)

    Xc = Pc.dropna(subset=PFEAT).copy()
    enc_c = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xc[["kab_id", "kom_id"]] = enc_c.fit_transform(Xc[["kabupaten_kota", "komoditas"]])
    feats = PFEAT + ["kab_id", "kom_id"]

    fit_c = Xc[Xc["time_idx"] <= PROPER_END].copy()
    cal_c = Xc[Xc["time_idx"].isin(CAL_IDX)].copy()
    test_c = Xc[Xc["time_idx"].isin(TEST_IDX)].copy()

    # Latih model penskor proper-train
    p_cal, model_conf = fit_predict_scaled_tree(LGBMRegressor, LGBM_KW, fit_c, cal_c, feats)

    def predict_scaled(model, frame):
        return np.clip(model.predict(frame[feats]) * frame["series_mean"].clip(lower=1.0).values, 0.0, None)

    # Prediksi uji dari model proper-train yang sama (tanpa refit)
    p_test_conf = predict_scaled(model_conf, test_c)

    # 1. Kalibrasi Conformal Produksi
    sigma_cal = cal_c["prod_seasmean"].clip(lower=1.0).values
    scores_all = np.abs(cal_c["produksi_ton"].values - p_cal) / sigma_cal
    cal_scored = cal_c.assign(_r=scores_all)

    q_map, n_map = {}, {}
    for kom, grp in cal_scored.groupby("komoditas"):
        sc_k = np.sort(grp["_r"].values)
        n_k = len(sc_k)
        level = min(1.0, math.ceil((n_k + 1) * (1.0 - ALPHA_CONF)) / n_k)
        q_map[kom] = float(np.quantile(sc_k, level, method="higher"))
        n_map[kom] = n_k

    # 2. Kalibrasi Conformal Surplus Langsung
    sig_sur_map = (Pc[Pc["time_idx"] <= PROPER_END]
                   .groupby(["kabupaten_kota", "komoditas"])["surplus_defisit_ton"]
                   .apply(lambda v: float(np.mean(np.abs(v)))))
    sig_sur_map = sig_sur_map.clip(lower=1.0)

    sur_cal_pred = (p_cal * cal_c["komoditas"].map(rend_pc).values
                    - cal_c["konsumsi_proyeksi_ton"].values)
    sig_sur_cal = cal_c.set_index(["kabupaten_kota", "komoditas"]).index.map(
        lambda k: float(sig_sur_map.get(k, 1.0))).values
    scores_sur_all = (np.abs(cal_c["surplus_defisit_ton"].values - sur_cal_pred) / sig_sur_cal)
    cal_scored_sur = cal_c.assign(_rs=scores_sur_all)

    q_sur_map, n_sur_map = {}, {}
    for kom, grp in cal_scored_sur.groupby("komoditas"):
        sc_k = np.sort(grp["_rs"].values)
        n_k = len(sc_k)
        level = min(1.0, math.ceil((n_k + 1) * (1.0 - ALPHA_CONF)) / n_k)
        q_sur_map[kom] = float(np.quantile(sc_k, level, method="higher"))
        n_sur_map[kom] = n_k

    # 3. Bentuk Interval di Data Uji
    sigma_te = test_c["prod_seasmean"].clip(lower=1.0).values
    q_te = test_c["komoditas"].map(q_map).values
    p10_prod = np.clip(p_test_conf - q_te * sigma_te, 0.0, None)
    p90_prod = p_test_conf + q_te * sigma_te

    rend_te = test_c["komoditas"].map(rend_pc).values
    sur_point = p_test_conf * rend_te - test_c["konsumsi_proyeksi_ton"].values
    sig_sur_te = test_c.set_index(["kabupaten_kota", "komoditas"]).index.map(
        lambda k: float(sig_sur_map.get(k, 1.0))).values
    q_sur_te = test_c["komoditas"].map(q_sur_map).values

    p10_sur = sur_point - q_sur_te * sig_sur_te
    p90_sur = sur_point + q_sur_te * sig_sur_te

    conf = test_c[KEYS + ["tanggal", "produksi_ton", "surplus_defisit_ton", "konsumsi_proyeksi_ton"]].copy()
    conf["prod_p10"], conf["prod_p50"], conf["prod_p90"] = p10_prod, p_test_conf, p90_prod
    conf["p10"], conf["p50"], conf["p90"] = p10_sur, sur_point, p90_sur

    conf["defisit_robust_ton"] = (-conf["p10"]).clip(lower=0.0)
    conf["surplus_aman_ton"] = conf["p10"].clip(lower=0.0)

    # Hitung cakupan empiris
    cov_prod = float(((conf["produksi_ton"] >= conf["prod_p10"]) &
                      (conf["produksi_ton"] <= conf["prod_p90"])).mean())
    cov_sur = float(((conf["surplus_defisit_ton"] >= conf["p10"]) &
                     (conf["surplus_defisit_ton"] <= conf["p90"])).mean())
    width_prod = float((conf["prod_p90"] - conf["prod_p10"]).mean())
    width_sur = float((conf["p90"] - conf["p10"]).mean())

    metrics = dict(
        cov_prod=cov_prod, cov_sur=cov_sur,
        width_prod=width_prod, width_sur=width_sur,
        q_map=q_map, q_sur_map=q_sur_map,
        n_map=n_map, n_sur_map=n_sur_map
    )
    return conf, metrics
