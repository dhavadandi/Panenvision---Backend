# -*- coding: utf-8 -*-
"""
PANENVISION Backend - Master Pipeline Runner (run_all.py)
==========================================================
Menjalankan seluruh siklus backend dari hulu ke hilir:
1. Uji Integritas Data & Zero Data Leakage Test
2. Pelatihan Model Utama (LightGBM + Lag 1) & Benchmark (XGBoost, CatBoost, SARIMA, SARIMAX)
3. Evaluasi Metrik Komprehensif (MAE, RMSE, MASE Global, MASE Macro)
4. Uji Signifikansi Statistik Diebold-Mariano (HAC Newey-West lag 3 vs LightGBM)
5. Split-Conformal Mondrian Calibration (P10, P50, P90 Bebas-Distribusi)
6. Optimasi Logistik Distribusi MILP 3-Skenario (HiGHS Solver)
7. Penyimpanan Seluruh Artefak ke folder outputs/

Jalankan dengan perintah:
    python run_all.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.config import (CSV_PATH, OUTPUT_DIR, CUT, N_IDX, TRAIN_IDX, TEST_IDX,
                        KEYS, LGBM_KW, XGB_KW, CAT_KW, SEED)
from src.features import (prepare_features, add_projected_consumption,
                          rend_from_train, compute_scales, PFEAT)
from src.models import fit_predict_scaled_tree, rolling_sarimax_forecast
from src.conformal import run_mondrian_conformal
from src.milp import run_3scenario_milp
from src.eval import evaluate_forecast, dm_hac_monthly, dm_iid_pooled
from tests.test_leakage import run_leakage_test


def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("🌾 PANENVISION BACKEND v2.0 - SISTEM PERAMALAN & OPTIMASI DISTRIBUSI PANGAN")
    print("=" * 80)
    print(f"[INFO] Dataset Path : {CSV_PATH}")
    print(f"[INFO] Output Dir   : {OUTPUT_DIR}")
    print(f"[INFO] Random Seed  : {SEED}")

    # ------------------------------------------------------------------------
    # 1. Uji Integritas & Kebocoran Data (Zero Data Leakage Test)
    # ------------------------------------------------------------------------
    print("\n[STEP 1/5] Memeriksa Integritas & Bebas Kebocoran Data...")
    run_leakage_test()

    # ------------------------------------------------------------------------
    # 2. Persiapan Fitur & Skala Normalisasi
    # ------------------------------------------------------------------------
    print("\n[STEP 2/5] Membangun Fitur Peramalan & Skala MASE...")
    df_raw = pd.read_csv(CSV_PATH)
    P = prepare_features(df_raw, CUT)
    P = add_projected_consumption(P, TRAIN_IDX)
    rend = rend_from_train(P, TRAIN_IDX)
    (series_scale_prod, series_scale_sur, kom_scales_prod,
     scale_prod_global, scale_sur_global) = compute_scales(P, TRAIN_IDX)

    TE_BASE = P[P["time_idx"].isin(TEST_IDX)][
        KEYS + ["tanggal", "bulan", "produksi_ton", "konsumsi_ton",
                "konsumsi_proyeksi_ton", "surplus_defisit_ton"]].copy()

    Xall = P.dropna(subset=PFEAT).copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xall[["kab_id", "kom_id"]] = enc.fit_transform(Xall[["kabupaten_kota", "komoditas"]])
    feats = PFEAT + ["kab_id", "kom_id"]

    tr = Xall[Xall["time_idx"].isin(TRAIN_IDX)].copy()
    te = Xall[Xall["time_idx"].isin(TEST_IDX)].copy()

    # ------------------------------------------------------------------------
    # 3. Pelatihan Model & Evaluasi Forecasting 1-Langkah
    # ------------------------------------------------------------------------
    print("\n[STEP 3/5] Melatih Model Peramalan Titik...")
    preds_store = {}
    prod_results = {}
    sur_results = {}
    kom_mase_table = {}

    def record_eval(name, pred_array):
        te_eval = TE_BASE.merge(te[KEYS].assign(prod_pred=pred_array), on=KEYS, how="inner")
        p = np.clip(te_eval["prod_pred"].values.astype(float), 0.0, None)
        zero_mask = te_eval.set_index(["kabupaten_kota", "komoditas"]).index.map(
            lambda k: np.isnan(series_scale_prod.get(k, np.nan))).values
        p = np.where(zero_mask, 0.0, p)
        a = te_eval["produksi_ton"].values.astype(float)
        te_eval["pred_clean"] = p
        preds_store[name] = te_eval

        # Evaluasi Produksi
        res = evaluate_forecast(a, p, scale_prod_global, kom_scales_prod, te_eval["komoditas"].values)
        prod_results[name] = {k: res[k] for k in ["MAE", "RMSE", "MASE_Global", "MASE_Macro"]}
        kom_mase_table[name] = res["kom_mase"]

        # Evaluasi Surplus Jujur
        sh = p * te_eval["komoditas"].map(rend).values - te_eval["konsumsi_proyeksi_ton"].values
        sa = te_eval["surplus_defisit_ton"].values.astype(float)
        sur_mae = float(np.mean(np.abs(sa - sh)))
        sur_results[name] = dict(
            Surplus_MAE=sur_mae,
            Surplus_RMSE=float(np.sqrt(np.mean((sa - sh) ** 2))),
            Surplus_MASE_Global=float(sur_mae / scale_sur_global)
        )
        print(f"  -> {name:16s} | MAE: {res['MAE']:7.2f} | RMSE: {res['RMSE']:7.2f} | "
              f"MASE Global: {res['MASE_Global']:.3f} | MASE Macro: {res['MASE_Macro']:.3f}")

    # (A) LightGBM (+ Lag 1) - Juara Utama
    print("  Fitting LightGBM (+ Lag 1)...")
    pred_lgb, _ = fit_predict_scaled_tree(LGBMRegressor, LGBM_KW, tr, te, feats)
    record_eval("LightGBM", pred_lgb)

    # (B) XGBoost (+ Lag 1)
    print("  Fitting XGBoost (+ Lag 1)...")
    pred_xgb, _ = fit_predict_scaled_tree(XGBRegressor, XGB_KW, tr, te, feats)
    record_eval("XGBoost", pred_xgb)

    # (C) CatBoost (+ Lag 1)
    print("  Fitting CatBoost (+ Lag 1)...")
    cfeats = PFEAT + ["kabupaten_kota", "komoditas"]
    pred_cat, _ = fit_predict_scaled_tree(
        CatBoostRegressor, CAT_KW, tr, te, cfeats, cat_cols=["kabupaten_kota", "komoditas"]
    )
    record_eval("CatBoost", pred_cat)

    # (D) Seasonal Naive Baseline
    snv_pred = te["prod_lag12"].values
    record_eval("Seasonal-Naive", snv_pred)

    # (E) Rolling SARIMA & SARIMAX
    print("  Fitting Rolling SARIMA (245 deret)...")
    sar_df, _, _, _ = rolling_sarimax_forecast(P, use_exog=False)
    sar_merged = te[KEYS].merge(sar_df, on=KEYS, how="left")
    record_eval("SARIMA", sar_merged["prod_pred"].values)

    print("  Fitting Rolling SARIMAX (245 deret)...")
    sax_df, res_sax, _, _ = rolling_sarimax_forecast(P, use_exog=True)
    sax_merged = te[KEYS].merge(sax_df, on=KEYS, how="left")
    record_eval("SARIMAX", sax_merged["prod_pred"].values)

    # (F) Hybrid SARIMAX + XGB
    print("  Fitting Hybrid SARIMAX + XGB...")
    DH = Xall.merge(res_sax, on=KEYS, how="left")
    trh = DH[DH["time_idx"].isin(TRAIN_IDX) & DH["resid"].notna()].copy()
    teh = DH[DH["time_idx"].isin(TEST_IDX)].copy()
    xgr = XGBRegressor(objective="reg:absoluteerror", n_estimators=400, max_depth=4,
                       learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                       reg_lambda=2.0, random_state=SEED, n_jobs=-1)
    xgr.fit(trh[feats], trh["resid"] / trh["series_mean"].clip(lower=1.0))
    teh = teh.merge(sax_df[KEYS + ["prod_pred"]].rename(columns={"prod_pred": "sarimax_pred"}), on=KEYS)
    teh["hybrid_pred"] = (teh["sarimax_pred"] + xgr.predict(teh[feats]) * teh["series_mean"].clip(lower=1.0)).clip(lower=0.0)
    record_eval("SARIMAX+XGB", teh["hybrid_pred"].values)

    # Uji Diebold-Mariano vs LightGBM
    base_lgbm = preds_store["LightGBM"][KEYS + ["produksi_ton", "pred_clean"]].rename(
        columns={"pred_clean": "pred_lgbm"})
    dm_rows = []
    for mname, mdf in preds_store.items():
        if mname == "LightGBM":
            dm_rows.append((mname, 0.0, 1.0, 0.0, 1.0))
            continue
        mm = base_lgbm.merge(mdf[KEYS + ["pred_clean"]].rename(columns={"pred_clean": "pred_other"}), on=KEYS)
        e_l = mm["produksi_ton"].values - mm["pred_lgbm"].values
        e_o = mm["produksi_ton"].values - mm["pred_other"].values
        mm = mm.assign(_d=np.abs(e_l) - np.abs(e_o))
        d_monthly = mm.groupby("time_idx")["_d"].mean().sort_index().values
        stat_hac, p_hac = dm_hac_monthly(d_monthly, lag=3)
        stat_iid, p_iid = dm_iid_pooled(e_l, e_o)
        dm_rows.append((mname, stat_hac, p_hac, stat_iid, p_iid))

    dm_df = pd.DataFrame(dm_rows, columns=["Model", "DM_vs_LightGBM", "p_value_HAC",
                                           "DM_iid_pooled_vs_LightGBM", "p_value_iid"])

    lb = pd.DataFrame(prod_results).T.reset_index().rename(columns={"index": "Model"})
    lb_sur = pd.DataFrame(sur_results).T.reset_index().rename(columns={"index": "Model"})
    lb = lb.merge(lb_sur, on="Model", how="left").merge(dm_df, on="Model", how="left")
    lb = lb.sort_values("MASE_Global")

    lb.to_csv(os.path.join(OUTPUT_DIR, "leaderboard_forecasting.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(kom_mase_table).T.to_csv(os.path.join(OUTPUT_DIR, "mase_per_komoditas.csv"), encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("🏆 LEADERBOARD FORECASTING RESMI (Peringkat Berdasarkan MASE Global)")
    print("=" * 80)
    print(lb[["Model", "MAE", "RMSE", "MASE_Global", "MASE_Macro", "DM_vs_LightGBM", "p_value_HAC"]].round(3).to_string(index=False))

    # ------------------------------------------------------------------------
    # 4. Split-Conformal Mondrian Calibration
    # ------------------------------------------------------------------------
    print("\n[STEP 4/5] Menjalankan Mondrian Conformal Prediction pada LightGBM...")
    conf_df, conf_meta = run_mondrian_conformal(df_raw)
    conf_df.round(2).to_csv(os.path.join(OUTPUT_DIR, "ramalan_kuantil_conformal.csv"), index=False, encoding="utf-8-sig")
    print(f"  Cakupan Empiris Produksi : {conf_meta['cov_prod']*100:.1f}% (Target Nominal: 80.0%)")
    print(f"  Cakupan Empiris Surplus  : {conf_meta['cov_sur']*100:.1f}% (Target Nominal: 80.0%)")
    print(f"  Lebar Rata-rata Interval : Produksi {conf_meta['width_prod']:,.1f} ton | Surplus {conf_meta['width_sur']:,.1f} ton")

    # ------------------------------------------------------------------------
    # 5. Optimasi MILP 3-Skenario
    # ------------------------------------------------------------------------
    print("\n[STEP 5/5] Menyelesaikan Optimasi Logistik Distribusi MILP 3-Skenario...")
    sum_df, ship_df, agg_milp = run_3scenario_milp(conf_df)
    sum_df.to_csv(os.path.join(OUTPUT_DIR, "ringkasan_milp_3skenario.csv"), index=False, encoding="utf-8-sig")
    ship_df.to_csv(os.path.join(OUTPUT_DIR, "rekomendasi_rute_distribusi_3skenario.csv"), index=False, encoding="utf-8-sig")
    agg_milp.to_csv(os.path.join(OUTPUT_DIR, "tabel_rekap_milp_3skenario.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("📦 REKAPITULASI HASIL DISTRIBUSI PANGAN 3-SKENARIO (MILP)")
    print("=" * 80)
    print(agg_milp.to_string(index=False))

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"✅ SELURUH PIPELINE SELESAI DALAM {elapsed:.1f} DETIK!")
    print(f"📁 Seluruh artefak telah disimpan di: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
