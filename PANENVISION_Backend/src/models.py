# -*- coding: utf-8 -*-
"""
PANENVISION Backend - Machine Learning & Econometric Models
============================================================
Implementasi model peramalan:
1. Scaled Tree Regressor (LightGBM - Juara Utama, XGBoost, CatBoost)
2. Rolling SARIMA / SARIMAX (Benchmark ekonometrika)
3. Residual Hybrid (SARIMAX + XGB)
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .config import SEED, KEYS, CUT, TEST_IDX, EXOG_SARIMAX


def fit_predict_scaled_tree(model_cls, kwargs, train_df, test_df, features,
                            target="produksi_ton", scale_col="series_mean",
                            cat_cols=None):
    """
    Melatih model pohon pada rasio ternormalisasi: y_scaled = target / series_mean.
    Prediksi dikalikan kembali dengan series_mean dan di-clip pada batas bawah 0.
    """
    sc_tr = train_df[scale_col].clip(lower=1.0).values
    y_tr = train_df[target].values / sc_tr
    X_tr = train_df[features].copy()
    X_te = test_df[features].copy()

    if cat_cols is not None:
        model = model_cls(cat_features=cat_cols, **kwargs)
    else:
        model = model_cls(**kwargs)

    model.fit(X_tr, y_tr)

    sc_te = test_df[scale_col].clip(lower=1.0).values
    pred_scaled = model.predict(X_te)
    if pred_scaled.ndim == 1:
        pred = np.clip(pred_scaled * sc_te, 0.0, None)
    else:
        pred = np.clip(pred_scaled * sc_te[:, None], 0.0, None)
    return pred, model


def rolling_sarimax_forecast(df, use_exog=False):
    """
    Peramalan 1-langkah rolling SARIMA(1,0,1)(1,0,1)12 atau SARIMAX.
    Model dilatih pada data latih (t <= 66) lalu state internalnya di-append
    setiap bulan uji tanpa refit penuh.
    """
    pats = []
    res_rows = []
    n_fallbacks = 0
    first_exc = None

    exog_cols = EXOG_SARIMAX if use_exog else None

    # Pengisian NaN eksogen data latih untuk stabilitas statsmodels
    if use_exog:
        ex_means = df[df["time_idx"] <= CUT][exog_cols].mean().to_dict()
        df = df.copy()
        for c in exog_cols:
            df[c] = df[c].fillna(ex_means.get(c, 0.0))

    for (kab, kom), grp in df.groupby(["kabupaten_kota", "komoditas"]):
        sub = grp.sort_values("time_idx").reset_index(drop=True)
        y = sub["produksi_ton"].values.astype(float)
        ex = sub[exog_cols].values.astype(float) if use_exog else None

        train_mask = sub["time_idx"] <= CUT
        y_tr = y[train_mask]
        ex_tr = ex[train_mask] if use_exog else None

        # Deret yang seluruhnya nol
        if (y_tr == 0).all():
            for t_idx in TEST_IDX:
                pats.append(dict(kabupaten_kota=kab, komoditas=kom, time_idx=t_idx, prod_pred=0.0))
            continue

        try:
            m = SARIMAX(
                y_tr,
                exog=ex_tr,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 12),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = m.fit(disp=False, maxiter=80)
            state = res

            # Catat residu in-sample untuk hybrid
            in_preds = res.fittedvalues
            for idx_tr, t_tr in enumerate(sub[train_mask]["time_idx"]):
                res_rows.append(dict(
                    kabupaten_kota=kab, komoditas=kom, time_idx=t_tr,
                    resid=float(y_tr[idx_tr] - in_preds[idx_tr])
                ))

            # Rolling 1-langkah di data uji
            for t_idx in TEST_IDX:
                pos = int(np.where(sub["time_idx"].values == t_idx)[0][0])
                ex_step = ex[pos:pos+1] if use_exog else None
                f = float(state.forecast(steps=1, exog=ex_step)[0])
                pats.append(dict(kabupaten_kota=kab, komoditas=kom, time_idx=t_idx, prod_pred=max(f, 0.0)))
                # Update state dengan data aktual
                state = state.append([y[pos]], exog=(ex[pos:pos+1] if use_exog else None), refit=False)

        except Exception as e:
            n_fallbacks += 1
            if first_exc is None:
                first_exc = f"{type(e).__name__}: {e}"
            # Fallback ke lag musiman t-12
            for t_idx in TEST_IDX:
                pos = int(np.where(sub["time_idx"].values == t_idx)[0][0])
                pats.append(dict(
                    kabupaten_kota=kab, komoditas=kom, time_idx=t_idx,
                    prod_pred=max(float(y[pos - 12]), 0.0) if pos >= 12 else 0.0
                ))

    pred_df = pd.DataFrame(pats)
    resid_df = pd.DataFrame(res_rows) if res_rows else pd.DataFrame(columns=KEYS + ["resid"])
    return pred_df, resid_df, n_fallbacks, first_exc
