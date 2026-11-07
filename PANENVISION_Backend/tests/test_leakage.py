# -*- coding: utf-8 -*-
"""
Uji Kebocoran Data (Zero Data Leakage Test)
===========================================
Memastikan bahwa seluruh fitur peramalan pada bulan t mematuhi aturan kausalitas:
1. Fitur lag (prod_lag1, prod_lag2, prod_lag12, rain_lag1, ndvi_lag1, dsb.)
   secara eksak berasal dari time_idx < t.
2. Statistik klimatologi (prod_seasmean, series_mean, ndvi_seasmean)
   secara eksak dihitung HANYA dari jendela data latih (time_idx <= 66).
3. Pindai daftar fitur: tidak ada fitur kontemporer (bulan berjalan) yang terlarang.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CSV_PATH, CUT, REG, build_spatial_weights
from src.features import prepare_features, PFEAT, EXOG_SARIMAX

TOL = 1e-6
FORBIDDEN = {"curah_hujan_mm", "cuaca_ekstrem", "ndvi_modis", "ndvi_anomaly"}
SERIES = [("Cilacap", "Padi"), ("Grobogan", "Jagung"), ("Kota Semarang", "Kedelai")]
TEST_T = [67, 74, 83]
failures = []


def check(label, got, want):
    ok = (np.isfinite(got) and np.isfinite(want)
          and abs(got - want) <= TOL * max(1.0, abs(want)))
    if not ok:
        failures.append(f"{label}: fitur={got} hitung-ulang={want}")
    return ok


def run_leakage_test():
    df = pd.read_csv(CSV_PATH)
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df["time_idx"] = (df["tanggal"].dt.year - 2019) * 12 + (df["tanggal"].dt.month - 1)
    P = prepare_features(df, CUT)

    W = build_spatial_weights()
    reg_idx = {r: i for i, r in enumerate(REG)}

    n_checks = 0
    for kab, kom in SERIES:
        raw = (df[(df["kabupaten_kota"] == kab) & (df["komoditas"] == kom)]
               .sort_values("time_idx").set_index("time_idx"))
        feat = (P[(P["kabupaten_kota"] == kab) & (P["komoditas"] == kom)]
                .sort_values("time_idx").set_index("time_idx"))
        train_raw = raw[raw.index <= CUT]
        series_mean_manual = train_raw["produksi_ton"].mean()

        for t in TEST_T:
            row = feat.loc[t]
            bulan_t = int(raw.loc[t, "bulan"])
            tag = f"{kab}/{kom}/t={t}"
            pairs = [
                ("prod_lag1", row["prod_lag1"], raw.loc[t - 1, "produksi_ton"]),
                ("prod_lag2", row["prod_lag2"], raw.loc[t - 2, "produksi_ton"]),
                ("prod_lag12", row["prod_lag12"], raw.loc[t - 12, "produksi_ton"]),
                ("prod_lag24", row["prod_lag24"], raw.loc[t - 24, "produksi_ton"]),
                ("rain_lag1", row["rain_lag1"], raw.loc[t - 1, "curah_hujan_mm"]),
                ("rain_lag2", row["rain_lag2"], raw.loc[t - 2, "curah_hujan_mm"]),
                ("rain_mean3", row["rain_mean3"],
                 raw.loc[t - 3:t - 1, "curah_hujan_mm"].mean()),
                ("extreme_lag1", row["extreme_lag1"], raw.loc[t - 1, "cuaca_ekstrem"]),
                ("ndvi_lag1", row["ndvi_lag1"], raw.loc[t - 1, "ndvi_modis"]),
                ("ndvi_lag2", row["ndvi_lag2"], raw.loc[t - 2, "ndvi_modis"]),
                ("prod_seasmean", row["prod_seasmean"],
                 train_raw[train_raw["bulan"] == bulan_t]["produksi_ton"].mean()),
                ("series_mean", row["series_mean"], series_mean_manual),
                ("ratio_lag1", row["ratio_lag1"],
                 raw.loc[t - 1, "produksi_ton"] / max(series_mean_manual, 1.0)),
                ("ratio_lag2", row["ratio_lag2"],
                 raw.loc[t - 2, "produksi_ton"] / max(series_mean_manual, 1.0)),
                ("ratio_lag12", row["ratio_lag12"],
                 raw.loc[t - 12, "produksi_ton"] / max(series_mean_manual, 1.0)),
            ]

            ndvi_seas_manual = (df[(df["kabupaten_kota"] == kab)
                                   & (df["time_idx"] <= CUT)
                                   & (df["bulan"] == bulan_t)]["ndvi_modis"].mean())
            pairs.append(("ndvi_anomaly_lag1", row["ndvi_anomaly_lag1"],
                          raw.loc[t - 1, "ndvi_modis"] / max(ndvi_seas_manual, 0.1)))

            lag12_vec = []
            for r in REG:
                rr = (df[(df["kabupaten_kota"] == r) & (df["komoditas"] == kom)]
                      .sort_values("time_idx").set_index("time_idx"))
                lag12_vec.append(rr.loc[t - 12, "produksi_ton"])
            sp_manual = float(np.array(lag12_vec) @ W[reg_idx[kab]])
            pairs.append(("spatial_lag12", row["spatial_lag12"], sp_manual))

            lag1_vec = []
            for r in REG:
                rr = (df[(df["kabupaten_kota"] == r) & (df["komoditas"] == kom)]
                      .sort_values("time_idx").set_index("time_idx"))
                lag1_vec.append(rr.loc[t - 1, "produksi_ton"])
            sp_manual1 = float(np.array(lag1_vec) @ W[reg_idx[kab]])
            pairs.append(("spatial_lag1", row["spatial_lag1"], sp_manual1))
            pairs.append(("ratio_spatial1", row["ratio_spatial1"],
                          sp_manual1 / max(series_mean_manual, 1.0)))

            for name, got, want in pairs:
                n_checks += 1
                check(f"{tag} {name}", float(got), float(want))

    for lst_name, lst in [("PFEAT", PFEAT), ("EXOG_SARIMAX", EXOG_SARIMAX)]:
        bad = FORBIDDEN.intersection(lst)
        if bad:
            failures.append(f"{lst_name} memuat variabel terlarang: {bad}")

    if failures:
        print(f"GAGAL: {len(failures)} uji tidak lolos:")
        for f in failures[:10]:
            print("  ", f)
        sys.exit(1)
    else:
        print(f"LOLOS: {n_checks} pemeriksaan fitur (3 seri x 3 bulan uji x 18 fitur dihitung ulang) + zero leakage.")
        return True


if __name__ == "__main__":
    run_leakage_test()
