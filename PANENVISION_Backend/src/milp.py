# -*- coding: utf-8 -*-
"""
PANENVISION Backend - Robust 3-Scenario MILP Optimization
==========================================================
Optimasi alokasi distribusi surplus-defisit pangan lintas kabupaten/kota
menggunakan formulasi Mixed-Integer Linear Programming (MILP) untuk 3 skenario:
1. Skenario Pesimis (Batas Bawah P10) - Mode Robust
2. Skenario Moderat (Nilai Tengah P50) - Mode Baseline
3. Skenario Optimis (Batas Atas P90) - Mode Surplus Melimpah
"""

import pulp
from .config import REG, DIST, TARIF, ALPHA, DMAX_KM, IMPORT_COST


def solve_milp(supply, demand, tarif=TARIF, alpha=ALPHA, dmax=DMAX_KM, import_cost=IMPORT_COST):
    """
    Menyelesaikan satu instans masalah transportasi MILP untuk satu komoditas dan bulan.
    """
    K_sup = [k for k, v in supply.items() if v > 1e-6]
    K_dem = [k for k, v in demand.items() if v > 1e-6]

    if not K_dem:
        return dict(status="Optimal", obj=0.0, ship=[], imp={}, time=0.0)

    prob = pulp.LpProblem("Logistik_Pangan", pulp.LpMinimize)

    pairs = [(i, j) for i in K_sup for j in K_dem if DIST[(i, j)] <= dmax]
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=min(supply[i], demand[j]))
        for (i, j) in pairs
    }
    y = {(i, j): pulp.LpVariable(f"y_{i}_{j}", cat=pulp.LpBinary) for (i, j) in pairs}
    imp = {j: pulp.LpVariable(f"imp_{j}", lowBound=0) for j in K_dem}

    # Fungsi Objektif: Minimalkan Biaya Angkut Lokal + Penalti Impor Luar Provinsi
    prob += (
        pulp.lpSum(tarif * DIST[(i, j)] * x[(i, j)] for (i, j) in pairs)
        + pulp.lpSum(import_cost * imp[j] for j in K_dem)
    )

    # Kendala Kapasitas Pasokan
    for i in K_sup:
        out_e = [x[(i, j)] for j in K_dem if (i, j) in pairs]
        if out_e:
            prob += pulp.lpSum(out_e) <= supply[i]

    # Kendala Pemenuhan Permintaan (Alpha = 1.0)
    for j in K_dem:
        in_e = [x[(i, j)] for i in K_sup if (i, j) in pairs]
        prob += pulp.lpSum(in_e) + imp[j] >= alpha * demand[j]

    # Kendala Muatan Minimum Adaptif
    for (i, j) in pairs:
        L_min = min(20.0, supply[i], demand[j])
        U_max = min(supply[i], demand[j])
        prob += x[(i, j)] >= L_min * y[(i, j)]
        prob += x[(i, j)] <= U_max * y[(i, j)]

    # Eksekusi Solver HiGHS via API PuLP
    solver = pulp.HiGHS(msg=False)
    prob.solve(solver)

    status_str = pulp.LpStatus[prob.status]
    ship_res = [
        (i, j, DIST[(i, j)], pulp.value(x[(i, j)]))
        for (i, j) in pairs if pulp.value(x[(i, j)]) > 1e-4
    ]
    imp_res = {j: pulp.value(imp[j]) for j in K_dem if pulp.value(imp[j]) > 1e-4}
    obj_val = float(pulp.value(prob.objective)) if prob.status == 1 else None

    return dict(status=status_str, obj=obj_val, ship=ship_res, imp=imp_res)


def run_3scenario_milp(qf_df):
    """
    Menjalankan simulasi optimasi MILP untuk seluruh 3 skenario (P10, P50, P90).
    """
    scenarios = [
        ("P10_Pesimis", "p10", "robust"),
        ("P50_Moderat", "p50", "moderat"),
        ("P90_Optimis", "p90", "optimis"),
    ]

    ship_rows, sum_rows = [], []

    for sken_name, col_p, mode_lbl in scenarios:
        for (kom, tgl), sub in qf_df.groupby(["komoditas", "tanggal"]):
            surplus_val = sub[col_p].values
            kab_names = sub["kabupaten_kota"].values

            sup = {kab: max(float(s), 0.0) for kab, s in zip(kab_names, surplus_val)}
            dem = {kab: max(float(-s), 0.0) for kab, s in zip(kab_names, surplus_val)}

            res = solve_milp(sup, dem)
            if res is None:
                continue

            for i, j, dst, t in res["ship"]:
                ship_rows.append(dict(
                    komoditas=kom, tanggal=tgl, skenario=sken_name,
                    asal=i, tujuan=j, jarak_km=round(dst, 1),
                    kirim_ton=round(t, 2), biaya_rp=round(dst * TARIF * t)
                ))

            dikirim = sum(t for *_, t in res["ship"])
            impor = sum(res["imp"].values())
            dtot = sum(dem.values())

            sum_rows.append(dict(
                komoditas=kom, tanggal=tgl, skenario=sken_name, mode=mode_lbl,
                status=res["status"], supply_aman_ton=round(sum(sup.values())),
                demand_robust_ton=round(dtot), dikirim_lokal_ton=round(dikirim),
                impor_luar_ton=round(impor), biaya_total_rp=round(res["obj"]),
                n_rute=len(res["ship"]),
                pemenuhan_lokal_pct=round(100.0 * dikirim / dtot, 1) if dtot > 0 else 100.0
            ))

    import pandas as pd
    ship_df = pd.DataFrame(ship_rows)
    sum_df = pd.DataFrame(sum_rows)

    agg = (sum_df.groupby(["skenario", "komoditas"])
           .agg(n_bulan=("tanggal", "nunique"),
                defisit_robust_ton=("demand_robust_ton", "sum"),
                dialihkan_lokal_ton=("dikirim_lokal_ton", "sum"),
                impor_luar_ton=("impor_luar_ton", "sum"),
                biaya_miliar_rp=("biaya_total_rp", lambda s: round(s.sum() / 1e9, 2)))
           .reset_index())
    agg["tertutup_lokal_pct"] = (100.0 * agg["dialihkan_lokal_ton"] / agg["defisit_robust_ton"].clip(lower=1e-3)).round(1)

    return sum_df, ship_df, agg
