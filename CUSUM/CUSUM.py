#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 17:20:01 2026

@author: lunarolland
"""

import numpy as np
import pandas as pd

#CUSUM
def cusum_alarm_time(x: np.ndarray, mu0: float, sigma: float, k: float, h: float) -> int | None:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    S=0
    T=0
    for t in range(len(x)):
        z = (x[t] - mu0) / sigma
        S = max(0.0, S + z - k)
        T = max(0.0, T - z - k)
        if S > h or T > h:
            return t
    return None

#DATA
def generate_in_control(T: int, mu0: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=mu0, scale=sigma, size=T)

def generate_one_change_definition_B(T: int, tau: int, delta: float, mu0: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if not (0 < tau < T):
        raise ValueError("tau must satisfy 0 < tau < T")
    sign = rng.choice([-1.0, 1.0])
    mu1 = mu0 + sign * delta * sigma
    x = np.empty(T, dtype=float)
    x[:tau] = rng.normal(loc=mu0, scale=sigma, size=tau)
    x[tau:] = rng.normal(loc=mu1, scale=sigma, size=T - tau)
    return x

#ARL0
def simulate_arl0(n_trials: int, T_max: int, mu0: float, sigma: float, k: float, h: float, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    run_lengths = []
    censored_count = 0
    for _ in range(n_trials):
        x = generate_in_control(T=T_max, mu0=mu0, sigma=sigma, rng=rng)
        t_alarm = cusum_alarm_time(x=x, mu0=mu0, sigma=sigma, k=k, h=h)
        if t_alarm is None:
            censored_count += 1
            run_lengths.append(T_max)
        else:
            run_lengths.append(t_alarm + 1)
    run_lengths = np.array(run_lengths, dtype=float)
    return {
        "run_lengths": run_lengths,
        "ARL0": float(np.mean(run_lengths)),
        "SDRL0": float(np.std(run_lengths, ddof=1)),
        "censored_frac": censored_count / n_trials
    }

#ARL1
def simulate_arl1(n_trials: int, T_max: int, tau: int, delta: float, mu0: float, sigma: float, k: float, h: float, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    delays = []
    early_false = 0
    censored = 0
    for _ in range(n_trials):
        x = generate_one_change_definition_B(
            T=T_max, tau=tau, delta=delta, mu0=mu0, sigma=sigma, rng=rng
        )
        t_alarm = cusum_alarm_time(x=x, mu0=mu0, sigma=sigma, k=k, h=h)

        if t_alarm is None:
            censored += 1
            continue

        if t_alarm < tau:
            early_false += 1
        else:
            delays.append(t_alarm - tau)

    delays = np.array(delays, dtype=float)

    return {
        "delta": delta,
        "delays": delays,
        "early false": early_false,
        "ARL1_delay_cond": float(np.mean(delays)) if len(delays) else np.nan,
        "SDRL1_cond": float(np.std(delays, ddof=1)) if len(delays) > 1 else np.nan,
        "early_false_frac": early_false / n_trials,
        "censored_frac": censored / n_trials,
        "n_detected_after_change": int(len(delays))
    }

def calibrate_h_for_target_arl0(target_arl0: float, mu0: float, sigma: float, k: float, BOUNDS: tuple[float, float] = (1.0, 20.0), n_iter: int = 12, n_trials_per_step: int = 2000, T_max: int = 200000, seed: int = 123) -> float:
    rng_seed_base = seed
    h_low, h_high = BOUNDS
    while True:
        est = simulate_arl0(
            n_trials=n_trials_per_step,
            T_max=T_max,
            mu0=mu0,
            sigma=sigma,
            k=k,
            h=h_high,
            seed=rng_seed_base
        )["ARL0"]

        if est >= target_arl0:
            break

        h_high *= 1.5
        if h_high > 200:
            raise RuntimeError("h_high got too large; check setup or increase T_max.")
    
    for i in range(n_iter):
        h_mid = 0.5 * (h_low + h_high)

        est_mid = simulate_arl0(n_trials=n_trials_per_step, T_max=T_max, mu0=mu0, sigma=sigma, k=k, h=h_mid, seed=rng_seed_base + i + 1)["ARL0"]

        if est_mid < target_arl0:
            h_low = h_mid
        else:
            h_high = h_mid

    return 0.5 * (h_low + h_high)

#EXAMPLE
if __name__ == "__main__":
    mu0 = 0.0
    sigma = 1.0

    # design choices
    delta_stars = [0.5, 1.0, 2.0]          # determines k = delta_star/2
    target_arl0 = 500.0

    # simulation choices
    T_max = 300_000
    tau = 200
    deltas = [0.25, 0.5, 1.0, 2.0, 3.0]

    n_trials_arl0 = 10_000
    n_trials_arl1 = 10_000

    rows = []

    for delta_star in delta_stars:
        k = delta_star / 2.0

        h = calibrate_h_for_target_arl0(
            target_arl0=target_arl0,
            mu0=mu0,
            sigma=sigma,
            k=k
        )
        #ARL0
        arl0 = simulate_arl0(
            n_trials=n_trials_arl0,
            T_max=T_max,
            mu0=mu0,
            sigma=sigma,
            k=k,
            h=h,
            seed=999
        )

        rows.append({
            # parameters
            "case": "ARL0",
            "delta_star": delta_star,
            "k": k,
            "h": h,
            "mu0": mu0,
            "sigma": sigma,
            "T_max": T_max,
            "tau": np.nan,          
            "delta": np.nan,        #don't need it for ARL0

            #results
            "ARL": arl0["ARL0"],
            "SDRL": arl0["SDRL0"],
            "early_false_frac": np.nan,
            "censored_frac": arl0["censored_frac"],
            "n_detected_after_change": np.nan,
        })

        # ARL1
        for d in deltas:
            arl1 = simulate_arl1(
                n_trials=n_trials_arl1,
                T_max=T_max,
                tau=tau,
                delta=d,
                mu0=mu0,
                sigma=sigma,
                k=k,
                h=h,
                seed=2020 + int(d * 100)
            )

            rows.append({
                # parameters
                "case": "ARL1",
                "delta_star": delta_star,
                "k": k,
                "h": h,
                "mu0": mu0,
                "sigma": sigma,
                "T_max": T_max,
                "tau": tau,
                "delta": d,

                # results
                "ARL": arl1["ARL1_delay_cond"],
                "SDRL": arl1["SDRL1_cond"],
                "early_false_frac": arl1["early_false_frac"],
                "censored_frac": arl1["censored_frac"],
                "n_detected_after_change": arl1["n_detected_after_change"],
            })

    big_table = pd.DataFrame(rows)

    big_table = big_table.sort_values(
        by=["delta_star", "case", "delta"],
        ascending=[True, True, True],
        na_position="last"
    )

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.width", 180)
    print(big_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    big_table.to_csv("cusum_big_table.csv", index=False)

    latex = big_table.to_latex(
        index=False,
        float_format="%.3f",
        na_rep="",
        caption="CUSUM simulation summary: ARL0 and conditional ARL1 delays for multiple (delta_star, delta).",
        label="tab:cusum_big_table"
    )
    with open("cusum_big_table.tex", "w") as f:
        f.write(latex)

    print("\nSaved: cusum_big_table.csv and cusum_big_table.tex")

    
    
    
    
