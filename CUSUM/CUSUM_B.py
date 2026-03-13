#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


#CUSUM logic
def estimate_baseline(x: np.ndarray) -> tuple[float, float]:
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    if len(x) < 2:
        raise ValueError("Burn-in sample too small")
    mu_hat = float(np.mean(x))
    sigma_hat = float(np.std(x, ddof=1))
    if sigma_hat <= 0:
        raise ValueError("Estimated sigma must be > 0")
    return mu_hat, sigma_hat


def cusum_alarm_time_stream(
    rng: np.random.Generator,
    T_max: int,
    B: int,
    k: float,
    h: float,
    mu0: float,
    sigma: float,
    tau: int | None = None,
    delta: float | None = None,
) -> int | None:

    if not (0 < B < T_max):
        raise ValueError("Need 0 < B < T_max")
    if h <= 0:
        raise ValueError("h must be > 0")

    # Burn-in sample 
    xB = rng.normal(loc=mu0, scale=sigma, size=B)
    mu_hat, sigma_hat = estimate_baseline(xB)

    if tau is not None:
        if not (0 < B < tau < T_max):
            raise ValueError("Need 0 < B < tau < T_max so change happens after burn-in.")
        if delta is None:
            raise ValueError("If tau is set, delta must be set.")
        sign = rng.choice([-1.0, 1.0])
        mu1 = mu0 + sign * delta * sigma
    else:
        mu1 = mu0

    S = 0.0
    T = 0.0

    for t in range(B, T_max):
        mean = mu0 if (tau is None or t < tau) else mu1
        x_t = rng.normal(loc=mean, scale=sigma)

        z = (x_t - mu_hat) / sigma_hat
        S = max(0.0, S + z - k)
        T = max(0.0, T - z - k)

        if S > h or T > h:
            return t

    return None


#SIMULATOR
def simulate_arl0_burnin(
    n_trials: int,
    T_max: int,
    B: int,
    mu0: float,
    sigma: float,
    k: float,
    h: float,
    seed: int,
    print_every: int = 1000,
) -> dict:
    rng = np.random.default_rng(seed)
    rl = np.empty(n_trials, dtype=float)
    censored = 0

    t0 = time.time()
    for i in range(n_trials):
        t_alarm = cusum_alarm_time_stream(
            rng=rng, T_max=T_max, B=B, k=k, h=h, mu0=mu0, sigma=sigma
        )

        if t_alarm is None:
            censored += 1
            rl[i] = T_max - B
        else:
            rl[i] = (t_alarm - B) + 1

        if (i + 1) % print_every == 0 or (i + 1) == n_trials:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else np.nan
            print(
                f"[ARL0] {i+1}/{n_trials} | {elapsed:.1f}s | {rate:.2f} trials/s | "
                f"censored={censored/(i+1):.3f}"
            )

    q10, q50, q90 = np.quantile(rl, [0.10, 0.50, 0.90])
    mean_rl = float(np.mean(rl))
    sd_rl = float(np.std(rl, ddof=1)) if n_trials > 1 else np.nan

    return {
        "run_lengths": rl,
        "ARL0": mean_rl,
        "SDRL0": sd_rl,
        "MEDRL0": float(q50),
        "Q10RL0": float(q10),
        "Q90RL0": float(q90),
        "SD_over_mean": (sd_rl / mean_rl) if mean_rl > 0 and np.isfinite(sd_rl) else np.nan,
        "censored_frac": censored / n_trials,
    }


def simulate_arl1_burnin(
    n_trials: int,
    T_max: int,
    B: int,
    tau: int,
    delta: float,
    mu0: float,
    sigma: float,
    k: float,
    h: float,
    seed: int,
    print_every: int = 1000,
) -> dict:
    rng = np.random.default_rng(seed)

    delays_list = []
    early_false = 0
    censored = 0

    if not (0 < B < tau < T_max):
        raise ValueError("Need 0 < B < tau < T_max so the change happens after burn-in.")

    t0 = time.time()
    for i in range(n_trials):
        t_alarm = cusum_alarm_time_stream(
            rng=rng, T_max=T_max, B=B, k=k, h=h, mu0=mu0, sigma=sigma, tau=tau, delta=delta
        )

        if t_alarm is None:
            censored += 1
        elif t_alarm < tau:
            early_false += 1
        else:
            delays_list.append(t_alarm - tau)

        if (i + 1) % print_every == 0 or (i + 1) == n_trials:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else np.nan
            print(
                f"[ARL1 δ={delta}] {i+1}/{n_trials} | {elapsed:.1f}s | {rate:.2f} trials/s | "
                f"early={early_false/(i+1):.3f} | cens={censored/(i+1):.3f} | "
                f"after_change_detect={len(delays_list)}"
            )

    delays = np.array(delays_list, dtype=float)

    if len(delays) > 0:
        q10, q50, q90 = np.quantile(delays, [0.10, 0.50, 0.90])
        mean_d = float(np.mean(delays))
        sd_d = float(np.std(delays, ddof=1)) if len(delays) > 1 else np.nan
        sd_over_mean = (sd_d / mean_d) if mean_d > 0 and np.isfinite(sd_d) else np.nan
    else:
        q10 = q50 = q90 = np.nan
        mean_d = sd_d = sd_over_mean = np.nan

    return {
        "delays": delays,
        "ARL1_delay_cond": mean_d,
        "SDRL1_cond": sd_d,
        "MED_delay_cond": float(q50) if np.isfinite(q50) else np.nan,
        "Q10_delay": float(q10) if np.isfinite(q10) else np.nan,
        "Q90_delay": float(q90) if np.isfinite(q90) else np.nan,
        "SD_over_mean": sd_over_mean,
        "early_false_frac": early_false / n_trials,
        "censored_frac": censored / n_trials,
        "n_detected_after_change": int(len(delays)),
    }


#TABLES
def make_parameter_table(
    designs: list[dict],
    B_grid: list[int],
    tau_grid: list[int],
    deltas: list[float],
    mu0: float,
    sigma: float,
    T_max: int,
    n_trials_arl0: int,
    n_trials_arl1: int,
    print_every: int,
) -> pd.DataFrame:
    rows = []

    # ARL0 
    for dsg in designs:
        for B in B_grid:
            rows.append({
                "case": "ARL0",
                "design": dsg["label"],
                "delta_star_design": dsg["delta_star"],
                "mu0": mu0,
                "sigma": sigma,
                "T_max": T_max,
                "B": B,
                "tau": np.nan,
                "delta": np.nan,
                "k": dsg["k"],
                "h": dsg["h"],
                "n_trials": n_trials_arl0,
                "print_every": print_every,
                "definition": "In-control. Burn-in length B used to estimate baseline. Phase II RL = t_alarm - B + 1."
            })

    # ARL1 
    for dsg in designs:
        for B in B_grid:
            for tau in tau_grid:
                if not (B < tau < T_max):
                    continue
                for delta in deltas:
                    rows.append({
                        "case": "ARL1",
                        "design": dsg["label"],
                        "delta_star_design": dsg["delta_star"],
                        "mu0": mu0,
                        "sigma": sigma,
                        "T_max": T_max,
                        "B": B,
                        "tau": tau,
                        "delta": delta,
                        "k": dsg["k"],
                        "h": dsg["h"],
                        "n_trials": n_trials_arl1,
                        "print_every": print_every,
                        "definition": "One change at tau: mean shifts to mu0 ± delta*sigma (random sign). "
                                     "Report conditional delay = t_alarm - tau (only when t_alarm >= tau)."
                    })

    return pd.DataFrame(rows)


def describe_global_parameters(mu0, sigma, T_max, n_trials_arl0, n_trials_arl1, print_every) -> pd.DataFrame:
    """A simple 'metadata' table you can paste in thesis / appendix."""
    return pd.DataFrame([
        {"name": "mu0", "value": mu0, "meaning": "In-control process mean"},
        {"name": "sigma", "value": sigma, "meaning": "Process standard deviation"},
        {"name": "T_max", "value": T_max, "meaning": "Max simulated time horizon per trial"},
        {"name": "n_trials_arl0", "value": n_trials_arl0, "meaning": "Number of Monte Carlo trials for ARL0"},
        {"name": "n_trials_arl1", "value": n_trials_arl1, "meaning": "Number of Monte Carlo trials for ARL1"},
        {"name": "print_every", "value": print_every, "meaning": "Progress print frequency (trials)"},
    ])


#HISTOGRAM
def plot_hist(values: np.ndarray, title: str, xlabel: str, bins: int = 70) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True)


def plot_log_hist(values: np.ndarray, title: str, xlabel: str, bins: int = 70) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(np.log1p(values), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True)


#MAIN
if __name__ == "__main__":
    mu0 = 0.0
    sigma = 1.0

    T_max = 200_000

    n_trials_arl0 = 5_000
    n_trials_arl1 = 5_000

    print_every = 1000

    B_grid = [20, 50, 100, 200]
    tau_grid = [200, 400, 800, 1500]
    deltas = [0.5, 2.0]

    designs = [
        {"label": "best_for_delta_0.5", "delta_star": 0.5, "k": 0.25, "h": 8.633},
        {"label": "best_for_delta_2.0", "delta_star": 2.0, "k": 1.00, "h": 2.644},
    ]

    param_table = make_parameter_table(
        designs=designs,
        B_grid=B_grid,
        tau_grid=tau_grid,
        deltas=deltas,
        mu0=mu0,
        sigma=sigma,
        T_max=T_max,
        n_trials_arl0=n_trials_arl0,
        n_trials_arl1=n_trials_arl1,
        print_every=print_every,
    )
    param_table.to_csv("cusum_parameter_table.csv", index=False)
    print("Saved: cusum_parameter_table.csv")

    global_table = describe_global_parameters(mu0, sigma, T_max, n_trials_arl0, n_trials_arl1, print_every)
    global_table.to_csv("cusum_global_parameters.csv", index=False)
    print("Saved: cusum_global_parameters.csv")

    results_rows = []

    FINAL_B_FOR_HISTS = 50
    FINAL_TAU_FOR_HISTS = 400
    FINAL_DELTAS_FOR_HISTS = (0.5, 2.0)

    RUN_DIAGNOSTICS = True
    ARL_METRIC = "mean"  
    EXTEND_DIAG_DESIGNS = True  

    
    arl0_data_for_diagnostics = [] 

    for dsg in designs:
        k, h = dsg["k"], dsg["h"]
        print(f"\n==========================================")
        print(f"DESIGN: {dsg['label']}  (k={k}, h={h})")
        print(f"==========================================")

        # ARL0
        for B in B_grid:
            print(f"\n--- ARL0: B={B} ---")
            out0 = simulate_arl0_burnin(
                n_trials=n_trials_arl0,
                T_max=T_max,
                B=B,
                mu0=mu0,
                sigma=sigma,
                k=k,
                h=h,
                seed=1000 + B + int(100 * dsg["delta_star"]),
                print_every=print_every,
            )

            results_rows.append({
                "case": "ARL0",
                "design": dsg["label"],
                "delta_star_design": dsg["delta_star"],
                "k": k, "h": h,
                "mu0": mu0, "sigma": sigma,
                "T_max": T_max,
                "B": B, "tau": np.nan, "delta": np.nan,
                "n_trials": n_trials_arl0,
                "ARL": out0["ARL0"],
                "SDRL": out0["SDRL0"],
                "MED": out0["MEDRL0"],
                "Q10": out0["Q10RL0"],
                "Q90": out0["Q90RL0"],
                "SD_over_mean": out0["SD_over_mean"],
                "early_false_frac": np.nan,
                "censored_frac": out0["censored_frac"],
            })

            if RUN_DIAGNOSTICS:
                arl0_data_for_diagnostics.append({
                    "B": B,
                    "k": k,
                    "h": h,
                    "design": dsg["label"],
                    "run_lengths": out0["run_lengths"].copy(),  # Store copy of run lengths
                    "ARL0_mean": out0["ARL0"],
                    "ARL0_median": out0["MEDRL0"],
                })

            if B == FINAL_B_FOR_HISTS:
                plot_hist(
                    out0["run_lengths"],
                    title=f"ARL0 Run Length Histogram — {dsg['label']} (B={FINAL_B_FOR_HISTS})",
                    xlabel="Phase II run length (t_alarm - B + 1)",
                    bins=70
                )
                plot_log_hist(
                    out0["run_lengths"],
                    title=f"ARL0 log(1+RL) Histogram — {dsg['label']} (B={FINAL_B_FOR_HISTS})",
                    xlabel="log(1 + run length)",
                    bins=70
                )

        # ARL1
        for B in B_grid:
            for tau in tau_grid:
                if not (B < tau < T_max):
                    continue
                for delta in deltas:
                    print(f"\n--- ARL1: B={B}, tau={tau}, delta={delta} ---")
                    out1 = simulate_arl1_burnin(
                        n_trials=n_trials_arl1,
                        T_max=T_max,
                        B=B,
                        tau=tau,
                        delta=delta,
                        mu0=mu0,
                        sigma=sigma,
                        k=k,
                        h=h,
                        seed=2000 + 10 * B + tau + int(delta * 100) + int(10 * dsg["delta_star"]),
                        print_every=print_every,
                    )

                    results_rows.append({
                        "case": "ARL1",
                        "design": dsg["label"],
                        "delta_star_design": dsg["delta_star"],
                        "k": k, "h": h,
                        "mu0": mu0, "sigma": sigma,
                        "T_max": T_max,
                        "B": B, "tau": tau, "delta": delta,
                        "n_trials": n_trials_arl1,
                        "ARL": out1["ARL1_delay_cond"],
                        "SDRL": out1["SDRL1_cond"],
                        "MED": out1["MED_delay_cond"],
                        "Q10": out1["Q10_delay"],
                        "Q90": out1["Q90_delay"],
                        "SD_over_mean": out1["SD_over_mean"],
                        "early_false_frac": out1["early_false_frac"],
                        "censored_frac": out1["censored_frac"],
                        "n_detected_after_change": out1["n_detected_after_change"],
                    })

         
                    if (B == FINAL_B_FOR_HISTS) and (tau == FINAL_TAU_FOR_HISTS) and (delta in FINAL_DELTAS_FOR_HISTS):
                        if len(out1["delays"]) > 0:
                            plot_hist(
                                out1["delays"],
                                title=f"ARL1 Delay Histogram — {dsg['label']} (B={B}, tau={tau}, delta={delta})",
                                xlabel="Detection delay (t_alarm - tau)",
                                bins=70
                            )
                        else:
                            print("No post-change detections for this setting; skipping delay histogram.")

    # RESULTS
    results = pd.DataFrame(results_rows).sort_values(
        by=["design", "case", "B", "tau", "delta"],
        na_position="last"
    )

    results.to_csv("cusum_results_table.csv", index=False)
    print("\nSaved: cusum_results_table.csv")

    latex = results.to_latex(
        index=False,
        float_format="%.3f",
        na_rep="",
        caption="CUSUM burn-in simulation results: ARL0 run lengths and conditional ARL1 detection delays.",
        label="tab:cusum_burnin_results"
    )
    with open("cusum_results_table.tex", "w") as f:
        f.write(latex)
    print("Saved: cusum_results_table.tex")

    # Preview in console
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 120)
    print("\n===== RESULTS TABLE PREVIEW =====")
    print(results.head(60).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    plt.show()

#DIAGNOSTIC
    if RUN_DIAGNOSTICS and len(arl0_data_for_diagnostics) > 0:
        print("\n" + "=" * 70)
        print("RUNNING DIAGNOSTICS")
        print("=" * 70)

        def get_arl0_value(data_dict):
            if ARL_METRIC == "median":
                return data_dict["ARL0_median"]
            else:  # default to mean
                return data_dict["ARL0_mean"]

#PLOTs
         #1. QQ-PLOT
        baseline_B = FINAL_B_FOR_HISTS if any(d["B"] == FINAL_B_FOR_HISTS for d in arl0_data_for_diagnostics) else B_grid[0]
        baseline_data = [d for d in arl0_data_for_diagnostics if d["B"] == baseline_B]
        
        if len(baseline_data) > 0:
            qq_data = baseline_data[0]
            run_lengths = qq_data["run_lengths"]
            k_qq, h_qq = qq_data["k"], qq_data["h"]
            B_qq = qq_data["B"]
            

            log_rl = np.log(run_lengths + 1)  
            
            plt.figure(figsize=(8, 6))
            stats.probplot(log_rl, dist="norm", plot=plt)
            plt.title(f"Q-Q Plot: log(1+RL) vs Normal\nB={B_qq}, k={k_qq}, h={h_qq:.3f}", fontsize=12, fontweight='bold')
            plt.xlabel("Theoretical quantiles (Normal)")
            plt.ylabel("Observed quantiles (log(1+run_length))")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            print(f"Q-Q plot created for B={B_qq}, k={k_qq}, h={h_qq:.3f}")


            raw_rl = run_lengths  
            
          
            log_rl_for_fit = np.log(raw_rl[raw_rl > 0] + 1) 
            if len(log_rl_for_fit) > 0:
                mu_ln = np.mean(log_rl_for_fit)
                sigma_ln = np.std(log_rl_for_fit, ddof=1)
                s_ln = sigma_ln
                scale_ln = np.exp(mu_ln)
            else:
                s_ln, scale_ln = 1.0, 1.0
                mu_ln, sigma_ln = 0.0, 1.0

            mean_rl = np.mean(raw_rl)
            lambda_exp = 1.0 / mean_rl if mean_rl > 0 else 1.0
            scale_exp = 1.0 / lambda_exp 
            

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Q-Q plot: Raw run-length vs Log-Normal
            stats.probplot(raw_rl, dist=stats.lognorm, sparams=(s_ln, 0, scale_ln), plot=ax1)
            ax1.set_title(f"Q-Q Plot: Raw RL vs Log-Normal\n(s={s_ln:.3f}, scale={scale_ln:.3f})\nB={B_qq}, k={k_qq}, h={h_qq:.3f}", 
                         fontsize=11, fontweight='bold')
            ax1.set_xlabel("Theoretical quantiles (Log-Normal)")
            ax1.set_ylabel("Observed quantiles (run_length)")
            ax1.grid(True, alpha=0.3)
            
            # Q-Q plot: Raw run-length vs Exponential
            stats.probplot(raw_rl, dist=stats.expon, sparams=(0, scale_exp), plot=ax2)
            ax2.set_title(f"Q-Q Plot: Raw RL vs Exponential\n(λ={lambda_exp:.6f}, scale={scale_exp:.3f})\nB={B_qq}, k={k_qq}, h={h_qq:.3f}", 
                         fontsize=11, fontweight='bold')
            ax2.set_xlabel("Theoretical quantiles (Exponential)")
            ax2.set_ylabel("Observed quantiles (run_length)")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            print(f"Additional Q-Q plots created: Log-Normal and Exponential for B={B_qq}, k={k_qq}, h={h_qq:.3f}")

            # Q-Q vs Normal for finite samples of i.i.d. N(0,1)
   
            rng_theory = np.random.default_rng(12345)
            sample_sizes = [50, 100, 200, 500]
            fig_norm, axes_norm = plt.subplots(2, 2, figsize=(12, 8))
            axes_norm = axes_norm.flatten()

            for i, n_samp in enumerate(sample_sizes):
                ax = axes_norm[i]
                data = rng_theory.normal(loc=0.0, scale=1.0, size=n_samp)
                stats.probplot(data, dist="norm", plot=ax)
                ax.set_title(f"N(0,1) sample, n={n_samp}", fontsize=10)
                ax.set_xlabel("Theoretical quantiles (Normal)")
                ax.set_ylabel("Sample quantiles")
                ax.grid(True, alpha=0.3)

            for j in range(len(sample_sizes), len(axes_norm)):
                fig_norm.delaxes(axes_norm[j])

            fig_norm.suptitle(
                "Q-Q plots for N(0,1) samples of different sizes",
                fontsize=12,
                fontweight="bold",
            )
            fig_norm.tight_layout(rect=(0, 0, 1, 0.96))
            plt.show()
            print("Normal-theory Q-Q sample-size demonstration created.")

            
            # Q-Q vs Normal for log(1+RL) using random subsamples
          
            subsizes = [50, 100, 500, 1000]
            subsizes = [n_sub for n_sub in subsizes if n_sub <= len(log_rl)]
            if len(subsizes) > 0:
                rng_sub = np.random.default_rng(23456)
                fig_rl, axes_rl = plt.subplots(2, 2, figsize=(12, 8))
                axes_rl = axes_rl.flatten()

                for i, n_sub in enumerate(subsizes):
                    idx = rng_sub.choice(len(log_rl), size=n_sub, replace=False)
                    sample = log_rl[idx]
                    ax = axes_rl[i]
                    stats.probplot(sample, dist="norm", plot=ax)
                    ax.set_title(f"log(1+RL) subsample, n={n_sub}", fontsize=10)
                    ax.set_xlabel("Theoretical quantiles (Normal)")
                    ax.set_ylabel("Sample quantiles")
                    ax.grid(True, alpha=0.3)

                for j in range(len(subsizes), len(axes_rl)):
                    fig_rl.delaxes(axes_rl[j])

                fig_rl.suptitle(
                    "Q-Q vs Normal for log(1+RL) subsamples (baseline RL)",
                    fontsize=12,
                    fontweight="bold",
                )
                fig_rl.tight_layout(rect=(0, 0, 1, 0.96))
                plt.show()
                print("RL subsample Q-Q sample-size demonstration created.")

           
            if EXTEND_DIAG_DESIGNS:
                print("\n[Diagnostics] Simulating additional ARL0 designs for richer boxplots...")
           
                k_multipliers = [0.8, 1.2]
                h_multipliers = [0.9, 1.1]
                diag_seed_base = 900000

                for dsg in designs:
                    base_k = dsg["k"]
                    base_h = dsg["h"]
                    base_label = dsg["label"]
                    delta_star = dsg["delta_star"]

                    for i_mk, mk in enumerate(k_multipliers):
                        for i_mh, mh in enumerate(h_multipliers):
                            k_new = base_k * mk
                            h_new = base_h * mh
                            label_new = f"{base_label}_k×{mk:.1f}_h×{mh:.1f}"

                            for B in B_grid:
                                seed_diag = (
                                    diag_seed_base
                                    + int(1000 * delta_star)
                                    + int(10 * B)
                                    + i_mk * 10
                                    + i_mh
                                )
                                out_diag = simulate_arl0_burnin(
                                    n_trials=n_trials_arl0,
                                    T_max=T_max,
                                    B=B,
                                    mu0=mu0,
                                    sigma=sigma,
                                    k=k_new,
                                    h=h_new,
                                    seed=seed_diag,
                                    print_every=print_every,
                                )

                                arl0_data_for_diagnostics.append({
                                    "B": B,
                                    "k": k_new,
                                    "h": h_new,
                                    "design": label_new,
                                    "run_lengths": out_diag["run_lengths"].copy(),
                                    "ARL0_mean": out_diag["ARL0"],
                                    "ARL0_median": out_diag["MEDRL0"],
                                })

      #LINE PLOT (burn in vs ALR0)
        kh_groups = {}
        for d in arl0_data_for_diagnostics:
            kh_key = (d["k"], d["h"], d["design"])
            if kh_key not in kh_groups:
                kh_groups[kh_key] = []
            kh_groups[kh_key].append(d)
        
        for kh_key in kh_groups:
            kh_groups[kh_key].sort(key=lambda x: x["B"])
        
        plt.figure(figsize=(10, 6))
        for (k_val, h_val, design_label), group_data in kh_groups.items():
            B_vals = [d["B"] for d in group_data]
            ARL0_vals = [get_arl0_value(d) for d in group_data]
            plt.plot(B_vals, ARL0_vals, marker='o', linewidth=1.5, alpha=0.7, 
                    label=f"k={k_val}, h={h_val:.3f} ({design_label})")
        
        plt.xlabel("Burn-in (B)", fontsize=11)
        plt.ylabel(f"ARL0 ({ARL_METRIC})", fontsize=11)
        plt.title(f"ARL0 vs Burn-in by (k,h) Parameter Pairs\n(ARL metric: {ARL_METRIC})", 
                 fontsize=12, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print(f"Line plot created: ARL0 ({ARL_METRIC}) vs burn-in")

      #BOX PLOT (burn in vs ARL0)
        B_groups = {}
        for d in arl0_data_for_diagnostics:
            B_val = d["B"]
            if B_val not in B_groups:
                B_groups[B_val] = []
            B_groups[B_val].append(get_arl0_value(d))
        
        sorted_B_vals = sorted(B_groups.keys())
        boxplot_data = [B_groups[B] for B in sorted_B_vals]
        
        plt.figure(figsize=(10, 6))
        bp = plt.boxplot(boxplot_data, labels=[str(B) for B in sorted_B_vals], 
                        patch_artist=True, showmeans=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        plt.xlabel("Burn-in (B)", fontsize=11)
        plt.ylabel(f"ARL0 ({ARL_METRIC})", fontsize=11)
        plt.title(f"ARL0 Distribution vs Burn-in (across all (k,h) pairs)\n(ARL metric: {ARL_METRIC})", 
                 fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
        print(f"Boxplot created: ARL0 ({ARL_METRIC}) distribution vs burn-in")

        print("\n" + "=" * 70)
        print("DIAGNOSTICS COMPLETE")
        print("=" * 70)
