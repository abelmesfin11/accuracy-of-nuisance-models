import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import statsmodels.api as sm

def expit(x):
  return 1 / (1 + np.exp(-x))

def generate_weighted_dataset(n=2000, seed=42, base_df=None):
    """
    If base_df is provided, reuse its Z, A, Y and only re-run the replication
    with a fresh seed (for rIPW CIs). Returns (base_with_Rs, expanded_replica).
    """
    np.random.seed(seed)

    if base_df is None:
        Z = np.random.binomial(1, 0.3, n)
        A = np.random.binomial(1, expit(-1.5 + 2*Z), n)
        Y = A + Z + 3*A*Z + np.random.normal(0, 1, n)
        df = pd.DataFrame({"Z": Z, "A": A, "Y": Y})
    else:
        df = base_df.copy()

    # Class weights
    n1, n0 = (df["A"] == 1).sum(), (df["A"] == 0).sum()
    n = len(df)
    w1 = n / (2 * n1) if n1 > 0 else 0.0
    w0 = n / (2 * n0) if n0 > 0 else 0.0
    df["W"] = np.where(df["A"] == 1, w1, w0)

    # Integer and fractional parts
    df["int_part"] = np.floor(df["W"]).astype(int)
    df["frac_prob"] = df["W"] - df["int_part"]

    # Fractional extra replicate
    df["_frac_draw"] = np.random.binomial(1, df["frac_prob"])

    # Total replicate count per original row
    repeat_counts = df["int_part"] + df["_frac_draw"]

    # Selection indicator: included >= 1 time
    df["R_S"] = (repeat_counts > 0).astype(int)

    # Expanded (replicated) dataset for naive class-weighted IPW
    expanded_df = df.loc[df.index.repeat(repeat_counts)].reset_index(drop=True)

    return df.drop(columns=["_frac_draw"]), expanded_df


def ipw(data: pd.DataFrame, a_name: str, y_name: str, z_names: list[str]) -> float:
    """
    Perform IPW for a given treatment A and outcome Y using
    the covariates in Z
    """
    z_formula = " + ".join(z_names)
    regression_formula = f"{a_name} ~ {z_formula}"
    model = sm.GLM.from_formula(formula=regression_formula, family=sm.families.Binomial(),data=data).fit()

    p_a1_z = model.predict(data)
    p_a0_z = 1 - p_a1_z

    e_ya1 = np.mean((data[a_name] / p_a1_z) * data[y_name])
    e_ya0 = np.mean(((1 - data[a_name]) / p_a0_z) * data[y_name])

    return round(e_ya1 - e_ya0, 3)

def reweighted_ipw(full_df, a_name, y_name, z_names, r_name):
    selected_df = full_df[full_df[r_name] == 1]
    f1 = f"{r_name} ~ " + " + ".join(z_names)
    rs_model = sm.GLM.from_formula(formula=f1, family=sm.families.Binomial(), data=full_df).fit()
    p_rs1_z = rs_model.predict(full_df)

    f2 = f"{a_name} ~ " + " + ".join(z_names)
    a_model = sm.GLM.from_formula(formula=f2,family=sm.families.Binomial(), data=selected_df).fit()
    p_a1_z = a_model.predict(full_df)
    p_a0_z = 1 - p_a1_z

    R = full_df[r_name]
    A = full_df[a_name]
    Y = full_df[y_name]

    w1 = 1 / (p_a1_z * p_rs1_z)
    w0 = 1 / (p_a0_z * p_rs1_z)

    e_ya1 = np.mean(w1 * R * A * Y)
    e_ya0 = np.mean(w0 * R * (1 - A) * Y)

    return round(e_ya1 - e_ya0, 3)

def compute_confidence_intervals(base, expanded, a_name, y_name, z_names, r_name, num_bootstraps=200, alpha=0.05, seed=42):
    np.random.seed(seed)
    Ql = alpha / 2
    Qu = 1 - alpha / 2
    estimates = []

    for i in range(num_bootstraps):
        data_sampled = expanded.sample(len(expanded), replace=True)
        data_sampled.reset_index(drop=True, inplace=True)
        estimates.append(ipw(data_sampled, a_name, y_name, z_names))

    # calculate the quantiles
    quantiles = np.quantile(estimates, q=[Ql, Qu])
    q_low = quantiles[0]
    q_up = quantiles[1]

    return q_low, q_up

def reweighted_confidence_intervals(base_df, a_name, y_name, z_names, r_name,
                                    num_resamples=200, alpha=0.05, seed=42):
    estimates = []
    for b in range(num_resamples):
        # IMPORTANT: reuse the same base sample, only re-run replication randomness
        base_b, _ = generate_weighted_dataset(seed=seed + b, base_df=base_df)
        est = reweighted_ipw(base_b, a_name, y_name, z_names, r_name)
        estimates.append(est)
    q_low, q_up = np.quantile(estimates, [alpha/2, 1 - alpha/2])
    return float(q_low), float(q_up)


base, expanded = generate_weighted_dataset(n=2000, seed=42)
print("Full data ACE ", ipw(base, a_name='A', y_name='Y', z_names=['Z']))
print("ACE using class weighting ", ipw(expanded, a_name='A', y_name='Y', z_names=['Z']))
print("CI for class weighting ", compute_confidence_intervals(base, expanded, 'A', 'Y', ['Z'], 'R_s'))
print()
print("ACE using reweighted IPW ", reweighted_ipw(base, a_name='A', y_name='Y', z_names=['Z'], r_name='R_S'))
print("CI for reweighted class-weighting ", reweighted_confidence_intervals(base,'A', 'Y', ['Z'], 'R_S'))