import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import statsmodels.api as sm

def expit(x):
    return 1 / (1 + np.exp(-x))

def data_generation(n=2000, seed=42):
    np.random.seed(seed)
    Z = np.random.binomial(1, 0.3, n)
    A = np.random.binomial(1, expit(-3.5 + 3*Z), n)
    Y = A + Z + 3*A*Z + np.random.normal(0, 1, n)
    data = pd.DataFrame({'Z': Z, 'A': A, 'Y': Y})
    return data

def undersample_majority_class(data, seed=42):
    np.random.seed(seed)
    class_counts = data['A'].value_counts()
    maj = class_counts.idxmax()
    minr = class_counts.idxmin()

    # Undersample majority class
    majority_samples = data[data['A'] == maj]
    minority_samples = data[data['A'] == minr]

    n_minority = len(minority_samples)
    majority_keep = majority_samples.sample(n=n_minority, random_state=seed)

    # Combine to form the balanced dataset
    balanced = pd.concat([minority_samples, majority_keep])

    # Add R_s = 1 if the row is kept, 0 otherwise
    data = data.assign(R_s=data.index.isin(balanced.index).astype(int))

    return balanced.reset_index(drop=True), data


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

def compute_confidence_intervals(selected_df, full_df, a_name, y_name, z_names, r_name, num_bootstraps=200, alpha=0.05, seed=42):
    np.random.seed(seed)
    Ql = alpha / 2
    Qu = 1 - alpha / 2
    estimates = []

    for i in range(num_bootstraps):
        data_sampled = selected_df.sample(len(selected_df), replace=True)
        data_sampled.reset_index(drop=True, inplace=True)
        estimates.append(ipw(data_sampled, a_name, y_name, z_names))

    quantiles = np.quantile(estimates, q=[Ql, Qu])
    q_low = quantiles[0]
    q_up = quantiles[1]

    return q_low, q_up

def reweighted_confidence_intervals(full_df, a_name, y_name, z_names, r_name, num_resamples=200, alpha=0.05, seed=42):
    estimates = []
    for b in range(num_resamples):
        # Recreate R_s by undersampling (changes which rows are kept)
        _, full_with_R = undersample_majority_class(full_df, seed=seed + b)
        est = reweighted_ipw(full_with_R, a_name, y_name, z_names, r_name)
        estimates.append(est)

    q_low, q_up = np.quantile(estimates, [alpha/2, 1 - alpha/2])
    return q_low, q_up

data = data_generation(100000)
selected, full = undersample_majority_class(data)

print("Full data ACE ", ipw(full, a_name='A', y_name='Y', z_names=['Z']))
print()
print("Undersampled ACE ", ipw(selected, a_name='A', y_name='Y', z_names=['Z']))
print("CI for undersampled data", compute_confidence_intervals(selected, full, 'A', 'Y', ['Z'], 'R_s'))
print()
print("Reweighted ACE ", reweighted_ipw(full, a_name='A', y_name='Y', z_names=['Z'], r_name='R_s'))
print("CI for reweighted", reweighted_confidence_intervals(full, 'A', 'Y', ['Z'], 'R_s'))