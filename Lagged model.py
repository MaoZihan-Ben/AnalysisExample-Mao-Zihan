"""Lagged model:Endogeneity Handling Logic, Result Stability Verification"""
"""Since these codes were extracted from my work records, 
I imported all the libraries in every section at once :)."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings

independent_path = r"C:\Users\30444\Desktop\data\cleaned data\cleaned_data1.csv"
dependent_path = r"C:\Users\30444\Desktop\data\cleaned data\cleaned_data 2.csv"
control_path = r"C:\Users\30444\Desktop\data\cleaned data\cleaned_contorl.csv"  
df_independent = pd.read_csv(independent_path)
df_dependent = pd.read_csv(dependent_path)
df_control = pd.read_csv(control_path)  
stock_code_col = df_control.columns[0]
print(f"股票代码列：{stock_code_col}")
date_col = "Accper"
for df in [df_independent, df_dependent, df_control]:
    df[date_col] = pd.to_datetime(df[date_col])
merged = pd.merge(
    df_independent,
    df_dependent,
    on=[stock_code_col, date_col],
    how="inner"
)
merged = pd.merge(
    merged,
    df_control,
    on=[stock_code_col, date_col],
    how="inner"
)
merged = merged.sort_values(by=[stock_code_col, date_col])
core_vars = ["TTM", "ROCA", "ROE"]
control_vars = ["CR", "CaR", "DAR", "DER"]
print(f"控制变量列：{control_vars}")
y_var = "bes"
lagged_core = merged.groupby(stock_code_col)[core_vars].shift(1)
lagged_core.columns = [f"{col}_lag1" for col in core_vars] 
lagged_control = merged.groupby(stock_code_col)[control_vars].shift(1)
lagged_control.columns = [f"{col}_lag1" for col in control_vars] 
merged_lag = pd.concat([merged, lagged_core, lagged_control], axis=1)
merged_lag = merged_lag.dropna(subset=lagged_core.columns.tolist() + lagged_control.columns.tolist())
merged_lag["Year"] = merged_lag[date_col].dt.year
year_dummies = pd.get_dummies(merged_lag["Year"], prefix="Year")
industry_cols = [col for col in df_control.columns if "Industry" in col]
industry_dummies = merged_lag[industry_cols].select_dtypes(include=['number'])
print(f"行业固定效应列：{industry_dummies.columns.tolist()}")
X = pd.concat(
    [
        lagged_core,  
        lagged_control,  
        year_dummies,
        industry_dummies
    ],
    axis=1
)
X = X.select_dtypes(include=['number'])
X = sm.add_constant(X)
y = pd.to_numeric(merged_lag[y_var], errors="coerce").dropna()
X = X.loc[y.index] 
model_lag1 = sm.OLS(y, X).fit()
print("\n=== 滞后一期内生性检验结果（控制变量聚合显示）===")
print(model_lag1.summary())
print(f"\nR²: {model_lag1.rsquared:.4f} | 调整后R²: {model_lag1.rsquared_adj:.4f}")
print("\n【核心自变量（滞后一期）系数】")
for var in lagged_core.columns:
    coef = model_lag1.params[var]
    p_val = model_lag1.pvalues[var]
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"{var}: {coef:.4f} ({p_val:.4f}) {sig}")
print("\n【控制变量（滞后一期）整体影响】")
if control_vars:
    lagged_control_names = lagged_control.columns.tolist()
    control_coefs = [model_lag1.params[var] for var in lagged_control_names if var in model_lag1.params]
    control_pvals = [model_lag1.pvalues[var] for var in lagged_control_names if var in model_lag1.params]
    avg_coef = np.mean(control_coefs) if control_coefs else 0
    sig_count = sum(1 for p in control_pvals if p < 0.05) if control_pvals else 0
    
    print(f"控制变量平均系数：{avg_coef:.4f}")  
else:
    print("无控制变量")
formula_parts = [f"{model_lag1.params['const']:.4f}"] if "const" in model_lag1.params else []
formula_parts.extend([f"{model_lag1.params[var]:.4f}×{var}" for var in lagged_core.columns if var in model_lag1.params])
formula_parts.append("[控制变量（滞后一期）组合效应]") 
formula = f"{y_var}(t) = " + " + ".join(formula_parts) + " + [时间固定效应] + [行业固定效应] + [扰动项]"
print(f"\n回归公式（t表示当期，滞后变量为t-1期）:\n{formula}")
