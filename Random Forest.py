"""Random Forest(Non-linear relationship &Characteristic coefficient measurement)"""
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
warnings.filterwarnings('ignore')

independent_path = r"C:\Users\30444\Desktop\data\cleaned data\cleaned_data1.csv"
control_path = r"C:\Users\30444\Desktop\data\cleaned data\cleaned_contorl.csv"
dependent_path = r"C:\Users\30444\Desktop\data\Robustness test\Earnings per share after deducting non recurring gains and losses.csv"

stock_code_col = "Stkcd"
date_col = "Accper"
target_col = "BES after deducting"
core_vars = ["TTM", "ROCA", "ROE"]
control_vars = ["CR", "CaR", "DAR", "DER"]

df_indep = pd.read_csv(independent_path, encoding='utf-8')
df_control = pd.read_csv(control_path, encoding='utf-8')
df_target = pd.read_csv(dependent_path, encoding='utf-8')

for df in [df_indep, df_control, df_target]:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

merged = pd.merge(df_indep, df_target, on=[stock_code_col, date_col], how='inner')
merged = pd.merge(merged, df_control, on=[stock_code_col, date_col], how='inner')

merged['Year'] = merged[date_col].dt.year.astype(str)

industry_col = merged.columns[0]
merged['Industry'] = merged[industry_col].astype(str)

feature_cols = core_vars + control_vars + ['Year', 'Industry']

X = merged[feature_cols].copy()
y = merged[target_col].copy()

for col in X.columns:
    if X[col].dtype != 'object':
        X[col].fillna(X[col].mean(), inplace=True)

X['Industry'] = X['Industry'].astype('category')
X['Year'] = X['Year'].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(f"测试集R²得分：{r2_score(y_test, y_pred):.4f}")

feature_importance = pd.DataFrame({
    '特征': feature_cols,
    '重要性': rf.feature_importances_
}).sort_values(by='重要性', ascending=False)

print("\n特征重要性排序：")
print(feature_importance.round(4))

core_importance = feature_importance[feature_importance['特征'].isin(core_vars)]
print("\n核心自变量重要性：")
print(core_importance.round(4))

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['特征'], feature_importance['重要性'])
plt.xlabel('特征重要性（越高表示对模型贡献越大）')
plt.title('随机森林特征重要性（含控制变量与固定效应）')
plt.gca().invert_yaxis()

for i, (feature, imp) in enumerate(zip(feature_importance['特征'], feature_importance['重要性'])):
    if feature in core_vars:
        plt.text(imp, i, f'  核心指标', color='red', fontweight='bold')

plt.tight_layout()
plt.show()
