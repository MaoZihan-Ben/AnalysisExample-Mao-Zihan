
"""Dear Sir or Madam, 
please note that the following code includes the baseline regression, 
lagged model, dependent variable replacement, Propensity Score Matching (PSM), 
Random Forest, and Multilayer Perceptron from my analytical work. 
If you wish to run it on your device, 
please kindly place the dataset on your computer's desktop 
and ensure that the dataset file path is: "C:Users\30444\Desktop\data." 
Thank you.

Respectfully, 
Mao Zihan
"""
#%%
"""data pre-processing"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def handle_missing_values(df, filename):
    """处理缺失值并生成缺失值热图"""
    original_rows = len(df)
    df_dropped = df.dropna() 
    missing_rows = original_rows - len(df_dropped)
    
    print(f"\n===== 处理 {filename} 缺失值 =====")
    print("缺失值统计:")
    print(df.isnull().sum())
    print(f"已删除 {missing_rows} 行包含缺失值的数据")
    print(f"清洗后剩余 {len(df_dropped)} 行数据")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title(f'{filename} Missing Value Distribution')
    plt.tight_layout()
    plt.show() 
    return df_dropped
def handle_duplicates(df, filename):
    """处理重复行"""
    original_rows = len(df)
    df_clean = df.drop_duplicates()
    duplicate_rows = original_rows - len(df_clean)
    
    print(f"\n===== 处理 {filename} 重复值 =====")
    if duplicate_rows > 0:
        print(f"检测到 {duplicate_rows} 行重复数据，已移除")
    else:
        print("未检测到重复数据")
    print(f"去重后剩余 {len(df_clean)} 行数据")
    return df_clean

def process_data_folder(input_folder, output_folder):
    """处理文件夹中所有CSV文件"""
    os.makedirs(output_folder, exist_ok=True)
    print(f"输出文件夹路径: {output_folder}\n")
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv") and not filename.startswith("cleaned_"):
            file_path = os.path.join(input_folder, filename)
            
            try:

                df = pd.read_csv(file_path)
                print(f"\n===== 开始处理文件: {filename} =====")
                print(f"原始数据行数: {len(df)}")
                df = handle_missing_values(df, filename)
                df = handle_duplicates(df, filename)
                cleaned_filename = f"cleaned_{filename}"
                cleaned_path = os.path.join(output_folder, cleaned_filename)
                df.to_csv(cleaned_path, index=False)
                print(f"\n清洗完成！文件已保存至: {cleaned_path}")
                print("----------------------------------------")
                
            except Exception as e:
                print(f"\n处理 {filename} 时出错: {str(e)}")
                print("----------------------------------------")
                continue
input_folder = r"C:\Users\30444\Desktop\data" 
output_folder = os.path.join(input_folder, "cleaned data") 
print(f"开始处理文件夹: {input_folder} 中的所有CSV文件")
process_data_folder(input_folder, output_folder)
print("\n所有文件处理完成！")
# %%
"""multiple regression"""
import statsmodels.api as sm

independent_path = r"C:\Users\30444\Desktop\data\cleaned data\cleaned_data1.csv"
dependent_path = r"C:\Users\30444\Desktop\data\cleaned data\cleaned_data 2.csv"
control_path = r"C:\Users\30444\Desktop\data\cleaned data\cleaned_contorl.csv"  

df_independent = pd.read_csv(independent_path)
df_dependent = pd.read_csv(dependent_path)
df_control = pd.read_csv(control_path) 
stock_code_col = df_control.columns[0]
print(f"股票代码列：{stock_code_col}")
date_col = "Accper"
df_independent[date_col] = pd.to_datetime(df_independent[date_col])
df_dependent[date_col] = pd.to_datetime(df_dependent[date_col])
df_control[date_col] = pd.to_datetime(df_control[date_col])
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
core_vars = ["TTM", "ROCA", "ROE"]
control_vars = ["CR", "CaR", "DAR", "DER"]  
print(f"控制变量列：{control_vars}")
y_var = "bes"
merged["Year"] = merged[date_col].dt.year
year_dummies = pd.get_dummies(merged["Year"], prefix="Year")
industry_cols = [col for col in df_control.columns if "Industry" in col] 
industry_dummies = merged[industry_cols].select_dtypes(include=['number'])
print(f"行业固定效应列：{industry_dummies.columns.tolist()}")
X = pd.concat(
    [
        merged[core_vars],
        merged[control_vars], 
        industry_dummies
    ],
    axis=1
)

X = X.select_dtypes(include=['number'])
X = sm.add_constant(X)
y = pd.to_numeric(merged[y_var], errors="coerce").dropna()
X = X.loc[y.index]
model = sm.OLS(y, X).fit()
print("\n=== 多元线性回归结果（控制变量聚合显示）===")
print(model.summary())
print(f"\nR²: {model.rsquared:.4f} | 调整后R²: {model.rsquared_adj:.4f}")
print("\n【核心自变量系数】")
for var in core_vars:
    if var in model.params:
        coef = model.params[var]
        p_val = model.pvalues[var]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{var}: {coef:.4f} ({p_val:.4f}) {sig}")
print("\n【控制变量整体影响】")
if control_vars:
    control_coefs = [model.params[var] for var in control_vars if var in model.params]
    control_pvals = [model.pvalues[var] for var in control_vars if var in model.params]
    avg_coef = np.mean(control_coefs) if control_coefs else 0
    sig_count = sum(1 for p in control_pvals if p < 0.05) if control_pvals else 0
    
    print(f"控制变量平均系数：{avg_coef:.4f}") 
else:
    print("无控制变量")
formula_parts = [f"{model.params['const']:.4f}"] if "const" in model.params else []
formula_parts.extend([f"{model.params[var]:.4f}×{var}" for var in core_vars if var in model.params])
formula_parts.append("[控制变量组合效应]") 
formula = f"{y_var} = " + " + ".join(formula_parts) + " + [时间固定效应] + [行业固定效应] + [扰动项]"
print(f"\n回归公式:\n{formula}")
    
# %%
"""Lagged model:Endogeneity Handling Logic, Result Stability Verification"""


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
# %%
"""Robustness Tests (Alternative Variable Tests) """


import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
independent_path = r"C:\Users\30444\Desktop\data\cleaned data\cleaned_data1.csv"
dependent_path = r"C:\Users\30444\Desktop\data\cleaned data\cleaned_data 2.csv"
control_path = r"C:\Users\30444\Desktop\data\cleaned data\cleaned_contorl.csv"
df_independent = pd.read_csv(independent_path)
df_dependent = pd.read_csv(dependent_path)
df_control = pd.read_csv(control_path)
stock_code_col = df_control.columns[0]  
date_col = "Accper"  
merged = pd.merge(df_independent, df_dependent, on=[stock_code_col, date_col], how="inner")
merged = pd.merge(merged, df_control, on=[stock_code_col, date_col], how="inner")
merged[date_col] = pd.to_datetime(merged[date_col])
roca_median = merged["ROCA"].median()
print(f"ROCA中间值（阈值）：{roca_median:.4f}")
merged["treatment"] = (merged["ROCA"] > roca_median).astype(int)
print(f"\n处理组（高ROCA）样本量：{merged[merged['treatment']==1].shape[0]}")
print(f"对照组（低ROCA）样本量：{merged[merged['treatment']==0].shape[0]}")
covariates = [
    "TTM",
    "CR", "CaR", "DAR", "DER"
]

merged[covariates] = merged[covariates].fillna(merged[covariates].mean())
X_cov = merged[covariates]  
y_treat = merged["treatment"] 
logit_model = LogisticRegression(max_iter=1000, class_weight="balanced")
logit_model.fit(X_cov, y_treat)
merged["ps_score"] = logit_model.predict_proba(X_cov)[:, 1] 
y_pred_proba = merged["ps_score"]
auc = roc_auc_score(y_treat, y_pred_proba)
print(f"\n倾向得分模型AUC值：{auc:.4f}（>0.7说明模型较好）")
plt.figure(figsize=(8, 5))
sns.kdeplot(merged[merged["treatment"]==1]["ps_score"], label="处理组（高ROCA）", fill=True)
sns.kdeplot(merged[merged["treatment"]==0]["ps_score"], label="对照组（低ROCA）", fill=True)
plt.xlabel("Propensity Score")  
plt.ylabel("Density") 
plt.title(f"Distribution of Propensity Scores: Treatment vs Control Groups (AUC={auc:.4f})")  
plt.legend()
plt.show()


treated = merged[merged["treatment"] == 1].copy()
control = merged[merged["treatment"] == 0].copy()


matched_pairs = []
treated_used = set() 

for i, treated_row in treated.iterrows():
    if i in treated_used:
        continue
    ps_diff = np.abs(control["ps_score"] - treated_row["ps_score"])
    min_idx = ps_diff.idxmin()
    if pd.notna(min_idx): 
        matched_pairs.append((i, min_idx))
        treated_used.add(i)
        control = control.drop(min_idx) 

matched_treated = merged.loc[[i for i, j in matched_pairs]]
matched_control = merged.loc[[j for i, j in matched_pairs]]
matched_data = pd.concat([matched_treated, matched_control], ignore_index=True)

print(f"\n匹配后样本量：处理组={matched_treated.shape[0]}, 对照组={matched_control.shape[0]}")


def calculate_smd(before_treat, before_control, after_treat, after_control, vars_list):
    smd_before = []
    smd_after = []
    for var in vars_list:
        mean_treat = before_treat[var].mean()
        mean_control = before_control[var].mean()
        std_treat = before_treat[var].std()
        std_control = before_control[var].std()
        smd_b = (mean_treat - mean_control) / np.sqrt((std_treat**2 + std_control**2) / 2)
        smd_before.append(abs(smd_b))
        
        mean_treat_a = after_treat[var].mean()
        mean_control_a = after_control[var].mean()
        std_treat_a = after_treat[var].std()
        std_control_a = after_control[var].std()
        smd_a = (mean_treat_a - mean_control_a) / np.sqrt((std_treat_a**2 + std_control_a**2) / 2)
        smd_after.append(abs(smd_a))
    
    smd_df = pd.DataFrame({
        "协变量": vars_list,
        "匹配前SMD": smd_before,
        "匹配后SMD": smd_after
    })
    return smd_df


smd_results = calculate_smd(
    before_treat=treated,
    before_control=merged[merged["treatment"]==0],
    after_treat=matched_treated,
    after_control=matched_control,
    vars_list=covariates
)

print("\n平衡性检验（标准化均值差异SMD）：")
print(smd_results.round(4))

plt.figure(figsize=(10, 6))
x = np.arange(len(covariates))
width = 0.35
plt.bar(x - width/2, smd_results["匹配前SMD"], width, label="Before Matching") 
plt.bar(x + width/2, smd_results["匹配后SMD"], width, label="After Matching")   
plt.axhline(y=0.1, color='r', linestyle='--', label="SMD=0.1 (Threshold)")    
plt.xticks(x, covariates, rotation=45)
plt.ylabel("Standardized Mean Difference (SMD)")  
plt.title("Comparison of Covariate Balance Before and After Matching")  
plt.legend()
plt.tight_layout()
plt.show()


att_bes = matched_treated["bes"].mean() - matched_control["bes"].mean()
print(f"\n匹配后平均处理效应（ATT）：{att_bes:.4f}")
print(f"解释：高ROCA组比低ROCA组的平均每股收益（bes）高{att_bes:.4f}单位（控制协变量后）")

X_att = sm.add_constant(matched_data[["treatment"] + covariates])
y_att = matched_data["bes"]
att_model = sm.OLS(y_att, X_att).fit()

print("\n匹配后处理效应回归结果：")
print(att_model.summary())
print(f"\n处理变量（treatment）系数：{att_model.params['treatment']:.4f}，p值：{att_model.pvalues['treatment']:.4f}")
#%%
import warnings
warnings.filterwarnings('ignore')

independent_path = r"C:\Users\30444\Desktop\data\cleaned data\cleaned_data1.csv"
control_path = r"C:\Users\30444\Desktop\data\cleaned data\cleaned_contorl.csv"
deduct_bes_path = r"C:\Users\30444\Desktop\data\Robustness test\Earnings per share after deducting non recurring gains and losses.csv"

date_col = "Accper"
new_y_col = "BES after deducting"

df_independent = pd.read_csv(independent_path, encoding='utf-8')
df_control = pd.read_csv(control_path, encoding='utf-8')
df_deduct_bes = pd.read_csv(deduct_bes_path, encoding='utf-8')

stock_code_col = df_control.columns[0]
print(f"股票代码列：{stock_code_col}")

for df in [df_independent, df_control, df_deduct_bes]:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

merged = pd.merge(
    left=df_independent,
    right=df_deduct_bes[[stock_code_col, date_col, new_y_col]],
    on=[stock_code_col, date_col],
    how="inner"
)
merged = pd.merge(
    left=merged,
    right=df_control,
    on=[stock_code_col, date_col],
    how="inner"
)

merged = merged.dropna(subset=[new_y_col])
core_vars = ["TTM", "ROCA", "ROE"]
control_vars = ["CR", "CaR", "DAR", "DER"]
fill_cols = core_vars + control_vars
merged[fill_cols] = merged[fill_cols].fillna(merged[fill_cols].mean())

print(f"\n合并后样本量：{len(merged)} 条（建议与原回归样本量对比）")
print(f"扣非后每股收益（{new_y_col}）描述性统计：")
print(merged[new_y_col].describe().round(4))

core_vars = ["TTM", "ROCA", "ROE"]
control_vars = ["CR", "CaR", "DAR", "DER"]
print(f"\n控制变量列：{control_vars}")
y = pd.to_numeric(merged[new_y_col], errors="coerce").dropna()

merged["Year"] = merged[date_col].dt.year
year_dummies = pd.get_dummies(merged["Year"], prefix="Year")
print(f"时间固定效应（年份）：{year_dummies.columns.tolist()}")

industry_cols = [col for col in df_control.columns if "Industry" in col]
if not industry_cols:
    print("警告：未检测到行业相关列，将不加入行业固定效应（与原回归保持一致）")
    industry_dummies = pd.DataFrame(index=merged.index)
else:
    industry_dummies = merged[industry_cols].select_dtypes(include=['number'])
print(f"行业固定效应列：{industry_dummies.columns.tolist()}")

X = pd.concat(
    [
        merged[core_vars],
        merged[control_vars],
        year_dummies,
        industry_dummies
    ],
    axis=1
)

X = X.select_dtypes(include=['number'])
X = sm.add_constant(X)
X = X.loc[y.index]

print(f"\n自变量矩阵维度：{X.shape}（行=样本数，列=变量数）")

model = sm.OLS(y, X).fit(cov_type='HC3')

print("\n" + "="*80)
print(f"稳健性检验结果：扣非后每股收益（{new_y_col}）作为因变量")
print("="*80)
print(model.summary())

print(f"\n核心拟合指标：")
print(f"R²: {model.rsquared:.4f} | 调整后R²: {model.rsquared_adj:.4f}")

print("\n【核心自变量系数（稳健性检验）】")
for var in core_vars:
    if var in model.params:
        coef = model.params[var]
        p_val = model.pvalues[var]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{var}: 系数={coef:.4f} | p值={p_val:.4f} {sig}")

print("\n【控制变量整体影响（稳健性检验）】")
if control_vars:
    valid_control_vars = [var for var in control_vars if var in model.params]
    control_coefs = [model.params[var] for var in valid_control_vars]
    control_pvals = [model.pvalues[var] for var in valid_control_vars]
    
    avg_coef = np.mean(control_coefs) if control_coefs else 0
    sig_count = sum(1 for p in control_pvals if p < 0.05) if control_pvals else 0
    
    print(f"有效控制变量数：{len(valid_control_vars)}")
    print(f"控制变量平均系数：{avg_coef:.4f}")
    print(f"5%水平显著的控制变量数：{sig_count}")
else:
    print("无控制变量")

print("\n【回归公式（稳健性检验）】")
formula_parts = []
if "const" in model.params:
    formula_parts.append(f"{model.params['const']:.4f}")
formula_parts.extend([f"{model.params[var]:.4f}×{var}" for var in core_vars if var in model.params])
formula_parts.append("[控制变量组合效应]")
formula_parts.append("[时间固定效应]")
if not industry_dummies.empty:
    formula_parts.append("[行业固定效应]")
formula_parts.append("[扰动项]")

final_formula = f"{new_y_col} = " + " + ".join(formula_parts)
print(final_formula)
#%%
"""Random Forest(Non-linear relationship &Characteristic coefficient measurement)"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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

#%%
import seaborn as sns
""")Multilayer Perceptron(Non-linear pattern & threshold effect quantification) """
core_vars = ["TTM", "ROCA", "ROE"]
lagged_core_vars = ["TTM_lag1", "ROCA_lag1", "ROE_lag1"]

base_coefs = [model.params[var] for var in core_vars if var in model.params]
base_stderr = [model.bse[var] for var in core_vars if var in model.params]
lagged_coefs = [model_lag1.params[var] for var in lagged_core_vars if var in model_lag1.params]
lagged_stderr = [model_lag1.bse[var] for var in lagged_core_vars if var in model_lag1.params]

base_fit = [model.rsquared, model.rsquared_adj]
lagged_fit = [model_lag1.rsquared, model_lag1.rsquared_adj]

base_resid = model.resid
lagged_resid = model_lag1.resid


x = np.arange(len(core_vars))
width = 0.35

plt.figure(figsize=(10, 6), dpi=120)
plt.bar(
    x - width/2, 
    base_coefs, 
    width, 
    yerr=base_stderr, 
    label="Baseline Regression", 
    capsize=5, 
    color="#457b9d", 
    alpha=0.8
)
plt.bar(
    x + width/2, 
    lagged_coefs, 
    width, 
    yerr=lagged_stderr, 
    label="Lagged Regression (1-period lag)", 
    capsize=5, 
    color="#e63946", 
    alpha=0.8
)

def annotate_significance(ax, coefs, stderrs, vars_list, model, x_offset):
    for i, var in enumerate(vars_list):
        if var in model.pvalues:
            p_val = model.pvalues[var]
            height = coefs[i]
            err = stderrs[i]
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            ax.text(
                x[i] + x_offset, 
                height + err + 0.2, 
                sig, 
                ha="center", 
                va="bottom", 
                color="red", 
                fontweight="bold"
            )

annotate_significance(plt.gca(), base_coefs, base_stderr, core_vars, model, -width/2)
annotate_significance(plt.gca(), lagged_coefs, lagged_stderr, lagged_core_vars, model_lag1, +width/2)

plt.ylabel("Coefficient Estimate")
plt.title("Coefficient Comparison: Baseline vs Lagged Model")
plt.xticks(x, core_vars)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


fit_metrics = ["$R^2$", "Adjusted $R^2$"]
x_fit = np.arange(len(fit_metrics))

plt.figure(figsize=(8, 5), dpi=120)
plt.bar(
    x_fit - width/2, 
    base_fit, 
    width, 
    label="Baseline Regression", 
    color="#457b9d", 
    alpha=0.8
)
plt.bar(
    x_fit + width/2, 
    lagged_fit, 
    width, 
    label="Lagged Regression", 
    color="#e63946", 
    alpha=0.8
)

for i, (val_base, val_lag) in enumerate(zip(base_fit, lagged_fit)):
    plt.text(x_fit[i] - width/2, val_base + 0.01, f"{val_base:.4f}", ha="center", va="bottom")
    plt.text(x_fit[i] + width/2, val_lag + 0.01, f"{val_lag:.4f}", ha="center", va="bottom")

plt.ylabel("Value")
plt.title("Model Fit Metrics: Baseline vs Lagged Model")
plt.xticks(x_fit, fit_metrics)
plt.legend()
plt.ylim(0, max(max(base_fit), max(lagged_fit)) + 0.1)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6), dpi=120)
sns.kdeplot(
    base_resid, 
    label="Baseline Regression Residuals", 
    fill=True, 
    color="#457b9d", 
    alpha=0.6
)
sns.kdeplot(
    lagged_resid, 
    label="Lagged Regression Residuals", 
    fill=True, 
    color="#e63946", 
    alpha=0.6
)

plt.xlabel("Residuals")
plt.ylabel("Density")
plt.title("Residual Distribution: Baseline vs Lagged Model")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
#%%
""")Multilayer Perceptron(Non-linear pattern & threshold effect quantification) """

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("📊 【Panel Step 1/6】Data Loading")
print("="*70)

data_paths = {
    "Core Independent Vars": r"C:\Users\30444\Desktop\data\cleaned data\cleaned_data1.csv",
    "Dependent Var (BES)": r"C:\Users\30444\Desktop\data\cleaned data\cleaned_data 2.csv",
    "Control Vars": r"C:\Users\30444\Desktop\data\cleaned data\cleaned_contorl.csv"
}

df_dict = {}
for name, path in data_paths.items():
    try:
        df = pd.read_csv(path)
        df_dict[name] = df
        print(f"✅ {name} Loaded Successfully: {len(df)} Rows × {len(df.columns)} Columns")
    except Exception as e:
        print(f"❌ {name} Loading Failed: {str(e)}")
        raise

df_indep = df_dict["Core Independent Vars"]
df_dep = df_dict["Dependent Var (BES)"]
df_control = df_dict["Control Vars"]

stock_code_col = df_control.columns[0]
date_col = "Accper"
y_var = "bes"
print(f"\n📋 Key Columns: {stock_code_col}, {date_col}, {y_var}")

print("\n" + "="*70)
print("📊 【Panel Step 2/6】Data Merging and Sample Filtering")
print("="*70)

print("⏳ Converting Date Format...")
for df in [df_indep, df_dep, df_control]:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
print("✅ Date Format Conversion Completed")

print(f"\n⏳ Merging Data by 「{stock_code_col}+{date_col}」...")
merged = pd.merge(
    df_indep, df_dep[[stock_code_col, date_col, y_var]],
    on=[stock_code_col, date_col], how="inner"
)
merged = pd.merge(
    merged, df_control[[stock_code_col, date_col] + ["CR", "CaR", "DAR", "DER"]],
    on=[stock_code_col, date_col], how="inner"
)

print(f"✅ Merged Samples: {len(merged)} Rows")
core_vars = ["TTM", "ROCA", "ROE"]
control_vars = ["CR", "CaR", "DAR", "DER"]
merged["Year"] = merged[date_col].dt.year
feature_cols = core_vars + control_vars + ["Year"]

print(f"\n⏳ Handling Missing Values...")
for col in feature_cols:
    merged[col] = merged[col].fillna(merged[col].mean())
merged = merged.dropna(subset=[y_var])
X = merged[feature_cols].copy()
y = merged[y_var].copy()
print(f"✅ Valid Samples: {len(X)} Rows")

print("\n" + "="*70)
print("📊 【Panel Step 3/6】Feature Standardization and Dataset Splitting")
print("="*70)

print("⏳ Standardizing Features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✅ Standardization Completed (First 3 Features Mean: {[f'{x:.4f}' for x in scaler.mean_[:3]]})")

print(f"\n⏳ Splitting into Train/Test Sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
print(f"✅ Training Set: {len(X_train)} Samples, Test Set: {len(X_test)} Samples")

print("\n" + "="*70)
print("📊 【Panel Step 4/6】MLP Model Training")
print("="*70)

mlp_model = MLPRegressor(
    hidden_layer_sizes=(16, 8),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.2,
    random_state=42,
    verbose=False
)

print(f"📋 Model Parameters: {mlp_model.get_params()['hidden_layer_sizes']} Hidden Layers")
print("\n⏳ Training Model...")
mlp_model.fit(X_train, y_train)
print(f"✅ Training Completed (Iterations: {mlp_model.n_iter_}, Loss: {mlp_model.loss_:.6f})")

print("\n" + "="*70)
print("📊 【Panel Step 5/6】Model Evaluation")
print("="*70)

y_train_pred = mlp_model.predict(X_train)
y_test_pred = mlp_model.predict(X_test)

eval_metrics = {
    "R² (Coefficient of Determination)": (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)),
    "RMSE (Root Mean Squared Error)": (np.sqrt(mean_squared_error(y_train, y_train_pred)), np.sqrt(mean_squared_error(y_test, y_test_pred))),
    "MAE (Mean Absolute Error)": (mean_absolute_error(y_train, y_train_pred), mean_absolute_error(y_test, y_test_pred))
}

print("📈 Evaluation Metrics:")
print(f"{'Metric':<30} {'Train Set':<12} {'Test Set':<12}")
print("-" * 55)
for metric, (train_val, test_val) in eval_metrics.items():
    print(f"{metric:<30} {train_val:<12.4f} {test_val:<12.4f}")

print("\n" + "="*70)
print("📊 【Panel Step 6/6】Visualization")
print("="*70)

print("🎨 Generating Predicted vs Actual Scatter Plot...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6, color="#2E86AB", label=f"Test Set (n={len(y_test)})")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2, label="Ideal Fit Line")
plt.xlabel("Actual BES Value")
plt.ylabel("MLP Predicted BES Value")
plt.title(f"Prediction Performance (Test Set R²={eval_metrics['R² (Coefficient of Determination)'][1]:.4f})")
plt.legend()
plt.grid(linestyle="--", alpha=0.5)
plt.show()

print("\n🎨 Generating Feature Importance Bar Chart...")
def perm_importance(model, X, y, feat_names):
    base_score = r2_score(y, model.predict(X))
    importances = []
    for col in range(len(feat_names)):
        X_perm = X.copy()
        np.random.shuffle(X_perm[:, col])
        importances.append(base_score - r2_score(y, model.predict(X_perm)))
    return pd.DataFrame({
        "Feature Name": feat_names, "Permutation Importance": importances
    }).sort_values(by="Permutation Importance", ascending=False)

feat_importance = perm_importance(mlp_model, X_test, y_test, feature_cols)
top10_feat = feat_importance.head(10)

plt.figure(figsize=(12, 6))
ax = sns.barplot(x="Permutation Importance", y="Feature Name", data=top10_feat, palette="Blues_d")
for i, (_, row) in enumerate(top10_feat.iterrows()):
    if row["Feature Name"] in core_vars:
        ax.text(row["Permutation Importance"]+0.005, i, "Core Profit Metric", color="red", fontweight="bold")
plt.xlabel("Permutation Importance (Higher = More Impactful)")
plt.title("Top 10 Feature Importance")
plt.show()

print("\n🎨 Generating Nonlinear Relationship Plot (ROCA vs BES)...")
def plot_roca_nonlinear(model, scaler, X_scaled, feat_cols):
    roca_idx = feat_cols.index("ROCA")
    roca_mean = scaler.mean_[roca_idx]
    roca_scale = scaler.scale_[roca_idx]
    
    roca_min = X_scaled[:, roca_idx].min()
    roca_max = X_scaled[:, roca_idx].max()
    roca_scaled = np.linspace(roca_min, roca_max, 100)
    
    base_data = np.mean(X_scaled, axis=0).reshape(1, -1)
    
    predictions = []
    for val in roca_scaled:
        temp_data = base_data.copy()
        temp_data[0, roca_idx] = val
        predictions.append(model.predict(temp_data)[0])
    
    roca_original = roca_scaled * roca_scale + roca_mean
    
    plt.figure(figsize=(10, 6))
    plt.plot(roca_original, predictions, color="#E63946", linewidth=2.5)
    plt.xlabel("ROCA (Original Scale)")
    plt.ylabel("Predicted BES Value")
    plt.title("Nonlinear Relationship Between ROCA and BES (Other Features Fixed at Mean)")
    plt.grid(linestyle="--", alpha=0.5)
    plt.show()

if "ROCA" in feature_cols:
    plot_roca_nonlinear(mlp_model, scaler, X_scaled, feature_cols)
    print("✅ Nonlinear Relationship Plot Generated Successfully")


print("="*70)
print("📊 【Panel Step 1/7】Data Loading")
print("="*70)

data_paths = {
    "Core Independent Vars": r"C:\Users\30444\Desktop\data\cleaned data\cleaned_data1.csv",
    "Dependent Var (BES)": r"C:\Users\30444\Desktop\data\cleaned data\cleaned_data 2.csv",
    "Control Vars": r"C:\Users\30444\Desktop\data\cleaned data\cleaned_contorl.csv"
}

df_dict = {}
for name, path in data_paths.items():
    try:
        df = pd.read_csv(path)
        df_dict[name] = df
        print(f"✅ {name}: {len(df)} rows × {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ {name} loading failed: {str(e)}")
        raise

df_indep = df_dict["Core Independent Vars"]
df_dep = df_dict["Dependent Var (BES)"]
df_control = df_dict["Control Vars"]

stock_code_col = df_control.columns[0]
date_col = "Accper"
y_var = "bes"
print(f"\n📋 Key Columns: {stock_code_col} | {date_col} | {y_var}")

print("\n" + "="*70)
print("📊 【Panel Step 2/7】Data Merging and Sample Filtering")
print("="*70)

print("⏳ Converting date format...")
for df in [df_indep, df_dep, df_control]:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
print("✅ Date format conversion completed")

print(f"\n⏳ Merging data...")
merged = pd.merge(
    df_indep, df_dep[[stock_code_col, date_col, y_var]],
    on=[stock_code_col, date_col], how="inner"
)
merged = pd.merge(
    merged, df_control[[stock_code_col, date_col] + ["CR", "CaR", "DAR", "DER"]],
    on=[stock_code_col, date_col], how="inner"
)

print(f"✅ After merging: {len(merged)} rows")
core_vars = ["TTM", "ROCA", "ROE"]
control_vars = ["CR", "CaR", "DAR", "DER"]
merged["Year"] = merged[date_col].dt.year
feature_cols = core_vars + control_vars + ["Year"]

print(f"\n⏳ Handling missing values...")
for col in feature_cols:
    merged[col] = merged[col].fillna(merged[col].mean())
merged = merged.dropna(subset=[y_var])
X = merged[feature_cols].copy()
y = merged[y_var].copy()
print(f"✅ Valid samples: {len(X)} rows")

print("\n" + "="*70)
print("📊 【Panel Step 3/7】Constructing Interaction Features (Capturing Heterogeneity)")
print("="*70)

print("⏳ Processing industry features...")
industry_cols = [col for col in df_control.columns if "Industry" in col and df_control[col].dtype != "object"]
if industry_cols:
    merged = pd.merge(
        merged, df_control[[stock_code_col, date_col, industry_cols[0]]],
        on=[stock_code_col, date_col], how="inner"
    )
    industry_feat = industry_cols[0]
else:
    merged["Industry_Code"] = merged[stock_code_col].astype(str).str[:2].astype(int)
    industry_feat = "Industry_Code"
print(f"✅ Industry feature: {industry_feat}")

print("⏳ Generating interaction features...")
interaction_feats = [
    "ROCA×Industry",
    "TTM×Industry",
    "TTM×Year",
    "ROE×Industry"
]

merged["ROCA×Industry"] = merged["ROCA"] * merged[industry_feat]
merged["TTM×Industry"] = merged["TTM"] * merged[industry_feat]
merged["TTM×Year"] = merged["TTM"] * merged["Year"]
merged["ROE×Industry"] = merged["ROE"] * merged[industry_feat]

feature_cols = core_vars + control_vars + ["Year", industry_feat] + interaction_feats
X = merged[feature_cols].copy()

print(f"✅ Interaction features ready: {interaction_feats}")
print(f"✅ Total features: {len(feature_cols)}")

print("\n" + "="*70)
print("📊 【Panel Step 4/7】Feature Standardization")
print("="*70)

print("⏳ Standardizing all features (including interaction features)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✅ Standardization completed:")
print(f"  - Original features mean: {[f'{x:.4f}' for x in scaler.mean_[:3]]}")
print(f"  - Interaction feature (ROCA×Industry) mean: {scaler.mean_[feature_cols.index('ROCA×Industry')]:.4f}")

print(f"\n⏳ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
print(f"✅ Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

print("\n" + "="*70)
print("📊 【Panel Step 5/7】MLP Model Training")
print("="*70)

mlp_model = MLPRegressor(
    hidden_layer_sizes=(16, 8),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.2,
    random_state=42,
    verbose=False
)

print(f"📋 Model parameters: {mlp_model.hidden_layer_sizes} hidden layers | Early stopping enabled")
print("\n⏳ Training in progress...")
mlp_model.fit(X_train, y_train)
print(f"✅ Training completed: {mlp_model.n_iter_} iterations | Loss: {mlp_model.loss_:.6f}")

print("\n" + "="*70)
print("📊 【Panel Step 6/7】Model Evaluation")
print("="*70)

y_train_pred = mlp_model.predict(X_train)
y_test_pred = mlp_model.predict(X_test)

eval_metrics = {
    "R²": (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)),
    "RMSE": (np.sqrt(mean_squared_error(y_train, y_train_pred)), np.sqrt(mean_squared_error(y_test, y_test_pred)))
}

print(f"{'Metric':<6} {'Train Set':<10} {'Test Set':<10}")
print("-" * 30)
for metric, (train_val, test_val) in eval_metrics.items():
    print(f"{metric:<6} {train_val:<10.4f} {test_val:<10.4f}")

print("\n" + "="*70)
print("📊 【Panel Step 7/7】Visualization Analysis")
print("="*70)

print("🎨 Generating prediction performance plot...")
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(y_test, y_test_pred, alpha=0.6, c="#2E86AB", label=f"Test Set (n={len(y_test)})")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2, label="Ideal Fit Line")
ax.set_xlabel("Actual BES Value", fontsize=11)
ax.set_ylabel("MLP Predicted BES Value", fontsize=11)
ax.set_title(f"Prediction Performance (Test Set R²={eval_metrics['R²'][1]:.4f})", fontsize=12)
ax.legend()
ax.grid(linestyle="--", alpha=0.5)
plt.show()

print("\n🎨 Generating feature importance plot...")
def perm_importance(model, X, y, feat_names):
    base_score = r2_score(y, model.predict(X))
    importances = []
    for col in range(len(feat_names)):
        X_perm = X.copy()
        np.random.shuffle(X_perm[:, col])
        importances.append(base_score - r2_score(y, model.predict(X_perm)))
    return pd.DataFrame({
        "Feature Name": feat_names, "Permutation Importance": importances
    }).sort_values(by="Permutation Importance", ascending=False)

feat_importance = perm_importance(mlp_model, X_test, y_test, feature_cols)
top12_feat = feat_importance.head(12)

fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x="Permutation Importance", y="Feature Name", data=top12_feat, palette="Blues_d", ax=ax)

for i, (_, row) in enumerate(top12_feat.iterrows()):
    if row["Feature Name"] in core_vars:
        ax.text(row["Permutation Importance"]+0.005, i, "Core Profit", color="red", fontweight="bold", fontsize=9)
    elif row["Feature Name"] in interaction_feats:
        ax.text(row["Permutation Importance"]+0.005, i, "Interaction", color="orange", fontweight="bold", fontsize=9)

ax.set_xlabel("Permutation Importance (Higher = More Impactful)", fontsize=11)
ax.set_title("Top 12 Feature Importance (Including Industry/Time Interactions)", fontsize=12)
plt.tight_layout()
plt.show()

print("\n🎨 Generating industry heterogeneity plot...")
def plot_industry_hetero(model, scaler, X_scaled, feat_cols, industry_feat):
    roca_idx = feat_cols.index("ROCA")
    industry_idx = feat_cols.index(industry_feat)
    roca_mean = scaler.mean_[roca_idx]
    roca_scale = scaler.scale_[roca_idx]
    
    industry_vals = merged[industry_feat].quantile([0.25, 0.5, 0.75]).values.astype(int)
    roca_range = np.linspace(X_scaled[:, roca_idx].min(), X_scaled[:, roca_idx].max(), 50)
    
    base_data = np.mean(X_scaled, axis=0).reshape(1, -1)
    colors = ["#E63946", "#2E86AB", "#264653"]
    labels = [f"Industry {val}" for val in industry_vals]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, industry_val in enumerate(industry_vals):
        predictions = []
        for roca_val in roca_range:
            temp_data = base_data.copy()
            temp_data[0, roca_idx] = roca_val
            temp_data[0, industry_idx] = (industry_val - scaler.mean_[industry_idx]) / scaler.scale_[industry_idx]
            predictions.append(model.predict(temp_data)[0])
        roca_original = roca_range * roca_scale + roca_mean
        ax.plot(roca_original, predictions, color=colors[idx], linewidth=2.5, label=labels[idx])
    
    ax.set_xlabel("ROCA (Original Scale)", fontsize=11)
    ax.set_ylabel("Predicted BES Value", fontsize=11)
    ax.set_title("Impact of ROCA on BES Across Industries (Industry Heterogeneity)", fontsize=12)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.5)
    plt.show()

if industry_feat in feature_cols:
    plot_industry_hetero(mlp_model, scaler, X_scaled, feature_cols, industry_feat)
    print("✅ Industry heterogeneity plot generated successfully")
