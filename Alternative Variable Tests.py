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
