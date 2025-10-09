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
