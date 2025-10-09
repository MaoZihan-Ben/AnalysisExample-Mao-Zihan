"""Robustness Tests (Alternative Variable Tests) """
"""This the analysis sections and cannot be directly run：Lack of necessary libraries. 
They are only for the viewing."""

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
