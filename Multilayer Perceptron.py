"""Multilayer Perceptron(Non-linear pattern & threshold effect quantification) """
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
