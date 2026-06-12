

import pandas as pd.result_df = pd.DataFrame(
    results,
    columns=["Model", "R2 Score", "MAE", "RMSE"]
)

result_df.sort_values(by="R2 Score", ascending=False)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')

# read file
data = pd.read_csv('/Users/simrannimje/Desktop/sem 2/iit_ropar_power_nov_feb.csv')

data.head()

# info 
print("Shape of dataset:", data.shape)

data.info()
data.isnull().sum()

# changing in to date time
data['Date'] = pd.to_datetime(data['Date'])

data['Day'] = data['Date'].dt.day
data['Month_Name'] = data['Date'].dt.month_name()

data.head()
data.describe()
monthly_power = data.groupby("Month_Name")["Units_kWh"].sum().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(x="Month_Name", y="Units_kWh", data=monthly_power)

plt.title("Monthly Power Consumption at IIT Ropar")
plt.xlabel("Month")
plt.ylabel("Units (kWh)")
plt.show()

plt.figure(figsize=(10,6))

sns.boxplot(x="Building_Type", y="Units_kWh", data=data)

plt.title("Power Consumption by Building Type")
plt.show()

# temp vs power consumption
plt.figure(figsize=(10,6))

sns.scatterplot(x="Temperature_C", y="Units_kWh", hue="Building_Type", data=data)

plt.title("Temperature vs Power Consumption")
plt.show()

#heater  usage impact 
plt.figure(figsize=(8,6))

sns.boxplot(x="Heater_Use", y="Units_kWh", data=data)

plt.title("Heater Usage Impact on Power")
plt.show()

# correlation heat map

plt.figure(figsize=(10,8))

sns.heatmap(data.corr(numeric_only=True),
            annot=True,
            cmap="YlGnBu",
            linewidths=1)

plt.title("Correlation Heatmap")
plt.show()

# heat map

pivot = data.pivot_table(values="Units_kWh",
                         index="Building",
                         columns="Month_Name")

plt.figure(figsize=(12,8))

sns.heatmap(pivot,
            cmap="plasma",
            annot=False)

plt.title("Building-wise Monthly Power Heatmap")
plt.show()

# feature correlation 

sns.pairplot(data[[
    "Occupancy",
    "Temperature_C",
    "Heater_Use",
    "AC_Use",
    "Units_kWh",
    "Peak_Load_kW",
    
    
]])

plt.show()

# model selection

features = data[[
    "Occupancy",
    "Temperature_C",
    "Heater_Use",
    "AC_Use"
]]

target = data["Units_kWh"]

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.figure(figsize=(8,6))

plt.scatter(y_test, predictions, color = "blue")

plt.xlabel("Actual Units")
plt.ylabel("Predicted Units")
plt.title("Actual vs Predicted Power")

plt.show()

importance = pd.DataFrame({
    "Feature": features.columns,
    "Coefficient": model.coef_
})

importance

plt.figure(figsize=(8,6))

sns.barplot(x="Coefficient", y="Feature", data=importance)

plt.title("Feature Importance")

plt.show()

new_data = pd.DataFrame({
    "Occupancy":[300],
    "Temperature_C":[8],
    "Heater_Use":[1],
    "AC_Use":[0]
})

prediction = model.predict(new_data)

print("Predicted Power Consumption:", prediction[0])

# different model result

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

models = {

    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=1.0),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    results.append([name, r2, mae, rmse])

result_df = pd.DataFrame(
    results,
    columns=["Model", "R2 Score", "MAE", "RMSE"]
)

result_df.sort_values(by="R2 Score", ascending=False)

plt.figure(figsize=(10,6))

sns.barplot(
    x="Model",
    y="R2 Score",
    data=result_df,
    color = "green"
)

plt.xticks(rotation=45)

plt.title("Machine Learning Model Comparison")

plt.xlabel("Model")

plt.ylabel("R2 Score")

plt.show()

# random forest prediction

rf_model = RandomForestRegressor(

    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

r2_rf = r2_score(y_test, rf_predictions)

mae_rf = mean_absolute_error(y_test, rf_predictions)

rmse_rf = np.sqrt(mean_squared_error(y_test, rf_predictions))

print("Random Forest R2:", r2_rf)
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)

plt.figure(figsize=(8,6))

plt.scatter(y_test, rf_predictions, color = "orange")

plt.xlabel("Actual Power")

plt.ylabel("Predicted Power")

plt.title("Random Forest Prediction")

plt.show()

importance_rf = pd.DataFrame({

    "Feature": features.columns,
    "Importance": rf_model.feature_importances_
})

importance_rf = importance_rf.sort_values(
    by="Importance",
    ascending=False
)

importance_rf

plt.figure(figsize=(8,6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_rf,
    color = "yellow"
)

plt.title("Random Forest Feature Importance")

plt.show()

"""
IIT Ropar Power Consumption: Prediction → Optimization Pipeline
================================================================
Stage 1: Predict daily Units_kWh per building (XGBoost + feature engineering)
Stage 2: Optimize scheduling using CPT + CCF framework
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ──
df = pd.read_csv('/mnt/user-data/uploads/iit_ropar_power_nov_feb.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['DayOfMonth'] = df['Date'].dt.day
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# Encode categoricals
le_bldg = LabelEncoder()
le_type = LabelEncoder()
df['Building_Enc'] = le_bldg.fit_transform(df['Building'])
df['Type_Enc'] = le_type.fit_transform(df['Building_Type'])

# Lag features (per building, previous day consumption)
df = df.sort_values(['Building', 'Date'])
df['Lag1_kWh'] = df.groupby('Building')['Units_kWh'].shift(1)
df['Lag7_kWh'] = df.groupby('Building')['Units_kWh'].shift(7)
df['RollingMean7'] = df.groupby('Building')['Units_kWh'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
df = df.dropna()

# ══════════════════════════════════════════════════════
#  STAGE 1: PREDICTION
# ══════════════════════════════════════════════════════
print("=" * 60)
print(" STAGE 1: PREDICTION MODEL")
print("=" * 60)

features = ['Occupancy', 'Temperature_C', 'Heater_Use', 'AC_Use',
            'Month', 'DayOfWeek', 'DayOfMonth', 'IsWeekend',
            'Building_Enc', 'Type_Enc',
            'Lag1_kWh', 'Lag7_kWh', 'RollingMean7']

X = df[features]
y = df['Units_kWh']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model comparison
models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
}

try:
    from xgboost import XGBRegressor
    models['XGBoost'] = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08,
                                      subsample=0.8, colsample_bytree=0.8, random_state=42)
except:
    pass

results = []
best_r2, best_model, best_name = -1, None, ""

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    cv = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
    results.append([name, r2, mae, rmse, cv])
    print(f"\n{name}:")
    print(f"  R² = {r2:.4f} | MAE = {mae:.2f} | RMSE = {rmse:.2f} | CV-R² = {cv:.4f}")
    if r2 > best_r2:
        best_r2, best_model, best_name = r2, model, name

print(f"\n✓ Best model: {best_name} (R² = {best_r2:.4f})")

# Feature importance
preds_best = best_model.predict(X_test)
imp = pd.DataFrame({'Feature': features,
                     'Importance': best_model.feature_importances_}).sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(imp.to_string(index=False))

# ── Plot 1: Actual vs Predicted ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.scatter(y_test, preds_best, alpha=0.5, s=15, c='#2563eb')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1.5)
ax.set_xlabel('Actual (kWh)'); ax.set_ylabel('Predicted (kWh)')
ax.set_title(f'{best_name}: Actual vs Predicted (R²={best_r2:.3f})')

# Plot 2: Feature Importance
ax = axes[1]
ax.barh(imp['Feature'], imp['Importance'], color='#10b981')
ax.set_xlabel('Importance'); ax.set_title('Feature Importance')
ax.invert_yaxis()

# Plot 3: Residuals
residuals = y_test.values - preds_best
ax = axes[2]
ax.hist(residuals, bins=30, color='#f59e0b', edgecolor='white')
ax.axvline(0, color='red', linestyle='--')
ax.set_xlabel('Residual (kWh)'); ax.set_title('Residual Distribution')

plt.tight_layout()
plt.savefig('/home/claude/prediction_results.png', dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════
#  STAGE 2: OPTIMIZATION (CPT + CCF Scheduling)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" STAGE 2: OPTIMIZATION (Cost Minimization)")
print("=" * 60)

# Simulate 24-hour RTP pricing (₹/kWh) based on typical Indian ToD tariffs
hours = np.arange(24)
np.random.seed(42)
base_rtp = np.array([4.5, 4.2, 4.0, 3.8, 3.9, 4.5, 5.5, 7.0, 8.0, 8.5,
                      8.2, 7.8, 7.5, 7.8, 8.0, 8.5, 9.0, 9.5, 9.0, 8.0,
                      7.0, 6.0, 5.5, 5.0])
rtp = base_rtp + np.random.normal(0, 0.3, 24)

# Define shiftable vs non-shiftable load fractions per building type
load_profiles = {
    'Hostel':      {'shiftable_frac': 0.30, 'appliances': ['Geyser', 'Washing Machine', 'Iron']},
    'Academic':    {'shiftable_frac': 0.20, 'appliances': ['HVAC', 'Lab Equipment']},
    'Facility':    {'shiftable_frac': 0.25, 'appliances': ['Kitchen Equipment', 'Water Pump']},
    'Residential': {'shiftable_frac': 0.35, 'appliances': ['Geyser', 'Washing Machine', 'Dishwasher']},
}

# Typical hourly load shape (normalized)
load_shape = np.array([0.3, 0.25, 0.2, 0.2, 0.25, 0.4, 0.7, 0.9, 1.0, 0.95,
                        0.85, 0.8, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 0.95, 0.85,
                        0.7, 0.55, 0.45, 0.35])
load_shape = load_shape / load_shape.sum()

def optimize_schedule(daily_kwh, building_type, ccf):
    """CPT-based scheduling: shift loads to cheap hours within CCF window."""
    profile = load_profiles[building_type]
    shiftable = daily_kwh * profile['shiftable_frac']
    fixed = daily_kwh * (1 - profile['shiftable_frac'])
    
    # Fixed load follows normal shape
    fixed_hourly = fixed * load_shape
    
    # Shiftable load: original POI (period of interest)
    shift_shape = load_shape.copy()
    shift_hourly_original = shiftable * shift_shape
    
    # CCF determines flexibility window (hours of allowed shift)
    max_shift = int((1 - ccf) * 6)  # CCF=1 → 0h shift, CCF=0.5 → 3h shift
    
    # For each hour with shiftable load, find cheaper slot within window
    shift_hourly_opt = np.zeros(24)
    for h in range(24):
        if shift_hourly_original[h] > 0.01:
            window = range(max(0, h - max_shift), min(24, h + max_shift + 1))
            best_h = min(window, key=lambda x: rtp[x])
            shift_hourly_opt[best_h] += shift_hourly_original[h]
    
    total_original = fixed_hourly + shift_hourly_original
    total_optimized = fixed_hourly + shift_hourly_opt
    
    cost_original = np.sum(total_original * rtp)
    cost_optimized = np.sum(total_optimized * rtp)
    
    return cost_original, cost_optimized, total_original, total_optimized

# Run optimization for all buildings, all CCF levels
latest_date = df['Date'].max()
latest = df[df['Date'] == latest_date].copy()

ccf_levels = [1.0, 0.75, 0.50]
opt_results = []

for _, row in latest.iterrows():
    for ccf in ccf_levels:
        cost_orig, cost_opt, _, _ = optimize_schedule(row['Units_kWh'], row['Building_Type'], ccf)
        savings_pct = (cost_orig - cost_opt) / cost_orig * 100
        opt_results.append({
            'Building': row['Building'],
            'Type': row['Building_Type'],
            'Daily_kWh': row['Units_kWh'],
            'CCF': ccf,
            'Cost_Original': cost_orig,
            'Cost_Optimized': cost_opt,
            'Savings_INR': cost_orig - cost_opt,
            'Savings_Pct': savings_pct
        })

opt_df = pd.DataFrame(opt_results)

# Summary
print("\n── Cost Savings Summary (₹/day) ──")
summary = opt_df.groupby('CCF').agg(
    Total_Original=('Cost_Original', 'sum'),
    Total_Optimized=('Cost_Optimized', 'sum'),
    Total_Savings=('Savings_INR', 'sum'),
    Avg_Savings_Pct=('Savings_Pct', 'mean')
).round(2)
print(summary)

print("\n── Per-Building Savings at CCF=0.50 ──")
ccf05 = opt_df[opt_df['CCF'] == 0.50].sort_values('Savings_INR', ascending=False)
print(ccf05[['Building', 'Type', 'Daily_kWh', 'Cost_Original', 'Cost_Optimized', 'Savings_INR', 'Savings_Pct']].to_string(index=False))

# ── Optimization Plots ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Savings by CCF
ax = axes[0, 0]
pivot_savings = opt_df.groupby(['Type', 'CCF'])['Savings_Pct'].mean().unstack()
pivot_savings.plot(kind='bar', ax=ax, colormap='viridis')
ax.set_title('Avg Savings (%) by Building Type & CCF')
ax.set_ylabel('Savings (%)'); ax.set_xlabel('Building Type')
ax.legend(title='CCF'); ax.tick_params(axis='x', rotation=45)

# Plot 2: Load shifting example (top consumer)
ax = axes[0, 1]
top_bldg = latest.loc[latest['Units_kWh'].idxmax()]
_, _, orig_profile, opt_profile = optimize_schedule(top_bldg['Units_kWh'], top_bldg['Building_Type'], 0.50)
ax.fill_between(hours, orig_profile, alpha=0.3, color='red', label='Original Load')
ax.plot(hours, orig_profile, 'r-', linewidth=1.5)
ax.fill_between(hours, opt_profile, alpha=0.3, color='green', label='Optimized (CCF=0.5)')
ax.plot(hours, opt_profile, 'g-', linewidth=1.5)
ax2 = ax.twinx()
ax2.plot(hours, rtp, 'k--', alpha=0.5, label='RTP (₹/kWh)')
ax2.set_ylabel('RTP (₹/kWh)')
ax.set_title(f'Load Shifting: {top_bldg["Building"]}')
ax.set_xlabel('Hour'); ax.set_ylabel('Load (kWh)')
ax.legend(loc='upper left'); ax2.legend(loc='upper right')

# Plot 3: Total campus savings by CCF
ax = axes[1, 0]
bars = summary['Total_Savings'].values
colors = ['#ef4444', '#f59e0b', '#10b981']
ax.bar([f'CCF={c}' for c in ccf_levels], bars, color=colors)
for i, v in enumerate(bars):
    ax.text(i, v + 5, f'₹{v:.0f}', ha='center', fontweight='bold')
ax.set_title('Total Daily Campus Savings (₹)')
ax.set_ylabel('Savings (₹/day)')

# Plot 4: Top 10 buildings by optimization potential
ax = axes[1, 1]
top10 = ccf05.nlargest(10, 'Savings_INR')
ax.barh(top10['Building'], top10['Savings_INR'], color='#6366f1')
ax.set_xlabel('Savings (₹/day)'); ax.set_title('Top 10 Buildings: Optimization Potential (CCF=0.5)')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('/home/claude/optimization_results.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Monthly projection ──
print("\n── Monthly Savings Projection ──")
for ccf in ccf_levels:
    daily = summary.loc[ccf, 'Total_Savings']
    print(f"  CCF={ccf:.2f}: ₹{daily:.0f}/day → ₹{daily*30:.0f}/month → ₹{daily*365:.0f}/year")

# Save results
opt_df.to_csv('/home/claude/optimization_details.csv', index=False)
results_df = pd.DataFrame(results, columns=['Model', 'R2', 'MAE', 'RMSE', 'CV_R2'])
results_df.to_csv('/home/claude/model_comparison.csv', index=False)

print("\n✓ Pipeline complete. Files saved.")
