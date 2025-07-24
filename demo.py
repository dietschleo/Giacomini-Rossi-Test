"""
This file demonstrates the capabilities of the Giacomini-Rossi Python implementation,
which is based on the Giacomini-Rossi (2010, Journal of Applied Econometrics) forecast fluctuation test
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from Giacomini_Rossi import fluctuation_test

# Load the test data
print("Loading test data...")
data = pd.read_csv('giacross_test_data.csv')

# Create time series variables (replicating Stata's time setup)
data['year'] = data['pdate'].astype(int)
data['quarter'] = ((data['pdate'] - data['year']) * 4 + 1).round().astype(int)

print(f"Data loaded: {len(data)} observations from {data['pdate'].min():.2f} to {data['pdate'].max():.2f}")
print(f"Variables: {list(data.columns)}")

# Prepare forecast losses (squared errors)
# Using squared errors as loss function: (actual - forecast)^2
loss_forc = (data['realiz'] - data['forc'])**2
loss_spf = (data['realiz'] - data['spf'])**2
 
print(f"\nMean squared errors:")
print(f"FORC model: {loss_forc.mean():.4f}")
print(f"SPF model: {loss_spf.mean():.4f}")
 
# Demo 1: lag length set to 3, default 2-sided test (window=60 observations)
print("\n" + "="*60)
print("DEMO 1: Window=60, alpha=0.05, lag_truncate=3, two-sided test")
print("="*60)

# Calculate window size as fraction of total sample
window_size = 60
mu = window_size / len(data)
print(f"Using mu = {mu:.3f} (window size {window_size} out of {len(data)} observations)")

result1 = fluctuation_test(
    loss1=loss_forc,
    loss2=loss_spf,
    mu=mu,
    lag_truncate=3,
    conf_level=0.05,
    side=2
)

# Find supreme test statistic (maximum absolute value)
tstat_sup = np.max(np.abs(result1['dm_stat']))
cv = result1['cv_high'].iloc[0]
print(f"The value of the test statistic is {tstat_sup:.4f}")
print(f"The critical value is {cv:.4f} at significance level 0.05")

# Create plot
plt.figure(figsize=(12, 6))
plt.plot(result1['time_idx'], result1['dm_stat'], 'b-', linewidth=2, label='DM statistic')
plt.axhline(y=cv, color='r', linestyle='--', linewidth=1, label=f'Critical value (+{cv:.3f})')
plt.axhline(y=-cv, color='r', linestyle='--', linewidth=1, label=f'Critical value (-{cv:.3f})')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('Time Index')
plt.ylabel('DM Statistic')
plt.title('Giacomini-Rossi Fluctuation Test - Demo 1\n(Window=60, lag=3, two-sided)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('GR_demo_1.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure

# Test interpretation
rejections = np.sum((np.abs(result1['dm_stat']) > cv))
print(f"Number of periods with rejection: {rejections} out of {len(result1)}")
 
# Demo 2: automatic lag length selection (use 0), one-sided test
print("\n" + "="*60)
print("DEMO 2: Window=60, alpha=0.05, lag_truncate=0 (no lags), one-sided test")
print("="*60)

result2 = fluctuation_test(
    loss1=loss_forc,
    loss2=loss_spf,
    mu=mu,
    lag_truncate=0,  # No lag truncation (automatic selection equivalent)
    conf_level=0.05,
    side=1
)

# Find supreme test statistic (maximum value for one-sided)
tstat_sup2 = np.max(result2['dm_stat'])
cv2 = result2['cv_high'].iloc[0]

print(f"The value of the test statistic is {tstat_sup2:.4f}")
print(f"The critical value is {cv2:.4f} at significance level 0.05")

# Create plot
plt.figure(figsize=(12, 6))
plt.plot(result2['time_idx'], result2['dm_stat'], 'b-', linewidth=2, label='DM statistic')
plt.axhline(y=cv2, color='r', linestyle='--', linewidth=1, label=f'Critical value ({cv2:.3f})')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('Time Index')
plt.ylabel('DM Statistic')
plt.title('Giacomini-Rossi Fluctuation Test - Demo 2\n(Window=60, lag=0, one-sided)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('GR_demo_2.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure

# Test interpretation
rejections2 = np.sum(result2['dm_stat'] > cv2)
print(f"Number of periods with rejection: {rejections2} out of {len(result2)}")

# Summary comparison
print("\n" + "="*60)
print("SUMMARY COMPARISON")
print("="*60)
print(f"Demo 1 (two-sided): Supreme statistic = {tstat_sup:.4f}, Critical value = {cv:.4f}")
print(f"Demo 2 (one-sided): Supreme statistic = {tstat_sup2:.4f}, Critical value = {cv2:.4f}")
print(f"FORC vs SPF overall performance: FORC MSE = {loss_forc.mean():.4f}, SPF MSE = {loss_spf.mean():.4f}")

# Additional analysis: Show periods of instability
print("\nPeriods of significant forecast performance differences:")
significant_periods1 = result1[np.abs(result1['dm_stat']) > cv]
significant_periods2 = result2[result2['dm_stat'] > cv2]

if len(significant_periods1) > 0:
    print(f"\nDemo 1 (two-sided) - {len(significant_periods1)} significant periods:")
    for _, row in significant_periods1.head(5).iterrows():
        period_date = data.iloc[int(row['time_idx'])-1]['pdate']
        print(f"  Time {int(row['time_idx']):.0f} ({period_date:.2f}): DM = {row['dm_stat']:.3f}")
    if len(significant_periods1) > 5:
        print(f"  ... and {len(significant_periods1)-5} more periods")
 
if len(significant_periods2) > 0:
    print(f"\nDemo 2 (one-sided) - {len(significant_periods2)} significant periods:")
    for _, row in significant_periods2.head(5).iterrows():
        period_date = data.iloc[int(row['time_idx'])-1]['pdate']
        print(f"  Time {int(row['time_idx']):.0f} ({period_date:.2f}): DM = {row['dm_stat']:.3f}")
    if len(significant_periods2) > 5:
        print(f"  ... and {len(significant_periods2)-5} more periods")

print("\nAnalysis complete. Plots saved as 'GR_demo_1.png' and 'GR_demo_2.png'")
