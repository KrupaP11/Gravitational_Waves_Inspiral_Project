# All the imports:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Importing the data file
GW = pd.read_csv("gravitational_wave_data.csv")
#print(GW)

# Dividing the file into Data and Targets

# Here is the data
# Was there a better way to do this? - Yes
# Would it have taken me longer to find the easier way than hardcoding it? - Also, yes
X = GW[['h_noisy_t1', 'h_noisy_t2', 'h_noisy_t3', 'h_noisy_t4',
		'h_noisy_t5', 'h_noisy_t6', 'h_noisy_t7', "h_noisy_t8", 
		"h_noisy_t9","h_noisy_t10", "h_noisy_t11", "h_noisy_t12",
		"h_noisy_t13", "h_noisy_t14","h_noisy_t15", "h_noisy_t16",
		"h_noisy_t17", "r0 [m]", "distance_kpc [kpc]", "reduced_mass [kg]"]]
#print(X)

# Here are the targets
Y = GW[["t_merge [s]", "chirp_mass [kg]"]]
#print(y)

# Need to get the y values in log scale because chirp mass to the power of 31
Y_log = np.log10(Y)

# split the data into train and test sets
trainX, testX, trainY_log, testY_log = train_test_split(X, Y_log, test_size = 0.25, random_state=42)

# build the classifier
model = RandomForestRegressor(n_jobs=-1)

# fit the classifier to the training data
model.fit(trainX, trainY_log)

# Need to convert back to linear space
trainY = 10**(trainY_log)

# This is being trained on the log Y because model is fitted to log Y
# Need to convert back to linear space
predictedY = 10**(model.predict(trainX))

print("============================== GW Train ============================")
print("\nMAE:", mean_absolute_error(trainY, predictedY))
print("\nMSE:", mean_squared_error(trainY, predictedY))
print("\nR^2:", r2_score(trainY, predictedY))

# Need to convert back to linear space
testY = 10**(testY_log)

# This is being tested on the log Y because model is fitted to log Y
# Need to convert back to linear space
testY_pred = 10**(model.predict(testX))

print("\n============================== GW Test ============================")
print("\nMAE:", mean_absolute_error(testY, testY_pred))
print("\nMSE:", mean_squared_error(testY, testY_pred))
print("\nR^2:", r2_score(testY, testY_pred))

def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

print("MAPE:", mape(testY.values, testY_pred))


# Plotting area

# Plotting the predictions vs the actual values
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

targets = ["t_merge [s]", "chirp_mass [kg]"]

for i, ax in enumerate(axes):
	actual = testY.values[:, i]
	predicted = testY_pred[:, i]

	ax.scatter(actual, predicted, alpha=0.5, color="mediumpurple", edgecolors="k", linewidths=0.3, label="Predictions")

	min_val = min(actual.min(), predicted.min())
	max_val = max(actual.max(), predicted.max())
	ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")

	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel(f"Actual {targets[i]}", fontsize=12)
	ax.set_ylabel(f"Predicted {targets[i]}", fontsize=12)
	ax.set_title(f"Predicted vs. Actual: {targets[i]}", fontsize=13)
	ax.legend()

plt.tight_layout()
plt.savefig('./predicted_vs_actual.png', dpi=150)
#plt.show()

# Plotting which features where used the most to get the predictions

feature_names = ['h_noisy_t1', 'h_noisy_t2', 'h_noisy_t3', 'h_noisy_t4', 'h_noisy_t5', 'h_noisy_t6', 'h_noisy_t7', 'h_noisy_t8',
                 'h_noisy_t9', 'h_noisy_t10', 'h_noisy_t11', 'h_noisy_t12', 'h_noisy_t13', 'h_noisy_t14', 'h_noisy_t15',
                 'h_noisy_t16','h_noisy_t17',"r0 [m]", 'distance_kpc [kpc]', 'reduced_mass [kg]']

importances = model.feature_importances_

sorted_idx = np.argsort(importances)

fig, ax = plt.subplots(figsize=(8,8))
ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color='steelblue', edgecolor='k', linewidth=0.5)
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Random Forest Feature Importances', fontsize=13)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
#plt.show()

# Plotting the residuals
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, ax in enumerate(axes):
    actual = testY.values[:, i]
    predicted = testY_pred[:, i]
    
    residuals = predicted - actual
    
    ax.scatter(actual, residuals, alpha=0.5, color='mediumseagreen', edgecolors='k', linewidths=0.3)
    
    # Zero line — perfect prediction sits on this line
    ax.axhline(y=0, color='r', linestyle='--', label='Zero Error')
    
    ax.set_xscale('log')
    ax.set_xlabel(f'Actual {targets[i]}', fontsize=12)
    ax.set_ylabel('Residual (Predicted - Actual)', fontsize=12)
    ax.set_title(f'Residuals: {targets[i]}', fontsize=13)
    ax.legend()

plt.tight_layout()
plt.savefig('residuals.png', dpi=150)
#plt.show()
