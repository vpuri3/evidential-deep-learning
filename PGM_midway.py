# 10708 class project
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
warnings.resetwarnings()
from sklearn.preprocessing import StandardScaler

import functools
import tensorflow as tf
import evidential_deep_learning as edl

# def log_likelihood(y, y_pred, variance):
#     log_likelihood = np.sum(-0.5 * np.log(2 * np.pi * variance) - 0.5 * ((y - y_pred) ** 2) / variance)
#     return log_likelihood

# def gaussian_log_likelihood(y, mu, sigma):
def log_likelihood(y, mu, sigma):
    """
    Calculate the log likelihood for a Gaussian likelihood function.

    Parameters:
    y (numpy array): observed data
    mu (float): mean parameter
    sigma (float): standard deviation parameter

    Returns:
    log_likelihood (float): log likelihood value
    """
    N = len(y)
    log_likelihood = -N/2 * np.log(2 * np.pi * sigma**2)
    log_likelihood -= 1/(2 * sigma**2) * np.sum((y - mu)**2)
    return np.mean(-log_likelihood)

##################
# LOAD DATA
##################

# # Load the California housing dataset
# housing = fetch_california_housing()

# # Load the Yacht Hydrodynamics dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
# column_names = ['Longitudinal Position', 'Prismatic Coefficient', 'Length-displacement Ratio','Beam-draught Ratio', 'Length-beam Ratio', 'Froude Number', 'Residuary Resistance']
# yacht_data = pd.read_csv(url, names=column_names, delim_whitespace=True)

# # Load the Wine Quality dataset
# red_wine_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
# white_wine_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
# wine_data = pd.concat([red_wine_data, white_wine_data], ignore_index=True)

# # Load the Heart Disease dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
# column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
# heart_data = pd.read_csv(url, names=column_names, na_values="?")
# heart_data.dropna(inplace=True)

##################
# NORMALIZE
##################

# # Choose dataset
# X, y = housing.data, housing.target
# X, y = red_wine_data.iloc[:,:-1], red_wine_data.iloc[:, -1].astype(float)

# # Normalize
# X = StandardScaler().fit_transform(X)
# y = StandardScaler().fit_transform(y.reshape(-1,1))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_test, y_test = X, y # test on entire dataset

##################
# TOY DATASET
##################

def toydata(x_min, x_max, n, train=True):
    x = np.linspace(x_min, x_max, n)
    x = np.expand_dims(x, -1).astype(np.float32)

    sigma = 3 * np.ones_like(x) if train else np.zeros_like(x)
    y = x**3 + np.random.normal(0, sigma).astype(np.float32)

    return x, y

X_train, y_train = toydata(-4, 4, 1000)
X_test , y_test  = toydata(-7, 7, 1000, train=False)

##################
# EDL Model
##################

# el_model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation="relu"),
#     tf.keras.layers.Dense(64, activation="relu"),
#     edl.layers.DenseNormalGamma(1),
# ])
#
# def EvidentialRegressionLoss(true, pred):
#     return edl.losses.EvidentialRegression(true, pred, coeff=1e-2)
#
# el_model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-4),
#     loss=EvidentialRegressionLoss)
# el_model.fit(X_train, y_train, batch_size=100, epochs=1000)
#
# # Predict and plot using the trained model
# y_pred = el_model(X_test)
# mu, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
el_preds = mu[:, 0]
el_vars = np.sqrt(beta / (v * (alpha - 1)))
el_var = np.mean(el_vars)
el_mse = mean_squared_error(y_test, el_preds)
el_llh = log_likelihood(y_test, el_preds, el_vars)

plt.figure()
plt.hist(el_vars, bins=20, density=True)
plt.savefig("hist_edl.png")

##################
# DNN-MODEL
##################

# dn_model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation="relu"),
#     tf.keras.layers.Dense(64, activation="relu"),
#     tf.keras.layers.Dense(1),
# ])
#
# dn_model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-4),
#     loss=edl.losses.MSE)
# dn_model.fit(X_train, y_train, batch_size=100, epochs=1000)
#
dn_preds = dn_model(X_test)
dn_mse = mean_squared_error(y_test, dn_preds)

##################
# RANDOM FOREST REGRESSOR
##################

rf_regressor = RandomForestRegressor(n_estimators=10, random_state=42)
rf_regressor.fit(X_train, y_train)
rf_preds = rf_regressor.predict(X_test)
rf_preds_tree = np.stack([tree.predict(X_test) for tree in rf_regressor.estimators_], axis=0)
rf_vars = np.var(rf_preds_tree, axis=0)

rf_var = np.mean(rf_vars)
rf_mse = mean_squared_error(y_test, rf_preds)
rf_llh = log_likelihood(y_test, rf_preds, rf_vars)

plt.figure()
plt.hist(rf_vars, bins=20, density=True)
plt.savefig("hist_rf.png")

##################
# 
##################

print(f"DNN - MSE: {dn_mse}")
print(f"EDL - MSE: {el_mse}\t VAR: {el_var}\t LLH: {el_llh}")
print(f"RFR - MSE: {rf_mse}\t VAR: {rf_var}\t LLH: {rf_llh}")

#
