#

import functools
import numpy as np
import tensorflow as tf
import evidential_deep_learning as edl
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

##################
# Determinsitc
##################

def train_DN(X_train, y_train, H, W=64, act="relu", batch_size=100):
    hidden = [tf.keras.layers.Dense(W, activation = act) for _ in range(H)]
    model = tf.keras.Sequential([
        *hidden,
        edl.layers.Dense(1),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss=edl.losses.MSE)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=500)
    return model

def predict_DN(model, X):
    return model(X)

##################
# Ensemble
##################

def train_SB(X_train, y_train, H, W=64, act="relu", E=5, batch_size=100):
    models = [
        train_DN(X_train, y_train, H, W, act, batch_size) for _ in range(E)
    ]
    return models

def predict_SB(models, X):
    ys = np.stack(
        [predict_DN(model, X) for model in models], axis=0,
    )
    y = np.mean(ys, axis=0)
    var = np.var(ys, axis=0)

    return y, var

##################
# GL
##################

def GL_loss(y, out):
    mu, sigma = tf.split(out, 2, axis=-1)
    return edl.losses.Gaussian_NLL(y, mu, sigma)

def train_GL(X_train, y_train, H, W = 64, act = "relu", batch_size=100):

    hidden = [tf.keras.layers.Dense(W, activation = act) for _ in range(H)]
    model = tf.keras.Sequential([
        *hidden,
        edl.layers.DenseNormal(1),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss=GL_loss)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=500)
    return model

def predict_GL(model, X):
    mu, sigma = tf.split(model(X), 2, axis=-1)
    return mu, sigma

##################
# EV
##################

def train_EV(X_train, y_train, H, W = 64, act = "relu", batch_size=100,
             coeff=1e-5):

    def EV_loss(y, out):
        return edl.losses.EvidentialRegression(y, out, coeff=coeff)

    hidden = [tf.keras.layers.Dense(W, activation = act) for _ in range(H)]
    model = tf.keras.Sequential([
        *hidden,
        edl.layers.DenseNormalGamma(1),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss=EV_loss)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=500)
    return model

def predict_EV(model, X):
    mu, v, alpha, beta = tf.split(model(X), 4, axis=-1)
    sigma = np.sqrt(beta / (alpha - 1)) # aleatoric
    var = beta / (v * (alpha - 1)) # epistemic

    return mu, sigma, var

##################
# RandomForest
##################
def train_RF(X_train, y_train, N, d):
    model = RandomForestRegressor(n_estimators=N, max_depth=d, random_state=42)
    model = model.fit(X_train, y_train)
    return model

def predict_RF(model, X):
    y = model.predict(X)
    y_tree = np.stack([tree.predict(X) for tree in model.estimators_], axis=0)
    var = np.var(y_tree, axis=0)

    return y, var

##################
# Data
##################

def makedata1D(n, s, train=True, ood=False):
    x = np.linspace(-1, 1.5, n) if ood else np.linspace(-1, 1, n)
    x = np.expand_dims(x, -1).astype(np.float32)

    sigma = s * np.ones_like(x) if train else np.zeros_like(x)

    y = np.sin(np.pi * x)
    y += np.random.normal(0, sigma).astype(np.float32)

    return x, y

#===================================================#

def experiment1():

    # experiment hyperparameters
    H = 4
    N = 1000
    s1 = 0.05
    s2 = 0.10
    s3 = 0.20

    ss1 = s1 * np.ones(N)
    ss2 = s2 * np.ones(N)
    ss3 = s3 * np.ones(N)

    # make data
    _X1, _y1 = makedata1D(N, s1)
    _X2, _y2 = makedata1D(N, s2)
    _X3, _y3 = makedata1D(N, s3)

    X1_, y1_ = makedata1D(N, s1, train=False)
    X2_, y2_ = makedata1D(N, s2, train=False)
    X3_, y3_ = makedata1D(N, s3, train=False)

    # train gaussian likelihood models
    model_G1 = train_GL(_X1, _y1, H)
    model_G2 = train_GL(_X2, _y2, H)
    model_G3 = train_GL(_X3, _y3, H)

    # train evidential regression models
    model_E1 = train_EV(_X1, _y1, H, coeff=0e-1)
    model_E2 = train_EV(_X2, _y2, H, coeff=0e-1)
    model_E3 = train_EV(_X3, _y3, H, coeff=0e-1)

    # predict GL
    y_G1, s_G1 = predict_GL(model_G1, X1_)
    y_G2, s_G2 = predict_GL(model_G2, X2_)
    y_G3, s_G3 = predict_GL(model_G3, X3_)

    y_E1, s_E1, _ = predict_EV(model_E1, X1_)
    y_E2, s_E2, _ = predict_EV(model_E2, X2_)
    y_E3, s_E3, _ = predict_EV(model_E3, X3_)

    fig, ax = plt.subplots(ncols = 2, nrows = 3, figsize = (12, 8))

    for i in range(3):
        ax[i,0].set_ylabel(f"Case {i+1}")
        ax[i,1].set_ylabel("std. dev")

    ax[0,0].set_title("Prediction")
    ax[0,1].set_title("Aleatoric uncertainty")

    ax[2,0].set_xlabel("x")
    ax[2,1].set_xlabel("x")

    #========================#
    ax[0,0].scatter(_X1, _y1, c = "black", s = 1, label = "Data")
    ax[0,0].plot(X1_, y1_, c = "red", linewidth = 3, label = "True function")
    ax[0,0].plot(X1_, y_G1, c = "brown", linewidth = 3, label = "Gaussian likelihood")
    ax[0,0].plot(X1_, y_E1, c = "blue" , linewidth = 3, label = "Evidential regression")
    ax[0,0].legend(loc = "upper left")

    ax[0,1].plot(X1_, s_G1, c = "brown", linewidth = 3, label = "Gaussian likelihood")
    ax[0,1].plot(X1_, s_E1, c = "blue" , linewidth = 3, label = "Evidential regression")
    ax[0,1].plot(X1_, ss1 , c = "black" , linewidth = 3, label = "True std. dev", linestyle = "--")
    ax[0,1].legend()

    #========================#
    ax[1,0].scatter(_X2, _y2, c = "black", s = 1, label = "Data")
    ax[1,0].plot(X2_, y2_, c = "red", linewidth = 3, label = "True function")
    ax[1,0].plot(X2_, y_G2, c = "brown", linewidth = 3, label = "Gaussian likelihood")
    ax[1,0].plot(X2_, y_E2, c = "blue" , linewidth = 3, label = "Evidential regression")
    # ax[1,0].legend(loc = "upper left")

    ax[1,1].plot(X2_, s_G2, c = "brown", linewidth = 3, label = "Gaussian likelihood")
    ax[1,1].plot(X2_, s_E2, c = "blue" , linewidth = 3, label = "Evidential regression")
    ax[1,1].plot(X2_, ss2 , c = "black" , linewidth = 3, label = "True std. dev", linestyle = "--")

    # ax[1,1].legend()

    #========================#
    ax[2,0].scatter(_X3, _y3, c = "black", s = 1, label = "Data")
    ax[2,0].plot(X3_, y3_, c = "red", linewidth = 3, label = "True function")
    ax[2,0].plot(X3_, y_G3, c = "brown", linewidth = 3, label = "Gaussian likelihood")
    ax[2,0].plot(X3_, y_E3, c = "blue" , linewidth = 3, label = "Evidential regression")
    # ax[2,0].legend(loc = "upper left")

    ax[2,1].plot(X3_, s_G3, c = "brown", linewidth = 3, label = "Gaussian likelihood")
    ax[2,1].plot(X3_, s_E3, c = "blue" , linewidth = 3, label = "Evidential regression")
    ax[2,1].plot(X3_, ss3 , c = "black" , linewidth = 3, label = "True std. dev", linestyle = "--")
    # ax[2,1].legend()

    #========================#
    fig.tight_layout()
    plt.savefig("exp1.png")

    return

#===================================================#

def experiment2a():
    s = 0.10
    N = 1000

    d1 = 1
    d2 = 5
    d3 = 10

    n_est = 100
    _X, _y = makedata1D(N, s)
    X_, y_ = makedata1D(N, s, train=False)

    model1 = train_RF(_X, _y, n_est, d1)
    model2 = train_RF(_X, _y, n_est, d2)
    model3 = train_RF(_X, _y, n_est, d3)

    y1_, _ = predict_RF(model1, X_)
    y2_, _ = predict_RF(model2, X_)
    y3_, _ = predict_RF(model3, X_)

    # FIGURE
    fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize = (12, 4))
    
    for i in range(3):
        ax[i].set_xlabel("x")

    ax[0].set_ylabel("f(x)")
    ax[0].set_title(f"depth = {d1}")
    ax[1].set_title(f"depth = {d2}")
    ax[2].set_title(f"depth = {d3}")

    #========================#
    ax[0].scatter(_X, _y, c = "black", s = 1, label = "Data")
    ax[0].plot(X_, y_, c = "red", linewidth = 3, label = "True function")
    ax[0].plot(X_, y1_, c = "blue", linewidth = 3, label = "Random forest")

    ax[0].legend()
    #========================#
    ax[1].scatter(_X, _y, c = "black", s = 1, label = "Data")
    ax[1].plot(X_, y_, c = "red", linewidth = 3, label = "True function")
    ax[1].plot(X_, y2_, c = "blue", linewidth = 3, label = "Random forest")

    # ax[1].legend()
    #========================#
    ax[2].scatter(_X, _y, c = "black", s = 1, label = "Data")
    ax[2].plot(X_, y_, c = "red", linewidth = 3, label = "True function")
    ax[2].plot(X_, y3_, c = "blue", linewidth = 3, label = "Random forest")

    # ax[2].legend()
    #========================#
    fig.tight_layout()
    plt.savefig("exp2a.png")


    return

#========================#

def experiment2b():

    # experiment hyperparameters
    H = 4
    N = 1000

    n_est = 100
    depth = 5

    s1 = 0.05
    s2 = 0.10
    s3 = 0.20

    # make data
    _X1, _y1 = makedata1D(N, s1)
    _X2, _y2 = makedata1D(N, s2)
    _X3, _y3 = makedata1D(N, s3)

    X1_, y1_ = makedata1D(N, s1, train=False, ood=True)
    X2_, y2_ = makedata1D(N, s2, train=False, ood=True)
    X3_, y3_ = makedata1D(N, s3, train=False, ood=True)

    # train
    model_E1 = train_EV(_X1, _y1, H)
    model_E2 = train_EV(_X2, _y2, H)
    model_E3 = train_EV(_X3, _y3, H)
    
    model_S1 = train_SB(_X1, _y1, H)
    model_S2 = train_SB(_X2, _y2, H)
    model_S3 = train_SB(_X3, _y3, H)
    
    model_R1 = train_RF(_X1, _y1, n_est, depth)
    model_R2 = train_RF(_X2, _y2, n_est, depth)
    model_R3 = train_RF(_X3, _y3, n_est, depth)
    
    # predict
    y_E1, _, v_E1 = predict_EV(model_E1, X1_)
    y_E2, _, v_E2 = predict_EV(model_E2, X2_)
    y_E3, _, v_E3 = predict_EV(model_E3, X3_)

    y_S1, v_S1 = predict_SB(model_S1, X1_)
    y_S2, v_S2 = predict_SB(model_S2, X2_)
    y_S3, v_S3 = predict_SB(model_S3, X3_)

    y_R1, v_R1 = predict_RF(model_R1, X1_)
    y_R2, v_R2 = predict_RF(model_R2, X2_)
    y_R3, v_R3 = predict_RF(model_R3, X3_)

    # FIGURE
    fig, ax = plt.subplots(ncols = 2, nrows = 3, figsize = (12, 8))

    for i in range(3):
        ax[i,0].set_ylabel(f"Case {i+1}")
        ax[i,1].set_ylabel("Var")
        for j in range(2):
            ax[i, j].axvspan(1, 1.5, alpha = 0.25, color = "gray")
    
    ax[0,0].set_title("Prediction")
    ax[0,1].set_title("Epistemic uncertainty")
    
    ax[2,0].set_xlabel("x")
    ax[2,1].set_xlabel("x")

    #========================#
    ax[0,0].scatter(_X1, _y1, c = "black", s = 1, label = "Data")
    ax[0,0].plot(X1_, y1_, c = "red", linewidth = 3, label = "True function")

    ax[0,0].plot(X1_, y_R1, c = "green" , linewidth = 3, label = "Random forest")
    ax[0,0].plot(X1_, y_E1, c = "blue"  , linewidth = 3, label = "Evidential regression")
    ax[0,0].plot(X1_, y_S1, c = "orange", linewidth = 3, label = "Deep ensemble")
    ax[0,0].legend(loc = "upper left")
    
    ax[0,1].plot(X1_, v_R1, c = "green" , linewidth = 3, label = "Random forest")
    ax[0,1].plot(X1_, v_E1, c = "blue"  , linewidth = 3, label = "Evidential regression")
    ax[0,1].plot(X1_, v_S1, c = "orange", linewidth = 3, label = "Deep ensemble")
    ax[0,1].legend()

    #========================#
    ax[1,0].scatter(_X2, _y2, c = "black", s = 1, label = "Data")
    ax[1,0].plot(X2_, y2_, c = "red", linewidth = 3, label = "True function")

    ax[1,0].plot(X2_, y_R2, c = "green" , linewidth = 3, label = "Random forest")
    ax[1,0].plot(X2_, y_E2, c = "blue"  , linewidth = 3, label = "Evidential regression")
    ax[1,0].plot(X1_, y_S2, c = "orange", linewidth = 3, label = "Deep ensemble")
    # ax[1,0].legend(loc = "upper left")
    
    ax[1,1].plot(X2_, v_R2, c = "green" , linewidth = 3, label = "Random forest")
    ax[1,1].plot(X2_, v_E2, c = "blue"  , linewidth = 3, label = "Evidential regression")
    ax[1,1].plot(X2_, v_S2, c = "orange", linewidth = 3, label = "Deep ensemble")
    # ax[1,1].legend()
    
    #========================#
    ax[2,0].scatter(_X3, _y3, c = "black", s = 1, label = "Data")
    ax[2,0].plot(X3_, y3_, c = "red", linewidth = 3, label = "True function")

    ax[2,0].plot(X3_, y_R3, c = "green" , linewidth = 3, label = "Random forest")
    ax[2,0].plot(X3_, y_E3, c = "blue"  , linewidth = 3, label = "Evidential regression")
    ax[2,0].plot(X3_, y_S3, c = "orange", linewidth = 3, label = "Deep ensemble")
    # ax[2,0].legend(loc = "upper left")
    
    ax[2,1].plot(X3_, v_R3, c = "green" , linewidth = 3, label = "Random forest")
    ax[2,1].plot(X3_, v_E3, c = "blue"  , linewidth = 3, label = "Evidential regression")
    ax[2,1].plot(X3_, v_S3, c = "orange", linewidth = 3, label = "Deep ensemble")
    # ax[2,1].legend()

    #========================#
    fig.tight_layout()
    plt.savefig("exp2b.png")

    return

#========================#
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def experiment3():
# if __name__ == "__main__":

    # HOUSING
    housing = fetch_california_housing()
    
    # YACHT
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
    column_names = ['Longitudinal Position', 'Prismatic Coefficient', 'Length-displacement Ratio','Beam-draught Ratio', 'Length-beam Ratio', 'Froude Number', 'Residuary Resistance']
    yacht = pd.read_csv(url, names=column_names, delim_whitespace=True)
    
    # WINE
    red_wine_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
    white_wine_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
    wine = pd.concat([red_wine_data, white_wine_data], ignore_index=True)
    
    # HEART
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    heart = pd.read_csv(url, names=column_names, na_values="?")
    heart.dropna(inplace=True)

    ##################
    # choose dataset
    ##################

    # Choose dataset
    X1, y1 = np.array(housing.data)     , np.array(housing.target)
    X2, y2 = np.array(yacht.iloc[:,:-1]), np.array(yacht.iloc[:, -1].astype(float))
    X3, y3 = np.array(wine.iloc[:,:-1] ), np.array(wine.iloc[:, -1].astype(float))
    X4, y4 = np.array(heart.iloc[:,:-1]), np.array(heart.iloc[:, -1].astype(float))
    
    # Normalize
    X1 = StandardScaler().fit_transform(X1)
    X2 = StandardScaler().fit_transform(X2)
    X3 = StandardScaler().fit_transform(X3)
    
    y1 = StandardScaler().fit_transform(y1.reshape(-1,1))
    y2 = StandardScaler().fit_transform(y2.reshape(-1,1))
    y3 = StandardScaler().fit_transform(y3.reshape(-1,1))
    y4 = StandardScaler().fit_transform(y4.reshape(-1,1))
    
    _X1, X1_, _y1, y1_ = train_test_split(X1, y1, test_size=0.2, random_state=42)
    _X2, X2_, _y2, y2_ = train_test_split(X2, y2, test_size=0.2, random_state=42)
    _X3, X3_, _y3, y3_ = train_test_split(X3, y3, test_size=0.2, random_state=42)
    _X4, X4_, _y4, y4_ = train_test_split(X4, y4, test_size=0.2, random_state=42)

    ##################
    # choose dataset
    ##################

    H = 4
    n_est = 100
    depth = 10
    
    # TRAIN
    model_S1 = train_SB(_X1, _y1, H)
    model_S2 = train_SB(_X2, _y2, H, batch_size=32)
    model_S3 = train_SB(_X3, _y3, H)
    model_S4 = train_SB(_X4, _y4, H, batch_size=32)
    
    model_G1 = train_GL(_X1, _y1, H)
    model_G2 = train_GL(_X2, _y2, H, batch_size=32)
    model_G3 = train_GL(_X3, _y3, H)
    model_G4 = train_GL(_X4, _y4, H, batch_size=32)
    
    model_R1 = train_RF(_X1, _y1, n_est, depth)
    model_R2 = train_RF(_X2, _y2, n_est, depth)
    model_R3 = train_RF(_X3, _y3, n_est, depth)
    model_R4 = train_RF(_X4, _y4, n_est, depth)

    model_E1 = train_EV(_X1, _y1, H)
    model_E2 = train_EV(_X2, _y2, H, batch_size=32)
    model_E3 = train_EV(_X3, _y3, H)
    model_E4 = train_EV(_X4, _y4, H, batch_size=32)
    
    # EVALUATE
    y_S1, v_S1 = predict_SB(model_S1, X1)
    y_S2, v_S2 = predict_SB(model_S2, X2)
    y_S3, v_S3 = predict_SB(model_S3, X3)
    y_S4, v_S4 = predict_SB(model_S4, X4)
    
    y_G1, s_G1 = predict_GL(model_G1, X1)
    y_G2, s_G2 = predict_GL(model_G2, X2)
    y_G3, s_G3 = predict_GL(model_G3, X3)
    y_G4, s_G4 = predict_GL(model_G4, X4)
    
    y_R1, v_R1 = predict_RF(model_R1, X1)
    y_R2, v_R2 = predict_RF(model_R2, X2)
    y_R3, v_R3 = predict_RF(model_R3, X3)
    y_R4, v_R4 = predict_RF(model_R4, X4)
    
    y_E1, s_E1, v_E1 = predict_EV(model_E1, X1)
    y_E2, s_E2, v_E2 = predict_EV(model_E2, X2)
    y_E3, s_E3, v_E3 = predict_EV(model_E3, X3)
    y_E4, s_E4, v_E4 = predict_EV(model_E4, X4)
    
    # model fit
    MSE_S1 = edl.losses.MSE(y_S1, y1)
    MSE_S2 = edl.losses.MSE(y_S2, y2)
    MSE_S3 = edl.losses.MSE(y_S3, y3)
    MSE_S4 = edl.losses.MSE(y_S4, y4)
    
    MSE_G1 = edl.losses.MSE(y_G1, y1)
    MSE_G2 = edl.losses.MSE(y_G2, y2)
    MSE_G3 = edl.losses.MSE(y_G3, y3)
    MSE_G4 = edl.losses.MSE(y_G4, y4)
    
    MSE_R1 = edl.losses.MSE(y_R1, y1)
    MSE_R2 = edl.losses.MSE(y_R2, y2)
    MSE_R3 = edl.losses.MSE(y_R3, y3)
    MSE_R4 = edl.losses.MSE(y_R4, y4)
    
    MSE_E1 = edl.losses.MSE(y_E1, y1)
    MSE_E2 = edl.losses.MSE(y_E2, y2)
    MSE_E3 = edl.losses.MSE(y_E3, y3)
    MSE_E4 = edl.losses.MSE(y_E4, y4)
    
    # convert s
    s_G1 = np.array(s_G1).reshape(-1)
    s_G2 = np.array(s_G2).reshape(-1)
    s_G3 = np.array(s_G3).reshape(-1)
    s_G4 = np.array(s_G4).reshape(-1)
    
    s_E1 = np.array(s_E1).reshape(-1)
    s_E2 = np.array(s_E2).reshape(-1)
    s_E3 = np.array(s_E3).reshape(-1)
    s_E4 = np.array(s_E4).reshape(-1)
    
    # convert v
    v_S1 = np.array(v_S1).reshape(-1)
    v_S2 = np.array(v_S2).reshape(-1)
    v_S3 = np.array(v_S3).reshape(-1)
    v_S4 = np.array(v_S4).reshape(-1)
    
    v_R1 = np.array(v_R1).reshape(-1)
    v_R2 = np.array(v_R2).reshape(-1)
    v_R3 = np.array(v_R3).reshape(-1)
    v_R4 = np.array(v_R4).reshape(-1)
    
    v_E1 = np.array(v_E1).reshape(-1)
    v_E2 = np.array(v_E2).reshape(-1)
    v_E3 = np.array(v_E3).reshape(-1)
    v_E4 = np.array(v_E4).reshape(-1)

    print(f"Dataset 1:")
    print(f"Ensemble, \t Gaussian MLE \t Random Forest \t Evidential Regression")
    print(f"{MSE_S1}, \t {MSE_G1}, \t {MSE_R1}, \t {MSE_E1}")
    print(f"{0}, \t {np.mean(s_G1)}, \t {0}, \t {np.mean(s_E1)}")
    print(f"{np.mean(v_S1)}, \t {0}, \t {np.mean(v_R1)}, \t {np.mean(v_E1)}")

    print(f"Dataset 2:")
    print(f"Ensemble, \t Gaussian MLE \t Random Forest \t Evidential Regression")
    print(f"{MSE_S1}, \t {MSE_G1}, \t {MSE_R1}, \t {MSE_E1}")
    print(f"{0}, \t {np.mean(s_G2)}, \t {0}, \t {np.mean(s_E2)}")
    print(f"{np.mean(v_S2)}, \t {0}, \t {np.mean(v_R2)}, \t {np.mean(v_E2)}")

    print(f"Dataset 3:")
    print(f"Ensemble, \t Gaussian MLE \t Random Forest \t Evidential Regression")
    print(f"{MSE_S1}, \t {MSE_G1}, \t {MSE_R1}, \t {MSE_E1}")
    print(f"{0}, \t {np.mean(s_G3)}, \t {0}, \t {np.mean(s_E3)}")
    print(f"{np.mean(v_S3)}, \t {0}, \t {np.mean(v_R3)}, \t {np.mean(v_E3)}")

    print(f"Dataset 4:")
    print(f"Ensemble, \t Gaussian MLE \t Random Forest \t Evidential Regression")
    print(f"{MSE_S4}, \t {MSE_G4}, \t {MSE_R4}, \t {MSE_E4}")
    print(f"{0}, \t {np.mean(s_G4)}, \t {0}, \t {np.mean(s_E4)}")
    print(f"{np.mean(v_S4)}, \t {0}, \t {np.mean(v_R4)}, \t {np.mean(v_E4)}")

    # FIGURE
    fig, ax = plt.subplots(ncols = 4, nrows = 2, figsize = (12, 8))

    d_names = ["California housing dataset", "Yacht dataset", "Wine quality dataset", "Heart disease dataset"]
    m_names = ["Deep ensemble", "Gaussian MLE", "Random forest", "Evidential regression"]
    colors = ['red', 'blue', 'green', 'black']

    for i in range(4):
        ax[0,i].set_title(d_names[i])

    for i in range(4):
        ax[0,i].set_xlabel("Aleatoric uncertainty (std. dev)")
        ax[1,i].set_xlabel("Epistemic uncertainty (var)")

    s_names = m_names[1], m_names[3]
    v_names = m_names[0], m_names[2], m_names[3]

    s_colors = colors[1], colors[3]
    v_colors = colors[0], colors[2], colors[3]

    ax[0,0].hist([s_G1, s_E1], range=(0,1  ), bins=30, color=s_colors, label=s_names, density=True)
    ax[0,1].hist([s_G2, s_E2], range=(0,0.1), bins=30, color=s_colors, label=s_names, density=True)
    ax[0,2].hist([s_G3, s_E3], range=(0,1  ), bins=30, color=s_colors, label=s_names, density=True)
    ax[0,3].hist([s_G4, s_E4], range=(0,1.2), bins=30, color=s_colors, label=s_names, density=True)

    ax[1,0].hist([v_S1, v_R1, v_E1], range=(0, 0.5  ), bins=30, color=v_colors, label=v_names, density=True)
    ax[1,1].hist([v_S2, v_R2, v_E2], range=(0, 0.002), bins=30, color=v_colors, label=v_names, density=True)
    ax[1,2].hist([v_S3, v_R3, v_E3], range=(0, 0.5  ), bins=30, color=v_colors, label=v_names, density=True)
    ax[1,3].hist([v_S4, v_R4, v_E4], range=(0, 0.5  ), bins=30, color=v_colors, label=v_names, density=True)

    ax[0,0].legend(loc = "upper right")
    ax[1,0].legend(loc = "upper right")

    #========================#
    fig.tight_layout()
    plt.savefig("exp3.png")

    # pass
    return

##################
if __name__ == "__main__":
    experiment1()
    # experiment2a()
    # experiment2b()
    # experiment3()
