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

def train_DN(X_train, y_train, H, W=64, act="relu"):
    hidden = [tf.keras.layers.Dense(W, activation = act) for _ in range(H)]
    model = tf.keras.Sequential([
        *hidden,
        edl.layers.Dense(1),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss=edl.losses.MSE)
    model.fit(X_train, y_train, batch_size=100, epochs=500)
    return model

def predict_DN(model, X):
    return model(X)

##################
# Ensemble
##################

def train_SB(X_train, y_train, H, W=64, act="relu", E=5):
    models = [
        train_DN(X_train, y_train, H, W, act) for _ in range(E)
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

def train_GL(X_train, y_train, H, W = 64, act = "relu"):

    hidden = [tf.keras.layers.Dense(W, activation = act) for _ in range(H)]
    model = tf.keras.Sequential([
        *hidden,
        edl.layers.DenseNormal(1),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss=GL_loss)
    model.fit(X_train, y_train, batch_size=100, epochs=500)
    return model

def predict_GL(model, X):
    mu, sigma = tf.split(model(X), 2, axis=-1)
    return mu, sigma

##################
# EV
##################

def EV_loss(y, out):
    return edl.losses.EvidentialRegression(y, out, coeff=1e-5)

def train_EV(X_train, y_train, H, W = 64, act = "relu"):
    hidden = [tf.keras.layers.Dense(W, activation = act) for _ in range(H)]
    model = tf.keras.Sequential([
        *hidden,
        edl.layers.DenseNormalGamma(1),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss=EV_loss)
    model.fit(X_train, y_train, batch_size=100, epochs=500)
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
    x = np.linspace(-1, 1, n) if ood else np.linspace(-1, 1.25, n)
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
    model_E1 = train_EV(_X1, _y1, H)
    model_E2 = train_EV(_X2, _y2, H)
    model_E3 = train_EV(_X3, _y3, H)

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

def experiment3():
    pass

##################
if __name__ == "__main__":
    # experiment1()
    # experiment2a()
    experiment2b()
    experiment3()
