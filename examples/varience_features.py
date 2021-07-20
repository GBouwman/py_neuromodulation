import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import mplcyberpunk
plt.style.use("cyberpunk")
mplcyberpunk.add_glow_effects()
import seaborn as sns
import varname

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pyneuromodulation.label_normalization as label_norm
from scipy.signal import find_peaks
from imblearn.datasets import make_imbalance
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Helper Functions

def oversample(X, y):
    oversample = RandomOverSampler(sampling_strategy='minority')
    X_over, y_over = oversample.fit_resample(np.hstack([X,y[:,np.newaxis]]),(y_train>0))
    print(Counter(y_over))
    return X_over[:,:-1],X_over[:,-1]

def undersample(X, y):
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_under, y_under = undersample.fit_resample(np.hstack([X,y[:,np.newaxis]]),(y>0))
    print(Counter(y_under))
    return X_under[:,:-1],X_under[:,-1]

def make_time_lag(dataframe, feature_name, n):
    for i in range(n):
        dataframe.loc[:,"shift_" + str(i + 1)] = dataframe.loc[:,feature_name].shift(i + 1)
    return dataframe.values[n:, :]


def baseline_corrected(data, param = 1e3, thr= 2e-1):
    y_baseline_corrected = label_norm.baseline_correction(data[:,0],method='baseline_rope', param=param, thr=thr,normalize=True, decimate=1, niter=10, verbose=True)
    y_baseline_corrected_peaks, _ = find_peaks(y_baseline_corrected[0])
    plt.plot(y_baseline_corrected[0])
    plt.plot(y_baseline_corrected_peaks, y_baseline_corrected[0][y_baseline_corrected_peaks], 'x')
    plt.title(f"parameters {param} \n Threshold {thr} \n Data {varname.argname2('data')}")
    plt.show()
    print(y_baseline_corrected_peaks.shape)
    return y_baseline_corrected[0]

def balance_data(X,y, thr = 0):
    sample_n = (y > thr).sum()
    print(sample_n)
    x = np.hstack([X_train, np.expand_dims(y_train,axis= 1)])
    X_tr, _ = make_imbalance(X= x, y=y >thr, sampling_strategy={0: sample_n, 1: sample_n},random_state=42)
    print(X_tr.shape)
    return X_tr[:,0:-1], X_tr[:,-1]

def plot_test(X_test, reg,k):
    plt.plot(reg.predict(X_test), label = 'prediction')
    plt.plot(y_test, label = 'ground truth')
    plt.title(f"linear regression prediction of {k}")
    plt.legend()
    plt.show()

def plot_rSquared(results):
    # plotting the R squared for each model test
    plt.figure(figsize=(10,8))
    plt.bar(results.keys(),results.values(), alpha = 0.3, edgecolor = '#08F7FE', linewidth = 5, label = "Channel 6")
    plt.title("Linear regression model \n Sub 002 \n Channel 6 \n $R^2$ score ")
    plt.xticks(rotation = -45)
    plt.xlabel("Frequency Band")
    plt.ylabel("$R^2$ score")
    plt.legend()
    mplcyberpunk.add_glow_effects()
    plt.show()

def get_modelCoeffs(models):
    # Rearranging the models coefficients
    model_coeffs = pd.DataFrame([])
    for k,model in models.items():
        model_coeffs[k] = model.coef_
    return model_coeffs

def plot_modelCoeffs(model_coeffs):
    # plotting the model coefficients
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(model_coeffs.transpose(), annot=True, cmap="coolwarm")
    ax.invert_xaxis()
    plt.ylabel("time step")
    plt.xlabel('Feature')
    plt.title("Linear regression coefficients ")
    plt.show()

# Reading the data file
csv_name = "sub-002_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg_FEATURES.csv"
csv_path = r"D:\Jupyter notebooks\Interventional Cognitive Neuromodulation\data\derivatives\sub-002_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg"

df = pd.read_csv(os.path.join(csv_path,csv_name))
cols = df.columns

# Extracting ECOG channel 6 features
data_ch6 = {}
temp_df = pd.DataFrame()
for i,colname in enumerate(cols):
    print(colname)
    if colname.startswith("ECOG_L_6"):
        k = colname.split("_")[-1]
        print(k," : ", i)
        feature_name = "feature"
        n = 0
        temp_df[feature_name] = df.loc[:,colname]
        data_ch6[k] = temp_df

y = df.iloc[:,-1].values[:,np.newaxis]
# Correcting the label baseling
neg_y = -y
y_baseline_corrected = baseline_corrected(data=neg_y,param=1e3,thr = 5.2e-1)

# checking if the corrected baseline peaks are the same number as the original
y_baseline_corrected.shape == y[:,0].shape

# initiating, training and testing linear regression model on each different band features
results = {}
models = {}
for k,v in data_ch6.items():
    print(k)
    X = v
    X_train, X_test, y_train, y_test = train_test_split(X, y_baseline_corrected, test_size=0.1, shuffle=False)
    n = 5
    # X_train.reset_index()
    X_train = make_time_lag(X_train, 'feature', n)
    y_train = y_train[n: ]
    print(X_train.shape)
    print(y_train.shape)
    # X_train = X_train.iloc[:,0].values
    X_train, y_train = balance_data(X_train, y_train)
    print(X_train.shape)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    X_test = make_time_lag(X_test, 'feature', n)
    y_test = y_test[n:,]
    # X_test = np.expand_dims(X_test,1)
    results[k] = r2_score(y_test, reg.predict(X_test))
    # results[k] = accuracy_score(y_test>0, reg.predict(X_test)>0.5)
    models[k] = reg
    plot_test(X_test,reg,k)

plot_rSquared(results)

model_coeffs = get_modelCoeffs(models)

plot_modelCoeffs(model_coeffs)


# Oversampling Vs Undersampling

results_overVSunder = {}
models_overVSunder = {}
for k, v in data_ch6.items():
    print(k)
    X = v
    X_train, X_test, y_train, y_test = train_test_split(X, y_baseline_corrected, test_size=0.2, shuffle=False)
    n = 5
    X_train = make_time_lag(X_train, 'feature', n)
    y_train = y_train[n: ]
    print(X_train.shape)
    print(y_train.shape)
    X_over, y_over = oversample(X_train,y_train)
    X_under, y_under = undersample(X_train,y_train)

    reg_original = LinearRegression()
    reg_over = LinearRegression()
    reg_under = LinearRegression()

    reg_original.fit(X_train,y_train)
    reg_over.fit(X_over,y_over)
    reg_under.fit(X_under,y_under)


    X_test = make_time_lag(X_test, 'feature', n)
    y_test = y_test[n:,]
    # X_test = np.expand_dims(X_test,1)
    strategy_dict = {}
    strategy_dict["Original"] = r2_score(y_test, reg_original.predict(X_test))
    strategy_dict["Oversampling"] = r2_score(y_test, reg_over.predict(X_test))
    strategy_dict["Undersampling"] = r2_score(y_test,reg_under.predict(X_test))
    results_overVSunder[k] = strategy_dict

    model_dict = {}
    model_dict["Original"] = reg_original
    model_dict["Oversampling"] = reg_over
    model_dict["Undersampling"] = reg_under
    models_overVSunder[k] = model_dict



results_overVSunder_df = pd.DataFrame.from_dict(results_overVSunder).T

plt.figure(figsize=(10,6))
results_overVSunder_df.plot.bar( rot = 0, alpha = 0.5, linewidth = 1, edgecolor = '#08F7FE')
plt.title("Oversampling Vs Undersampling strategy")
plt.xticks(rotation = -30)
plt.show()

# results_overVSunder_df
