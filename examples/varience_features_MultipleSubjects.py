import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import gridspec
import mplcyberpunk
plt.style.use("cyberpunk")
mplcyberpunk.add_glow_effects()
import seaborn as sns
import varname

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, f1_score
import pyneuromodulation.label_normalization as label_norm
from scipy.signal import find_peaks
from imblearn.datasets import make_imbalance
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

task = 'Classification' # | "Regression"
feature_level = "Channels" # | "Features"
classification_linearModel = linear_model.LogisticRegression
Regression_linearModel = LinearRegression

# Helper Functions

def oversample(X, y):
    oversample = RandomOverSampler(sampling_strategy='minority')
    X_over, y_over = oversample.fit_resample(np.hstack([X,y]),(y_train>0))
    print(Counter(y_over))
    return X_over[:,:-1],X_over[:,-1]

def undersample(X, y):
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_under, y_under = undersample.fit_resample(np.hstack([X,y]),(y>0))
    print(Counter(y_under))
    return X_under[:,:-1],X_under[:,-1]

def make_time_lag(dataframe, feature_name, n):
    for i in range(n):
        dataframe.loc[:,"shift_" + str(i + 1)] = dataframe.loc[:, feature_name].shift(i + 1)
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

def plot_test(model, X_test,y_test,k):
    pred = model.predict(X_test)
    fig = plt.figure(figsize=(12,6))
    spec = gridspec.GridSpec(ncols=1, nrows=3,height_ratios=[4,0.5, 0.5])
    ax0 = fig.add_subplot(spec[0])
    ax0.plot(pred, label='prediction')
    ax0.plot(y_test, label='ground truth', alpha = 0.5)
    ax0.legend(title='Rotation',bbox_to_anchor=(1.0, 1), loc='upper left')
    ax0.title.set_text("Test Prediction")
    ax1 = fig.add_subplot(spec[1])
    rugdata =  pd.DataFrame((y_test == pred[:,np.newaxis])[:,0]).iloc[:,0]
    g = sns.rugplot(ax = ax1, x=rugdata.index, hue= rugdata, height=1, lw=1, palette = ["yellow","black"], legend=False, clip_on = False)
    ax1.title.set_text("Incorrect prediction")
    ax1.set_yticks([])
    ax2 = fig.add_subplot(spec[2])
    ax2.plot(X_test[:,0], color = 'orange')
    ax2.title.set_text(f"{k} Featrues")
    ax2.set_ylabel("microvolt")
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
    model_coeffs = {}
    for k, model in models.items():
        model_coeffs[k] = model.coef_[0]
    model_coeffs = pd.DataFrame.from_dict(model_coeffs)
    return model_coeffs


def plot_modelCoeffs(model_coeffs, name = "Linear regression coefficients " ):
    # plotting the model coefficients
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(model_coeffs.transpose(), annot=True, cmap="coolwarm")
    plt.xlabel("time step")
    plt.ylabel('Feature')
    ax.invert_xaxis()
    plt.title(name)
    plt.show()

def get_chNumbers(df):
    ch_nr = []
    for name in df.columns:
        if len(name.split('_')) >5:
            ch_nr.append(name.split('_')[2])
    return list(dict.fromkeys(ch_nr))


def seperate_channels(df):
    chs = get_chNumbers(df)
    data = {}
    for ch in chs:
        data[f'channel_{ch}'] = df.filter(regex = str(ch))
    return data

def seperate_features(df):
    cols = df.columns
    data = {}
    for i, colname in enumerate(cols):
        print(colname)
        if colname.startswith("ECOG"):
            k = ("_").join([colname.split("_")[1], colname.split("_")[2], colname.split("_")[-1]])
            print(k, " : ", i)
            data[k] = df.loc[:, colname]
    return data


# Reading the data file
subjects = ['sub-002']
sessions = ['ses-EphysMedOff01','ses-EphysMedOff02','ses-EphysMedOff03']
tasks = ['task-SelfpacedRotationR','task-SelfpacedRotationL']
stims = ['acq-StimOff', 'acq-StimOn']
runs = ['run-01']

csv_name = "sub-002_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg_FEATURES.csv"
csv_path = r"D:\Jupyter notebooks\Interventional Cognitive Neuromodulation\data\derivatives\sub-002_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg"
PATH = r"D:\Jupyter notebooks\Interventional Cognitive Neuromodulation\data\derivatives"

("_").join([subjects[0],sessions[0],tasks[0],stims[0],runs[0],"ieeg","FEATURES.csv"]) == csv_name

filenames =  os.listdir(PATH)
filenames[0].split("_")

# TODO ask Timon about joining different sessions together in the same continous time series data file
Subjects_results = {}
for sub in subjects:
    for sess in sessions:
        for sub_task in tasks:
            for stim in stims:

                folder_name = "_".join([sub,sess,sub_task,stim,runs[0],"ieeg"])
                print(folder_name)
                if folder_name in filenames:
                    print("Folder exists")

                    features_file = os.path.join(PATH, folder_name,folder_name+"_FEATURES.csv")
                    df = pd.read_csv(os.path.join(csv_path, csv_name))

                    # Extracting ECOG features (for training models on single features)
                    if feature_level == "Feature":
                        data = seperate_features(df)
                    elif feature_level == "Channels":
                        data = seperate_channels(df)


                    # Plotting the features time series
                    # data_df = pd.DataFrame.from_dict(data)
                    # ax = sns.heatmap(data_df.T, annot=False)
                    # plt.xlabel("time step")
                    # plt.ylabel('Feature')
                    # plt.title("features dataframe")
                    # plt.show()

                    y = df.iloc[:, -1].values[:, np.newaxis]
                    # Correcting the label baseling
                    neg_y = -y
                    y_baseline_corrected = baseline_corrected(data=neg_y, param=1e3, thr=5.2e-1)

                    # checking if the corrected baseline peaks are the same number as the original
                    y_baseline_corrected.shape == y[:, 0].shape

                    # Oversampling Vs Undersampling
                    thr = 0.0
                    results_overVSunder = {}
                    models_overVSunder = {}
                    for k, v in data.items():
                        print(k)
                        X = v
                        if type(v) != pd.core.frame.DataFrame:
                            X = pd.Series.to_frame(v)
                            X.columns = [k]

                        X_train, X_test, y_train, y_test = train_test_split(X, y_baseline_corrected, test_size=0.2,shuffle=False)
                        n = 5
                        X_train = make_time_lag(X_train, k, n)
                        y_train = y_train[n:, np.newaxis]

                        if task == "Classification":
                            y_train = y_train > thr

                        print(X_train.shape)
                        print(y_train.shape)
                        X_over, y_over = oversample(X_train, y_train)
                        X_under, y_under = undersample(X_train, y_train)

                        print(task)
                        if task == "Regression":
                            model_original = Regression_linearModel()
                            model_over = Regression_linearModel()
                            model_under = Regression_linearModel()
                            metric = r2_score
                            model_original.fit(X_train, y_train)
                            model_over.fit(X_over, y_over)
                            model_under.fit(X_under, y_under)

                        elif task == "Classification":
                            model_original = classification_linearModel()
                            model_over = classification_linearModel()
                            model_under = classification_linearModel()
                            y_test = y_test > thr
                            metric = accuracy_score
                            model_original.fit(X_train, y_train)
                            model_over.fit(X_over, y_over)
                            model_under.fit(X_under, y_under)

                        X_test = make_time_lag(X_test, k, n)
                        y_test = y_test[n:, np.newaxis]
                        # X_test = np.expand_dims(X_test,1)
                        strategy_dict = {}
                        strategy_dict["Original"] = metric(y_test, model_original.predict(X_test))
                        strategy_dict["Oversampling"] = metric(y_test, model_over.predict(X_test))
                        strategy_dict["Undersampling"] = metric(y_test, model_under.predict(X_test))
                        results_overVSunder[k] = strategy_dict

                        model_dict = {}
                        model_dict["Original"] = model_original
                        model_dict["Oversampling"] = model_over
                        model_dict["Undersampling"] = model_under
                        models_overVSunder[k] = model_dict
                        # plot_test(model_original, X_test, y_test,
                        #           k + "\n No resampling strategy \n Acc " + str(strategy_dict["Original"]))
                        # plot_test(model_under, X_test, y_test,
                        #           k + "\n Oversampling strategy \n" + str(strategy_dict["Oversampling"]))
                        # plot_test(model_over, X_test, y_test,
                        #           k + "\n Undersampling strategy \n" + str(strategy_dict["Undersampling"]))

                    models_overVSunder.keys()
                    model_coeffs = get_modelCoeffs(models_overVSunder[k])

                    results_overVSunder_df = pd.DataFrame.from_dict(results_overVSunder).T
                    Subjects_results[str(sub + sess + "_" + sub_task[-1])] = [model_dict, results_overVSunder_df]



                    plot_modelCoeffs(model_coeffs,name="Classification models coefficients \n for All Beta band variance features")

                    plt.figure(figsize=(10, 6))
                    results_overVSunder_df.plot.bar(rot=0, alpha=0.5, linewidth=1, edgecolor='#08F7FE')
                    plt.title(f"{sub} Oversampling Vs Undersampling strategy for {task}")
                    plt.xticks(rotation=-30)
                    plt.show()

            else:
                print("Folder does not exist")



# results_overVSunder_df
