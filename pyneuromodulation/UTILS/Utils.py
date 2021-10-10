import pandas as pd
import numpy as np
import pyneuromodulation.label_normalization as label_norm
from scipy.signal import find_peaks
from imblearn.datasets import make_imbalance
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
import seaborn as sns
import pprint
import varname
from matplotlib import gridspec
import matplotlib.pyplot as plt
import mplcyberpunk
plt.style.use("cyberpunk")
mplcyberpunk.add_glow_effects()
from sklearn.metrics import r2_score, accuracy_score, f1_score
from sklearn.base import clone





def subjects_df(Subjects_results):
    temp_df = pd.DataFrame(columns=['subject','session','channel', 'model','strategy', 'score'])
    for subject in Subjects_results.keys():
        for session in Subjects_results[subject].keys():
            for channel in Subjects_results[subject][session].keys():
                for model in Subjects_results[subject][session][channel]:
                    # print(model)
                    # print(Subjects_dict[session][channel][model]['test_score'])
                    for strategy, score in Subjects_results[subject][session][channel][model]['test_score'].items():
                        temp_df = temp_df.append({'subject':subject,'session':session,'channel':channel, "model":model, 'strategy': strategy, 'score':score}, ignore_index=True )
    return temp_df

def get_best_score(temp_df):
    return temp_df.loc[temp_df.groupby(by=['subject','session', 'channel', 'model'])['score'].idxmax().values]



# Helper Functions

def oversample(X, y):
    oversample = RandomOverSampler(sampling_strategy='minority')
    X_over, y_over = oversample.fit_resample(np.hstack([X,y]),(y>0))
    print(Counter(y_over))
    return X_over[:,:-1],X_over[:,-1]

def undersample(X, y):
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_under, y_under = undersample.fit_resample(np.hstack([X,y]),(y>0))
    print(Counter(y_under))
    return X_under[:,:-1],X_under[:,-1]

# def make_time_lag(dataframe, feature_name, n):
#     for i in range(n):
#         dataframe.loc[:,"shift_" + str(i + 1)] = dataframe.loc[:, feature_name].shift(i + 1)
#     return dataframe.values[n:, :]


def make_time_lag(df,n):
    df_temp = df
    cols = df_temp.columns
    for i in range(n):
        df_temp1 = df.shift(i)
        df_temp1.columns = f"shift_{i + 1}_" + df.columns
        df_temp = pd.concat([df_temp, df_temp1], axis=1)

    return df_temp.values[n:,:]


def baseline_corrected(data, sub, sess, param = 1e3, thr= 2e-1, distance = None):
    y_baseline_corrected = label_norm.baseline_correction(data[:,0],method='baseline_rope', param=param, thr=thr,normalize=True, decimate=1, niter=10, verbose=True)
    y_baseline_corrected_peaks, _ = find_peaks(y_baseline_corrected[0], distance= distance)
    plt.plot(y_baseline_corrected[0])
    plt.plot(y_baseline_corrected_peaks, y_baseline_corrected[0][y_baseline_corrected_peaks], 'x')
    # plt.suptitle(f"parameters {param} \n Threshold {thr} \n Data {varname.argname2('data')}")
    plt.title(sub + sess + " label")
    plt.show()
    print(y_baseline_corrected_peaks.shape)
    return y_baseline_corrected[0]

def balance_data(X,y, thr = 0):
    sample_n = (y > thr).sum()
    print(sample_n)
    x = np.hstack([X, np.expand_dims(y,axis= 1)])
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


def train_samplingstratigies(models_list , X_train, y_train, X_test, y_test, metric = f1_score, n = 5):

    models_dict = {}
    X_test = make_time_lag(X_test, n)
    y_test = y_test[n:, np.newaxis]
    # X_test = np.expand_dims(X_test,1)
    # print(X_test.shape, y_test.shape)
    X_over, y_over = oversample(X_train, y_train)
    X_under, y_under = undersample(X_train, y_train)
    for model in models_list:
        print(model)
        model_original = clone(model)
        model_over = clone(model)
        model_under = clone(model)
        # metric = accuracy_score
        model_original.fit(X_train, y_train)
        model_over.fit(X_over, y_over)
        model_under.fit(X_under, y_under)
        print("Training finished")

        predicion_dict = {}
        predicion_dict["Original"] = model_original.predict(X_test)
        predicion_dict["Oversampling"] = model_over.predict(X_test)
        predicion_dict["Undersampling"] = model_under.predict(X_test)

        strategy_dict = {}
        strategy_dict["Original"] = metric(y_test, model_original.predict(X_test))
        strategy_dict["Oversampling"] = metric(y_test, model_over.predict(X_test))
        strategy_dict["Undersampling"] = metric(y_test, model_under.predict(X_test))

        model_dict = {}
        model_dict["Original"] = model_original
        model_dict["Oversampling"] = model_over
        model_dict["Undersampling"] = model_under

        temp_dict = {}
        temp_dict['test_score'] = strategy_dict
        temp_dict['predictions'] = predicion_dict
        temp_dict['y_test'] = y_test
        # temp_dict['model'] =  model_dict

        models_dict[str(model).split(".")[-1].split("'")[0].lower()] = temp_dict

    return models_dict


