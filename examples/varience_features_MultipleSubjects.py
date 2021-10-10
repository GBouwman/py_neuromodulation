import os
import pandas as pd
import numpy as np
import sklearn.neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import matplotlib.pyplot as plt
import mplcyberpunk
plt.style.use("cyberpunk")
mplcyberpunk.add_glow_effects()
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split,  KFold
from sklearn.metrics import r2_score, accuracy_score, f1_score, balanced_accuracy_score

import xgboost
from sklearn.ensemble import RandomForestClassifier
from UTILS.Utils import *

task = 'Classification' # | "Regression"
feature_level = "Channels" # | "Features"
classification_Models = [
                          xgboost.XGBClassifier(),
                          RandomForestClassifier(),
                          linear_model.LogisticRegression(),
                          LinearDiscriminantAnalysis(),
                          LinearDiscriminantAnalysis(solver='lsqr',shrinkage= 'auto'),
                          sklearn.neighbors.KNeighborsClassifier(),
                          sklearn.svm.SVC(),
                          sklearn.ensemble.GradientBoostingClassifier(),
                        ]

# Regression_Models = LinearRegression


# Reading the data file
subjects = ['sub-002', 'sub-003', 'sub-004', 'sub-005']
sessions = ['ses-EphysMedOff01','ses-EphysMedOff02','ses-EphysMedOff03'] # 'ses-EphysMedOn03', 'ses-EphysMedOn01']
tasks = ['task-SelfpacedRotationR'] #,'task-SelfpacedRotationL']
stims = ['acq-StimOff'] #, 'acq-StimOn']
runs = ['run-01']

# csv_name = "sub-002_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg_FEATURES.csv"
# csv_path = r"D:\Jupyter notebooks\Interventional Cognitive Neuromodulation\data\derivatives\sub-002_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg"
PATH = r"D:\Jupyter notebooks\Interventional Cognitive Neuromodulation\data\derivatives"

# ("_").join([subjects[0],sessions[0],tasks[0],stims[0],runs[0],"ieeg","FEATURES.csv"]) == csv_name

filenames =  os.listdir(PATH)

Subjects_results = {}
for sub in subjects:
    session_dict = {}
    for sess in sessions:
        for sub_task in tasks:
            for stim in stims:

                folder_name = "_".join([sub,sess,sub_task,stim,runs[0],"ieeg"])
                print(folder_name)
                if folder_name in filenames:
                    print("Folder exists")

                    features_file = os.path.join(PATH, folder_name,folder_name+"_FEATURES.csv")
                    df = pd.read_csv(os.path.join(PATH, features_file))

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
                    if sub in ['sub-004'] and sub_task not in ['task-SelfpacedRotationR']:
                        y_baseline_corrected = baseline_corrected(data=y, sub=sub, sess=sess, param=1e3, thr=5.2e-1)
                    else:
                        y_baseline_corrected = baseline_corrected(data=neg_y, sub=sub, sess=sess, param=1e3, thr=5.2e-1)

                    # checking if the corrected baseline peaks are the same number as the original
                    y_baseline_corrected.shape == y[:, 0].shape

                    # Oversampling Vs Undersampling
                    thr = 0.0
                    feature_dict = {}
                    # fold_dict = {}
                    # session_dict = {}
                    for k, v in data.items():
                        print(k)
                        X = v
                        if type(v) != pd.core.frame.DataFrame:
                            X = pd.Series.to_frame(v)
                            X.columns = [k]


                        #Cross validation
                        n_folds = 3
                        cv = KFold(n_splits=n_folds)
                        fold_dict = {}
                        for fold,(train_index, test_index) in enumerate(cv.split(X)):
                            print("Crossvalidation fold",fold)
                            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                            y_train, y_test = y_baseline_corrected[train_index], y_baseline_corrected[test_index]

                        # X_train, X_test, y_train, y_test = train_test_split(X, y_baseline_corrected, test_size=0.2,shuffle=False)
                            n = 5
                            X_train = make_time_lag(X_train, n)
                            y_train = y_train[n:, np.newaxis]

                            if task == "Classification":
                                y_train = y_train > thr
                                y_test = y_test > thr

                            print(X_train.shape)
                            print(y_train.shape)
                            # X_over, y_over = oversample(X_train, y_train)
                            # X_under, y_under = undersample(X_train, y_train)

                            print(task)
                            # TODO edit the Regression if statement so it be like the Classification one
                            if task == "Regression":
                                print("Not Ready Yet")

                            elif task == "Classification":
                                models_list = classification_Models

                            fold_dict[fold] = train_samplingstratigies(models_list, X_train,y_train,X_test,y_test, metric=balanced_accuracy_score)
                            # fold_dict[fold] = f"{sub} {sess} {fold}"
                        feature_dict[k] = fold_dict
                    print("finished session: ", str(sub + sess + "_" + sub_task[-1]))
                    session_dict[str(sess + "_" + sub_task[-1])] = feature_dict

                # Subjects_dict[str(sub + sess + "_" + sub_task[-1])] = feature_dict

            # else:
            #     pass
                # print("Folder does not exist")

    Subjects_results[sub] = session_dict








plt.plot(Subjects_results['sub-002']['ses-EphysMedOff01_R']['channel_5'][0]['xgbclassifier']['predictions']['Oversampling'])
plt.show()

session_dict.keys()

Classical_ml_results_backup = Subjects_results

classical_ml_results =  Subjects_results
np.save('classical_ml_resutls.npz', classical_ml_results)

# feature_dict ==> channel ==> model type ===> test_score ==> strategy
#                                              model      ==> strategy

y_true = [0, 1, 0, 0, 0, 0, 0, 0]
y_pred = [0, 0, 0, 0, 0, 1, 0, 0]
balanced_accuracy_score(y_true, y_pred)
accuracy_score(y_true, y_pred)

results_df = pd.DataFrame([])
for sub in subjects:
    for sess in sessions:
        for sub_task in tasks:
            for ch in data.keys():
                for fold in range(n_folds):
                    for m in models_list:
                        m_name = str(m).split(".")[-1].split("'")[0].lower()
                        for sampling in ["Original","Oversampling",'Undersampling']:
                            # if isinstance(Subjects_results[sub][sub+sess+"_"+sub_task[-1]][ch][fold][m_name]['test_score'],dict):
                            if sess+"_"+sub_task[-1] in Subjects_results[sub]:
                                score = Subjects_results[sub][sess+"_"+sub_task[-1]][ch][fold][m_name]['test_score'][sampling]
                                results_df = results_df.append([pd.DataFrame(columns=['subject','session','task','channel','fold','model','sampling','score'], data=[[sub,sess,sub_task,ch,fold,m_name,sampling,score ]])])
                            else:
                                results_df = results_df


results_df
results_df.to_csv('Classical_ML_results.csv')

results_df.columns # ['subject', 'session', 'task', 'channel', 'fold', 'model', 'sampling','score']
processed_df = results_df.groupby(['subject', 'session','task','model', 'fold'], as_index=False).max().\
    groupby(['subject', 'session','task','channel','model'], as_index=False).mean()

processed_df = processed_df[processed_df['task']=='task-SelfpacedRotationR']

import seaborn as sns

ax = sns.catplot(data = results_df.loc[:,['subject','session','task','channel','model','score']],
                 x = 'subject',
                 y = 'score',
                 col = 'model',
                hue = 'channel',
                 )
plt.show()




plt.figure(figsize=(16,8))
ax = sns.barplot(data = processed_df, x = "subject", y = "score", hue = "model")
ax = sns.swarmplot(data = processed_df, x = 'subject', y = 'score', hue = 'model').set_title('Classical Maching Learning Model Performance')
plt.ylim(0.5,1)
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
plt.xlabel("Subject")
plt.ylabel("Accuracy score")
plt.show()

plt.plot(Subjects_results['sub-004']['ses-EphysMedOff01_R']['channel_6'][2]['xgbclassifier']['predictions']['Undersampling'])
plt.plot(y_test)
plt.show()


classical_df = results_df

classical_df['group'] = 'classical'