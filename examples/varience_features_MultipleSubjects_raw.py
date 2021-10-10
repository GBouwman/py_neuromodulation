import os
import keras
import pandas as pd
import numpy as np
import sklearn
from tqdm import tqdm
import mne
import matplotlib.pyplot as plt
import mplcyberpunk
plt.style.use("cyberpunk")
mplcyberpunk.add_glow_effects()
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import class_weight
from sklearn.metrics import r2_score, accuracy_score, f1_score
import tensorflow as tf
from scipy import signal
from UTILS.Utils import *
import UTILS.IO as IO
from UTILS.archetictures.Archs import *
from UTILS.archetictures.DeepLearning_base import *
from UTILS import preprocessing, utils_DL as utils
from keras.callbacks import ModelCheckpoint

task = 'Classification' # | "Regression"
feature_level = "Channels" # | "Features"

batch_size = 64
n_timelag = 127

classification_Models = [
                         Tabnet(name = "Tabnet", num_features=n_timelag+1, feature_dim=n_timelag+1, num_decision_steps=6, batch_size=batch_size,virtual_batch_size=batch_size),
                         # Resnet(name = 'Resnet', input_shape=(n_timelag+1,1)),
                         MLP(name = 'MLP', input_shape=(n_timelag+1,1)),
                         # CNN1D(name = "CNN1D_ks64",input_shape = (n_timelag+1,1)),
                         # CNN1D(name = "CNN1D_ks64_batchnorm", batch_norm = True,input_shape = (n_timelag+1,1)),
                         # CNN1D(input_shape = (n_timelag+1,1),name = "CNN1D_ks3", n_conv=4, conv_units=[32,64, 64, 128], conv_act=ReLU
                         #     , kernel_sizes=[3, 3, 3, 3]
                         #     , pool_func=MaxPool1D, pool=[0, 1, 1, 1]
                         #     , n_dense=2, dense_units=[200, 120], dense_activation=ReLU
                         #     , batch_norm=True),
                         # LSTM1D(name = "LSTM1D", input_shape=(n_timelag+1,1)),
                         ]

# Reading the data file
subjects = ['sub-002'] #, 'sub-003', 'sub-004']
sessions = ['ses-EphysMedOff01'] #,'ses-EphysMedOff02','ses-EphysMedOff03'] # 'ses-EphysMedOn03', 'ses-EphysMedOn01']
tasks = ['task-SelfpacedRotationR']#,'task-SelfpacedRotationL']
stims = ['acq-StimOff'] #, 'acq-StimOn']
runs = ['run-01']

PATH = "D:\Jupyter notebooks\Interventional Cognitive Neuromodulation\data\BIDS Berlin\Raw"

filenames =  os.listdir(PATH)

vhdr_files = IO.get_all_vhdr_files(PATH)
vhdr_files = list(map(lambda x:x.split("\\")[-1],vhdr_files))

Subjects_results = {}
Subjects_dict = {}
for sub in tqdm(subjects):
    for sess in sessions:
        for sub_task in tasks:
            for stim in stims:

                file_name = "_".join([sub,sess,sub_task,stim,runs[0],"ieeg.vhdr"])
                print(file_name)
                if file_name in vhdr_files:
                    print("File exists")
                    # os.path.exists(os.path.join(PATH,sub,sess,'ieeg'))
                    raw = mne.io.brainvision.read_raw_brainvision(os.path.join(PATH,sub,sess,'ieeg',file_name))
                    info = raw.info
                    ch_names = info['ch_names']
                    fs = np.floor(info['sfreq'])
                    raw = raw.get_data()
                    data = raw[list(map(lambda x:x.startswith("ECOG"),ch_names)),]
                    label = raw[ch_names.index("ANALOG_R_ROTA_CH"),]
                    label = label[:,np.newaxis]
                    fs_new = 128

                    data_downsampled = signal.resample(data.T, int(data.T.shape[0] * fs_new / fs), axis=0)
                    label_downsampled = signal.resample(label, int(label.shape[0] * fs_new / fs), axis=0)

                    # Correcting the label baseling
                    neg_y = -label_downsampled
                    if sub in ['sub-004'] and sub_task not in ['task-SelfpacedRotationR']:
                        y_baseline_corrected = baseline_corrected(data=label_downsampled, sess=sess, param=1e4, thr=2.5e-1, distance = 100)
                    else:
                        y_baseline_corrected = baseline_corrected(data=neg_y, sub=sub, sess=sess, param=1e4, thr=2.5e-1, distance = 100)

                    # Oversampling Vs Undersampling
                    thr = 0.0
                    results_dict = {}
                    models_dict = {}
                    feature_dict = {}
                    f1_dict = {}
                    f1_dict_o = {}
                    fold_dict = {}
                    n_fold = 2
                    for ch_indx in range(1):
                        ch_indx = 5
                        print(f"Subject {sub} ,Channel {ch_indx}")
                        cv = KFold(n_splits=n_fold)
                        for fold,(train_index, test_index) in enumerate(cv.split(data_downsampled)):
                            print(f"fold {fold}")
                            X_train, X_test = data_downsampled[train_index, [ch_indx]], data_downsampled[test_index,[ch_indx] ]
                            y_train, y_test = y_baseline_corrected[train_index,np.newaxis], y_baseline_corrected[test_index,np.newaxis]

                        # X_train, X_test, y_train, y_test = train_test_split(data_downsampled[:,[ch_indx]], y_baseline_corrected[:, np.newaxis], test_size=0.2,shuffle=False, random_state=42)
                            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,shuffle=False)



                            X_train_timeLagged = make_time_lag(pd.DataFrame(X_train, columns=[f'ch_{ch_indx}']), n_timelag)
                            y_train_timeLagged = y_train[n_timelag:, :]

                            X_oversampled, y_oversampled = oversample(X_train_timeLagged, y_train_timeLagged)
                            X_oversampled = X_oversampled[:, :, np.newaxis]

                            train_cut = int(np.floor(X_oversampled.shape[0]/batch_size)*batch_size)
                            X_oversampled = X_oversampled[:train_cut]
                            y_oversampled = y_oversampled[:train_cut]

                            X_val_timeLagged = make_time_lag(pd.DataFrame(X_val, columns=[f'ch_{ch_indx}']), n_timelag)
                            X_val_timeLagged = X_val_timeLagged[:, :, np.newaxis]
                            y_val_timeLagged = y_val[n_timelag:, :]

                            val_cut = int(np.floor(X_val_timeLagged.shape[0] / batch_size) * batch_size)
                            X_val_timeLagged = X_val_timeLagged[:val_cut]
                            y_val_timeLagged = y_val_timeLagged[:val_cut]

                            X_test_timeLagged = make_time_lag(pd.DataFrame(X_val, columns=[f'ch_{ch_indx}']), n_timelag)
                            X_test_timeLagged = X_test_timeLagged[:, :, np.newaxis]
                            y_test_timeLagged = y_test[n_timelag:, :]

                            test_cut = int(np.floor(X_test_timeLagged.shape[0] / batch_size) * batch_size)
                            X_test_timeLagged = X_test_timeLagged[:test_cut]
                            y_test_timeLagged = y_test_timeLagged[:test_cut]

                            # TODO edit the Regression if statement so it be like the Classification one
                            if task == "Regression":

                                print("Not Ready Yet")

                            elif task == "Classification":
                                models_list = classification_Models

                            for m in models_list:

                                model = m.build()
                                print(model.name)
                                filepath = f"/tmp/checkpoint/{model.name}/"
                                callback = [tf.keras.callbacks.ModelCheckpoint(filepath,monitor="val_loss",verbose=1,
                                                                                save_best_only=True,
                                                                                save_weights_only=False,
                                                                                mode="min",
                                                                                save_freq="epoch"
                                                                            )]
                                classes = np.unique(y_val_timeLagged>0, return_counts = True)[0]
                                class_weights = np.unique(y_val_timeLagged>0, return_counts = True)[1]
                                weights_dict = dict(zip(classes, class_weights))
                                model.compile(optimizer='rmsprop', loss = k.losses.BinaryCrossentropy(), metrics=['accuracy', k.metrics.Recall(), k.metrics.Precision()])
                                model.fit(X_oversampled,y_oversampled>thr, validation_data = (X_val_timeLagged, y_val_timeLagged>0) ,batch_size=batch_size, epochs= 1000, shuffle= True, callbacks=callback, class_weight=weights_dict)
                                model.load_weights(filepath)
                                prediction = model.predict(X_test_timeLagged, batch_size= batch_size)
                                results_dict[model.name] = accuracy_score(y_true=y_test_timeLagged>0, y_pred= prediction>0.5)
                                f1_dict[model.name] = f1_score(y_true=y_test_timeLagged>0, y_pred= prediction>0.5)
                                # f1_dict_o[model.name] = f1_score(y_true=y_test_timeLagged > 0, y_pred=prediction)
                                plt.plot(y_test_timeLagged > 0, label='Ground Truth', alpha=0.5)
                                plt.plot(prediction, label='Prediction', alpha=0.5)
                                plt.plot(prediction>0.5, label='Prediction', alpha=0.5)
                                plt.legend()
                                plt.title(f"{model.name} test")
                                plt.show()


                                # models_dict[model.name] = model

                            fold_dict[fold] = {"loss": model.evaluate(X_test_timeLagged,y_test_timeLagged>0 ),
                                              "accuracy": results_dict, "f1_score":f1_dict }#, "models":models_dict}
                        feature_dict[f"ch_{ch_indx}"] = fold_dict

                    print(str(sub + sess + "_" + sub_task[-1]))
                    Subjects_dict[str(sub + sess + "_" + sub_task[-1])] = feature_dict

            else:
                print("Folder does not exist")

    Subjects_results[sub] = Subjects_dict


Subjects_results

DeepLearning_results = Subjects_results

np.save('deeplearning_results.npz', DeepLearning_results)



results_df = pd.DataFrame([])
for sub in subjects:
    for sess in sessions:
        for sub_task in tasks:
            for ch in [5]:
                for fold in range(n_fold):
                    for m in models_list:
                        m_name = m.name
                        for sampling in ["Oversampling"]:
                            # if isinstance(Subjects_results[sub][sub+sess+"_"+sub_task[-1]][ch][fold][m_name]['test_score'],dict):
                            if sess+"_"+sub_task[-1] in Subjects_results[sub]:
                                score = DeepLearning_results[sub][sub + sess+"_"+sub_task[-1]][f"ch_{ch}"][fold]['accuracy'][m_name]
                                results_df = results_df.append([pd.DataFrame(columns=['subject','session','task','channel','fold','model','sampling','score'], data=[[sub,sess,sub_task,ch,fold,m_name,sampling,score ]])])
                            else:
                                results_df = results_df


results_df

results_df.columns # ['subject', 'session', 'task', 'channel', 'fold', 'model', 'sampling','score']
processed_df = results_df.groupby(['subject', 'session','task','model', 'fold'], as_index=False).max().\
    groupby(['subject', 'session','task','channel','model'], as_index=False).mean()

processed_df = processed_df[processed_df['task']=='task-SelfpacedRotationR']

deeplearning_df = results_df
deeplearning_df["group"] = 'deeplearning'


plt.figure(figsize=(16,8))
# ax = sns.boxplot(data = processed_df, x = "subject", y = "score", hue = "model")
ax = sns.swarmplot(data = processed_df, x = 'subject', y = 'score', hue = 'model').set_title('Deep Learning Model Performance')
plt.ylim(0.75,1)
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
plt.xlabel("Subject")
plt.ylabel("Accuracy score")
plt.show()


both_df = deeplearning_df.append(classical_df)

both_df

processed_df = both_df
processed_df = processed_df[processed_df['task']=='task-SelfpacedRotationR']
processed_df = processed_df[processed_df['subject']=='sub-002']


processed_df = both_df.groupby(['subject', 'session','task','model', 'fold', "group"], as_index=False).max().\
    groupby(['subject', 'session','task','channel','model', "group"], as_index=False).mean()



plt.figure(figsize=(10,5))
ax = sns.boxplot(data = processed_df, x = "group", y = "score")
ax = sns.scatterplot(data = processed_df, x = 'group', y = 'score', hue = 'model', sizes = (500), style = 'group',
                     x_jitter = 100).set_title('Deep Learning vs Classical machine learning Performance')
plt.ylim(0.75,1)
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
plt.xlabel("Learning Method")
plt.ylabel("Accuracy score")
plt.show()




