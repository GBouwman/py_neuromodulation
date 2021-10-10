import os
import matplotlib.pyplot as plt
import mplcyberpunk
plt.style.use("cyberpunk")
mplcyberpunk.add_glow_effects()
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from UTILS.Utils import *
import UTILS.IO as IO, UTILS.label_generator as label_generator
from scipy import signal
from UTILS.archetictures import CNN

task = 'Classification' # | "Regression"
feature_level = "Channels" # | "Features"
# classification_Models = [ xgboost.XGBClassifier, RandomForestClassifier]
# Regression_Models = LinearRegression

classification_Models = [CNN.make_cnnModel(input_shape=(128, 1, 1), summary = True, ch_first=False, output_units=1, output_activation='sigmoid')]
import keras
keras.backend.set_image_data_format('channels_last')

# Reading the data file
subjects = ['sub-000'] #, 'sub-003', 'sub-004']
sessions = ['ses-right']#,'ses-EphysMedOff02','ses-EphysMedOff03'] # 'ses-EphysMedOn03', 'ses-EphysMedOn01']
tasks = ['task-force']#,'task-SelfpacedRotationL']
# stims = ['acq-StimOff'] #, 'acq-StimOn']
runs = ['run-0']

# csv_name = "sub-002_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg_FEATURES.csv"
# csv_path = r"D:\Jupyter notebooks\Interventional Cognitive Neuromodulation\data\derivatives\sub-002_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg"
# PATH = r"D:\Jupyter notebooks\Interventional Cognitive Neuromodulation\data\derivatives_cnt"
#
# ("_").join([subjects[0],sessions[0],tasks[0],stims[0],runs[0],"ieeg","FEATURES.csv"]) == csv_name


PATH = "D:\Jupyter notebooks\Interventional Cognitive Neuromodulation\data"

filenames =  os.listdir(PATH)

vhdr_files = IO.get_all_vhdr_files(PATH)
vhdr_files = list(map(lambda x:x.split("\\")[-1],vhdr_files))

batch_size = 4000
fs_new = 256
fs = 1000

Subjects_results = {}
Subjects_dict = {}
for sub in subjects:
    for sess in sessions:
        for sub_task in tasks:

            file_name = "_".join([sub,sess,sub_task,runs[0],"ieeg.vhdr"])
            print(file_name)
            if file_name in vhdr_files:
                print("File exists")

                f_ = [file for file in vhdr_files if sub in file and sess in file and sub_task in file]

                X, y = IO.get_data_raw_combined_berlin(sub,"right","ECOG",f_, os.path.join(PATH,"BIDS Berlin",'Raw',sub,sess,'ieeg') )

                print(X.shape)
                # resample data to fs_new
                y = signal.resample(y, int(y.shape[0] * fs_new / fs), axis=0)
                X = signal.resample(X, int(X.shape[1] * fs_new / fs), axis=1).T


                # info = raw.info
                # ch_names = info['ch_names']
                # fs = np.floor(info['sfreq'])
                # raw = raw.get_data()
                # # raw, ch_names = IO.read_BIDS_file(os.path.join(PATH,sub,sess,'ieeg',file_name))
                # data = raw[list(map(lambda x:x.startswith("ECOG"),ch_names)),]
                # label = raw[ch_names.index("MOV_RIGHT_CLEAN"),]
                # label = label[:,np.newaxis]
                # fs_new = np.ceil(fs/100)
                #
                # data = signal.resample(data.T, int(data.T.shape[0] * fs_new / fs), axis=0)
                # label_downsampled = signal.resample(label, int(label.shape[0] * fs_new / fs), axis=0)
                #

                # y = df.iloc[:, -1].values[:, np.newaxis]
                # Correcting the label baseling
                # neg_y = -label_downsampled
                # if sub in ['sub-004'] and sub_task not in ['task-SelfpacedRotationR']:
                #     y_baseline_corrected = baseline_corrected(data=label_downsampled,sub = sub, sess=sess, param=1e4, thr=2.5e-1, distance = 10)
                # else:
                #     y_baseline_corrected = baseline_corrected(data=neg_y, sub=sub, sess=sess, param=1e4, thr=2.5e-1, distance = 100)
                #

                plt.plot(y_baseline_corrected)
                plt.plot()

                X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, shuffle=False)

                # Oversampling Vs Undersampling
                thr = 0.1
                results_overVSunder = {}
                models_overVSunder = {}
                feature_dict = {}

                for ch_indx in range(X_train.shape[1]):
                    print(ch_indx)
                    ch_indx = 6
                    # X_train, X_test, y_train, y_test = train_test_split(data[:,ch_indx, np.newaxis], y_baseline_corrected[:, np.newaxis], test_size=0.2,shuffle=False)
                    train_gen = label_generator.generator(X_train[:, [1]], y_train, 4000, 128, rebalance=True, rebalanced_thr=0, task = task)
                    test_gen = label_generator.generator(X_val[:, [1]], y_val, 4000, 128, rebalance=False, rebalanced_thr=0, task = task)

                    # TODO edit the Regression if statement so it be like the Classification one
                    if task == "Regression":
                        # model_original = Regression_Models()
                        # model_over = Regression_Models()
                        # model_under = Regression_Models()
                        # metric = r2_score
                        # model_original.fit(X_train, y_train)
                        # model_over.fit(X_over, y_over)
                        # model_under.fit(X_under, y_under)
                        print("Not Ready Yet")

                    elif task == "Classification":
                        models_list = classification_Models

                    for model in models_list:
                        model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])
                        model.fit(train_gen, epochs=5, steps_per_epoch= np.int(X_train.shape[0]/4000) )
                        # TODO change the label generator to give the data as NHWC (channels last)
                    feature_dict[ch_indx] = train_samplingstratigies(models_list, X_train,y_train,X_test,y_test)

                print(str(sub + sess + "_" + sub_task[-1]))
                Subjects_dict[str(sub + sess + "_" + sub_task[-1])] = feature_dict

                    # plot_test(model_original, X_test, y_test,
                    #           k + "\n No resampling strategy \n Acc " + str(strategy_dict["Original"]))
                    # plot_test(model_under, X_test, y_test,
                    #           k + "\n Oversampling strategy \n" + str(strategy_dict["Oversampling"]))
                    # plot_test(model_over, X_test, y_test,
                    #           k + "\n Undersampling strategy \n" + str(strategy_dict["Undersampling"]))

                # # TODO fix getting the model coefficients for XGBoost
                # for m in Subjects_dict[str(sub + sess + "_" + sub_task[-1])].keys():
                #     print(m)
                # m = 'logisticregression'
                # model_coeffs = get_modelCoeffs(Subjects_dict[str(sub + sess + "_" + sub_task[-1])][m][1])
                # results_overVSunder = Subjects_dict[str(sub + sess + "_" + sub_task[-1])][m][0]
                # results_overVSunder_df = pd.DataFrame.from_dict(results_overVSunder).T
                # # Subjects_dict[str(sub + sess + "_" + sub_task[-1])] = [model_dict, results_overVSunder_df]
                #
                #
                #
                # # plot_modelCoeffs(model_coeffs,name="Classification models coefficients \n for All Beta band variance features")
                #
                # plt.figure(figsize=(10, 6))
                # results_overVSunder_df.plot.bar(rot=0, alpha=0.5, linewidth=1, edgecolor='#08F7FE')
                # plt.title(f"{sub} Oversampling Vs Undersampling strategy for {task} \n {sess}")
                # plt.xticks(rotation=-30)
                # plt.show()

            # Subjects_dict[str(sub + sess + "_" + sub_task[-1])] = feature_dict

            else:
                print("Folder does not exist")

    Subjects_results[sub] = Subjects_dict

# feature_dict ==> channel ==> model type ===> test_score ==> strategy
#                                              model      ==> strategy

results_df = get_best_score(subjects_df(Subjects_results))

for model in classification_Models:
    print(str(model) )


Subjects_results.keys()

# results_overVSunder_df
