import os
import matplotlib.pyplot as plt
import mplcyberpunk
plt.style.use("cyberpunk")
mplcyberpunk.add_glow_effects()
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from UTILS.Utils import *
from UTILS.archetictures import CNN
import UTILS.label_generator as label_generator


task = 'Classification' # | "Regression"
feature_level = "Channels" # | "Features"

classification_models = [CNN.make_cnnModel(input_shape=(1, 128, 1), summary = True, ch_first=False, output_units=2, output_activation='sigmoid')]
import keras
keras.backend.set_image_data_format('channels_last')



# Reading the data file
subjects = ['sub-002']#, 'sub-003', 'sub-004']
sessions = ['ses-EphysMedOff01']#,'ses-EphysMedOff02','ses-EphysMedOff03'] # 'ses-EphysMedOn03', 'ses-EphysMedOn01']
tasks = ['task-SelfpacedRotationR']#,'task-SelfpacedRotationL']
stims = ['acq-StimOff'] #, 'acq-StimOn']
runs = ['run-01']

csv_name = "sub-002_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg_FEATURES.csv"
csv_path = r"D:\Jupyter notebooks\Interventional Cognitive Neuromodulation\data\derivatives_cnt\sub-002_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg"
PATH = r"D:\Jupyter notebooks\Interventional Cognitive Neuromodulation\data\derivatives_cnt"

("_").join([subjects[0],sessions[0],tasks[0],stims[0],runs[0],"ieeg","FEATURES.csv"]) == csv_name

filenames =  os.listdir(PATH)
filenames[0].split("_")


Subjects_results = {}
Subjects_dict = {}
for sub in subjects:
    for sess in sessions:
        for sub_task in tasks:
            for stim in stims:

                folder_name = "_".join([sub,sess,sub_task,stim,runs[0],"ieeg"])
                print(folder_name)
                if folder_name in filenames:
                    print("Folder exists")

                    features_file = os.path.join(PATH, folder_name,folder_name+"_FEATURES.csv")
                    df = pd.read_csv(os.path.join(PATH, features_file))

                    data = df.filter(like= "ECOG").values
                    label = df.filter(like = "ROTA").values


                    neg_y = -label
                    if sub in ['sub-004'] and sub_task not in ['task-SelfpacedRotationR']:
                        y_baseline_corrected = baseline_corrected(data=label,sub = sub, sess=sess, param=1e4, thr=2.5e-1, distance=100)
                    else:
                        y_baseline_corrected = baseline_corrected(data=neg_y, sub=sub, sess=sess, param=1e4, thr=2.5e-1, distance=100)

                    thr = 0.0
                    results_overVSunder = {}
                    models_overVSunder = {}
                    feature_dict = {}
                    import sklearn.preprocessing
                    le = sklearn.preprocessing.LabelEncoder()
                    y = le.fit_transform(y_baseline_corrected[:, np.newaxis])

                    for ch_indx in range(data.shape[1]):
                        print(ch_indx)
                        ch_indx = 5
                        X_train, X_test, y_train, y_test = train_test_split(data[:, ch_indx, np.newaxis],
                                                                            y[:, np.newaxis],
                                                                            test_size=0.2, shuffle=False)
                        train_gen = label_generator.generator(X_train, y_train, 4000, 128, rebalance=True,
                                                              rebalanced_thr=0, task=task)
                        test_gen = label_generator.generator(X_test, y_test, 4000, 128, rebalance=False,
                                                             rebalanced_thr=0, task=task)

                        next(train_gen)[0].shape

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
                            models_list = classification_models

                        for model in models_list:
                            model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
                            model.fit(train_gen, epochs=3, steps_per_epoch=np.int8(X_train.shape[0] / 4000))
                            # TODO change the label generator to give the data as NHWC (channels last)
                        feature_dict[ch_indx] = train_samplingstratigies(models_list, X_train, y_train, X_test, y_test)

                    print(str(sub + sess + "_" + sub_task[-1]))
                    Subjects_dict[str(sub + sess + "_" + sub_task[-1])] = feature_dict

                    # plot_test(model_original
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
