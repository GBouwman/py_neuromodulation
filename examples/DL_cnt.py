import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import mplcyberpunk
plt.style.use("cyberpunk")
mplcyberpunk.add_glow_effects()
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, f1_score

from UTILS.Utils import *
import mne

import keras
import keras as k
from keras.layers import *
from keras.models import Sequential



f_ = "sub-002_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg.vhdr"
path = r"D:\Jupyter notebooks\Interventional Cognitive Neuromodulation\data\BIDS Berlin\Raw\sub-002\ses-EphysMedOff01\ieeg"
raw = mne.io.brainvision.read_raw_brainvision(os.path.join(path,f_))

# file = r"D:\Jupyter notebooks\Interventional Cognitive Neuromodulation\data\sub-000\ses-right\ieeg\sub-000_ses-right_task-force_run-0_ieeg.vhdr"
# raw = mne.io.brainvision.read_raw_brainvision(file)
info = raw.info
ch_names = info['ch_names']
fs = np.floor(info['sfreq'])
raw = raw.get_data()
data = raw[list(map(lambda x:x.startswith("ECOG"),ch_names)),]
label = raw[ch_names.index("ANALOG_R_ROTA_CH"),]
label = label[:,np.newaxis]


# PATH = "D:\Jupyter notebooks\Interventional Cognitive Neuromodulation\data"
# f_ = ["sub-000_ses-right_task-force_run-0_ieeg.vhdr","sub-000_ses-right_task-force_run-1_ieeg.vhdr", "sub-000_ses-right_task-force_run-2_ieeg.vhdr"]
# data, label = IO.get_data_raw_combined_berlin("sub-000","right","ECOG",f_, os.path.join(PATH,'sub-000','ses-right','ieeg') )
# fs = 1000


fs_new = 128
data_downsampled = signal.resample(data.T, int(data.T.shape[0] * fs_new / fs), axis=0)
label_downsampled = signal.resample(label, int(label.shape[0] * fs_new / fs), axis=0)

neg_y = -label_downsampled
y_baseline_corrected = baseline_corrected(data=neg_y, sub='sub-002', sess='self based-rotation', param=1e4, thr=2.5e-1, distance = 100)

X_train, X_test, y_train, y_test = train_test_split(data_downsampled, y_baseline_corrected[:, np.newaxis], test_size=0.2,shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,shuffle=False)

X_train_timeLagged = make_time_lag(pd.DataFrame(X_train[:,5], columns=['ch_5']), 100)
y_train_timeLagged = y_train[100:,:]

X_val_timeLagged = make_time_lag(pd.DataFrame(X_val[:,5], columns=['ch_5']), 100)
X_val_timeLagged = X_val_timeLagged[:,:,np.newaxis]
y_val_timeLagged = y_val[100:,:]

X_oversampled, y_oversampled = oversample(X_train_timeLagged, y_train_timeLagged)
X_oversampled = X_oversampled[:,:,np.newaxis]


input_shape = (101,1)

class MLP(DeepLearningModel):
    def __init__(self,n_layers = 4,neurons = [32,32,64,128], batch_normalize = True, bias = False,  *args, **kwargs):
        super(MLP,self).__init__(*args,**kwargs)
        self.n_layers = n_layers
        self.neurons = neurons
        self.batch_normalize = batch_normalize
        self.bias = bias
        self.model = self.build()

    def build(self):
        x = BatchNormalization()(self.input_layer)
        x = Flatten()(x)
        for i in range(self.n_layers):
            x = Dense(self.neurons[i], activation='relu', use_bias=self.bias)(x)
            if self.batch_normalize and i+1 != self.n_layers :
                x = BatchNormalization()(x)
        if self.use_dropout:
            x = Dropout(self.dropout)(x)
        self.output_layer = Dense(1, 'sigmoid', use_bias=self.bias)(x)

        return keras.Model(self.input_layer, self.output_layer)

    def call(self, inputs, training=None, mask=None):

        return self.model(inputs)

MLP().build().summary()


model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
model.add(Flatten())
model.add(Dense(32,activation = 'relu',use_bias=False))
model.add(BatchNormalization(axis=1))
model.add(Dense(32, activation='relu', use_bias=False))
model.add(BatchNormalization(axis=1))
model.add(Dense(64, activation='relu', use_bias=False))
model.add(BatchNormalization(axis=1))
model.add(Dense(128, activation='relu', use_bias=False))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid', use_bias=False))

model.summary()

model.compile(optimizer= "adam", loss = keras.losses.BinaryCrossentropy(from_logits = False), metrics=['accuracy', k.metrics.Recall(), k.metrics.Precision()])

model.fit(X_oversampled,y_oversampled>0, validation_data = (X_val_timeLagged, y_val_timeLagged>0) ,batch_size=256, epochs= 100)

X_test_timeLagged = make_time_lag(pd.DataFrame(X_test[:,5], columns=['ch_5']), 100)
y_test_timeLagged = y_test[100:,:]

X_test_timeLagged = X_test_timeLagged[:,:, np.newaxis]

model.evaluate(X_test_timeLagged, y_test_timeLagged>0 )

pred = model.predict(X_test_timeLagged)
plt.plot(y_test_timeLagged>0, label = 'Ground Truth', alpha = 0.5)
plt.plot(pred>0.5, label = 'Prediction', alpha = 0.5)
plt.legend()
plt.show()

class CNN1D(DeepLearningModel):
    def __init__(self, n_conv=3, conv_units=[32, 64, 128], conv_act = ReLU
                          , kernel_sizes=[64, 64, 64]
                          , pool_func=MaxPool1D, pool=[0, 1, 1]
                          , n_dense=2, dense_units=[200, 120], dense_activation=ReLU
                          ,batch_norm = False,*args,**kwargs):

        super(CNN1D, self).__init__(*args,**kwargs)
        self.n_conv = n_conv
        self.conv_units = conv_units
        self.conv_act = conv_act
        self.kernel_sizes = kernel_sizes
        self.pool_func = pool_func
        self.pool = pool
        self.n_dense = n_dense
        self.dense_units = dense_units
        self.dense_act = dense_activation
        self.batch_norm = batch_norm
        self.model = self.build()

    def build(self):
        x = BatchNormalization()(self.input_layer)
        for i in range(self.n_conv):
            x = Conv1D(self.conv_units[i], self.kernel_sizes[i], padding=self.padding)(x)
            x = self.conv_act()(x)
            if self.batch_norm:
                x = BatchNormalization()(x)

            if self.pool[i]:
                x = self.pool_func(pool_size=2, strides=2, padding='same')(x)

        x = Flatten()(x)

        for i in range(self.n_dense):
            x = Dense(self.dense_units[i])(x)
            x = self.dense_act()(x)
            if self.use_dropout:
                x = Dropout(self.dropout)(x)

        self.output_layer = Dense(self.n_out, self.out_act)(x)

        return keras.Model(self.input_layer, self.output_layer)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)


cnn1d = CNN1D(n_conv=3, kernel_sizes=[64,64,64],
              conv_units=[1,2,4],pool=[0,1,1], n_dense=1, dense_units=[128],
              batch_norm=True, use_dropout = False, dropout = 0).build()

cnn1d.summary()

cnn = Sequential()
cnn.add(Input(input_shape))
cnn.add(BatchNormalization())
# cnn.add(Reshape( (101,1) ))
cnn.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
cnn.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
cnn.add(Conv1D(64, kernel_size=3, padding= 'same', activation='relu'))
cnn.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
cnn.add(Conv1D(128, kernel_size=3, padding= 'same', activation='relu'))
cnn.add(MaxPool1D(pool_size=2, strides=2, padding='same'))
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu'))
cnn.add(Dense(128, activation='relu'))
cnn.add(Dense(1, activation='sigmoid'))
cnn.summary()

cnn1d.compile(optimizer= "adam", loss = keras.losses.BinaryCrossentropy(from_logits = False), metrics=['accuracy', k.metrics.Recall(), k.metrics.Precision()])

cnn1d.fit(X_oversampled,y_oversampled>0, validation_data = (X_val_timeLagged,y_val_timeLagged>0) ,batch_size=64, epochs= 10, shuffle = True)

cnn1d.evaluate(X_test_timeLagged,y_test_timeLagged>0)

pred = cnn1d.predict(X_test_timeLagged)
plt.plot(y_test_timeLagged>0, label = 'Ground Truth', alpha = 0.5)
plt.plot(pred, label = 'Prediction', alpha = 0.5)
plt.legend()
plt.show()



class LSTM1D(DeepLearningModel):
    def __init__(self, n_dense=3, dense_units=[100, 100, 100], dense_activation=ReLU
                  , n_lstm=2, lstm_units=[100, 100], lstm_activation=None
                  , *args,**kwargs):
        super(LSTM1D, self).__init__(*args, **kwargs)
        self.n_dense = n_dense
        self.dense_units = dense_units
        self.dense_act = dense_activation
        self.n_lstm = n_lstm
        self.lstm_units = lstm_units
        self.lstm_activation = lstm_activation
        self.model = self.build()

    def build(self):
        x = Reshape((-1,))(self.input_layer)
        x = BatchNormalization()(x)
        for i in range(self.n_dense):
            x = Dense(self.dense_units[i])(x)
            x = self.dense_act()(x)

        x = Reshape((-1,1))(x)

        for i in range(self.n_lstm):
            if i+1 == self.n_lstm:
                x = LSTM(self.lstm_units[i], return_sequences = False)(x)

            else:
                x = LSTM(self.lstm_units[i], return_sequences = True)(x)

            if self.lstm_activation:
                x = self.lstm_activation()(x)

            self.output_layer = Dense(self.n_out, activation = 'sigmoid')(x)

        return keras.Model(self.input_layer, self.output_layer)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)


rnn = Sequential()
rnn.add(Input((input_shape)))
rnn.add(Reshape((-1,)))
rnn.add(BatchNormalization())
rnn.add(Dense(100, activation='relu'))
rnn.add(Dense(100, activation='relu'))
rnn.add(Dense(100, activation='relu'))
rnn.add(Reshape( (100,1) ))
rnn.add(LSTM(100, use_bias=False, return_sequences=True))
rnn.add(LSTM(100, use_bias=False))
rnn.add(Dense(1,activation='sigmoid'))
rnn.summary()

rnn.compile(optimizer= "adam", loss = keras.losses.BinaryCrossentropy(from_logits = False), metrics=['accuracy'])

rnn.fit(X_oversampled,y_oversampled>0, validation_split=0.2 ,batch_size=64, epochs= 100)


pred = rnn.predict(X_test_timeLagged)
plt.plot(y_test_timeLagged>0, label = 'Ground Truth')
plt.plot(pred, label = 'Prediction', alpha = 0.5)
plt.legend()
plt.show()



class DeepLearningModel(keras.Model):
    def __new__(cls, custom_train = False, *args, **kwargs):
        if custom_train:
            return DLModelCustomTraining(*args, **kwargs)
        else:
            return super(DeepLearningModel, cls).__new__(cls, *args, **kwargs)

    def __init__(self,input_shape = (101,1), n_out = 1, out_act = 'sigmoid',
                 start_filters=32, kernel_size=3, activation=keras.layers.ReLU,
                 padding='same', dropout=0.25, use_dropout = False, custom_train = False):
        super(DeepLearningModel, self).__init__()
        DeepLearningModel.input_shape = input_shape
        self.n_out = n_out
        self.input_layer = k.Input(shape = self.input_shape)
        self.out_act = out_act
        self.filters = start_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.dropout = dropout
        self.custom_train = custom_train
        self.use_dropout = use_dropout

class DLModelCustomTraining(k.Model):
    def __init__(self,input_shape = (101,1), n_out = 1, out_act = 'sigmoid',
                 start_filters=32, kernel_size=3, activation=keras.layers.ReLU,
                 padding='same', dropout=0.25, use_dropout = False):
        super(DLModelCustomTraining, self).__init__()
        DeepLearningModel.input_shape = input_shape
        self.n_out = n_out
        self.input_layer = k.Input(shape = self.input_shape)
        self.out_act = out_act
        self.filters = start_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.dropout = dropout
        self.use_dropout = use_dropout

    def fit(self, epochs):
        print(epochs)


m = DeepLearningModel(custom_train=True)
m

m.fit('did this work?')


class Resnet(DeepLearningModel):
    def __init__(self, n_res = 3, *args, **kwargs):
        super(Resnet, self).__init__(*args, **kwargs)
        self.n_res = n_res
        self.residual_x = self.start_block(self.input_layer)
        self.model = self.build()

    def start_block(self, layer):
        x = BatchNormalization()(layer)
        x = Conv1D(self.filters, kernel_size=7, padding=self.padding, strides=2)(x)
        return MaxPool1D(pool_size=3, strides=2, padding= 'same')(x)

    def end_block(self,layer):
        x = GlobalAvgPool1D()(layer)
        return Dense(self.n_out, activation=self.out_act)(x)


    def build(self, batch_normalization = False):
        for n in range(self.n_res):
            self.x = Conv1D(self.filters, self.kernel_size, padding = self.padding, strides=1)(self.residual_x)
            if batch_normalization: self.x = BatchNormalization()(self.x)
            self.x = self.activation()(self.x)
            self.x = Conv1D(self.filters, self.kernel_size, padding = self.padding, strides=1)(self.x)
            if batch_normalization: self.x = BatchNormalization()(self.x)
            self.residual_x = Add()([self.x, self.residual_x])

        self.output_layer = self.end_block(self.residual_x)

        return keras.Model(self.input_layer, self.output_layer)

    def call(self, inputs, training=None, mask=None):

        return self.model(inputs)


resnet = Resnet(custom_train=False)


resnet.summary()

resnet.compile(optimizer= "adam", loss = keras.losses.BinaryCrossentropy(from_logits = False), metrics=['accuracy'])

resnet.fit(X_oversampled,y_oversampled>0, validation_data = (X_val_timeLagged, y_val_timeLagged) ,batch_size=64, epochs= 100, shuffle = True)

resnet.summary()

pred = resnet.predict(X_test_timeLagged)
plt.plot(y_test_timeLagged>0, label = 'Ground Truth')
plt.plot(pred, label = 'Prediction', alpha = 0.5)
plt.legend()
plt.show()

resnet.evaluate(X_test_timeLagged, y_test_timeLagged>0)


eegnet = Sequential()
eegnet.add(Input(input_shape))
eegnet.add(Reshape( (101,1,1) ))
eegnet.add(Conv2D(8,kernel_size=(50,1), padding='same', use_bias=False))
eegnet.add(BatchNormalization())
# eegnet.add(Reshape( (101,1,8) ))
eegnet.add(DepthwiseConv2D((1,1), use_bias=False,
                           depth_multiplier=2,
                           depthwise_constraint=max_norm(1.)))
eegnet.add(BatchNormalization())
eegnet.add(Activation('elu'))
eegnet.add(AveragePooling2D(((4,1))))
eegnet.add(SeparableConv2D(16, (16,1), use_bias=False, padding='same' ) )
eegnet.add(BatchNormalization())
eegnet.add(Activation('elu'))
eegnet.add(AveragePooling2D((2,1)))
eegnet.add(Dropout(0.5))
eegnet.add(Flatten())
eegnet.add(Dense(1, kernel_constraint=max_norm(0.25)))
eegnet.add(Activation('sigmoid'))
eegnet.summary()


eegnet.compile(optimizer= "adam", loss = keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

eegnet.fit(X_oversampled,y_oversampled>0, validation_data = (X_val_timeLagged, y_val_timeLagged) ,batch_size=64, epochs= 100, shuffle = True)

pred = eegnet.predict(X_test_timeLagged)
plt.plot(y_test_timeLagged>0, label = 'Ground Truth')
plt.plot(pred, label = 'Prediction', alpha = 0.5)
plt.legend()
plt.show()


# from tensorflow import keras
# from tensorflow.keras import layers

class Transformer(DeepLearningModel):
    def __init__(self, head_size=101, num_heads=2, ff_dim=8, num_transformer_blocks=2,
                mlp_units=[100], mlp_dropout=0.4, *args, **kwargs):
        super(Transformer, self).__init__(*args, **kwargs)
        self.head_size =head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.mlp_dropout = mlp_dropout

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        # x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = inputs
        x = k.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = k.layers.Dropout(dropout)(x)
        # res = x + inputs

        # Feed Forward Part
        # x = k.layers.LayerNormalization(epsilon=1e-6)(res)

        x = k.layers.Conv1D(filters=ff_dim, kernel_size=3, activation="relu")(x)
        x = k.layers.Dropout(dropout)(x)
        x = k.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x

    def build(self):

        inputs = keras.Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x, self.head_size, self.num_heads, self.ff_dim, self.dropout)

        x = k.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in self.mlp_units:
            x = k.layers.Dense(dim, activation="relu")(x)
            x = k.layers.Dropout(self.mlp_dropout)(x)
        outputs = k.layers.Dense(1, activation="sigmoid")(x)
        return k.Model(inputs, outputs)


transformer = Transformer(num_heads=3).build()
# transformer.summary()

transformer.compile(loss="binary_crossentropy",optimizer='adam',metrics=["accuracy"])
transformer.fit(X_oversampled,y_oversampled>0, validation_data = (X_val_timeLagged, y_val_timeLagged) ,batch_size=64, epochs= 100, shuffle = True)


# Tabnet

from pyneuromodulation.UTILS.archetictures.dl_archs import tabnet_classification



class Tabnet(DeepLearningModel):
    def __init__(self, num_features=128
                              , output_dim=1
                              , feature_dim=128
                              , num_decision_steps=6
                              , relaxation_factor=1.5
                              , batch_momentum=0.7
                              , epsilon=0.00001
                              , batch_size=64
                              , virtual_batch_size=32
                              , is_training=True, *args, **kwargs):

        super(Tabnet, self).__init__(*args, **kwargs)
        self.num_features=num_features
        self.output_dim=output_dim
        self.feature_dim=feature_dim
        self.num_decision_steps=num_decision_steps
        self.relaxation_factor=relaxation_factor
        self.batch_momentum=batch_momentum
        self.epsilon=epsilon
        self.batch_size=batch_size
        self.virtual_batch_size=virtual_batch_size
        self.is_training=is_training

    def glu(self,act, n_units):
        """Generalized linear unit nonlinear activation."""
        return act[:, :n_units] * k.activations.sigmoid(act[:, n_units:])

    def build(self):
        self.input_layer = keras.Input(shape=(self.num_features,))
        features = BatchNormalization()(self.input_layer)
        output_aggregated = tf.zeros((self.batch_size, self.output_dim))
        masked_features = features

        mask_values = tf.zeros((self.batch_size, self.num_features))
        aggregated_mask_values = tf.zeros((self.batch_size, self.num_features))
        complemantary_aggregated_mask_values = tf.ones((self.batch_size, self.num_features))
        total_entropy = 0

        if self.is_training:
            v_b = self.virtual_batch_size
        else:
            v_b = 1

        shared_transform_f1 = tf.keras.layers.Dense(
            self.feature_dim * 2,
            name="Transform_f1",
            use_bias=False)

        shared_transform_f2 = tf.keras.layers.Dense(
            self.feature_dim * 2,
            name="Transform_f2",
            use_bias=False)

        for ni in range(self.num_decision_steps):
            # Feature transformer with two shared and two decision step dependent
            # blocks is used below.

            reuse_flag = (ni > 0)

            transform_f1 = shared_transform_f1(masked_features)

            transform_f1 = k.layers.BatchNormalization(
                momentum=self.batch_momentum,
                virtual_batch_size=v_b
            )(transform_f1)

            transform_f1 = self.glu(transform_f1, self.feature_dim)

            transform_f2 = shared_transform_f2(transform_f1)
            transform_f2 = k.layers.BatchNormalization(
                # training=is_training,
                momentum=self.batch_momentum,
                virtual_batch_size=v_b
            )(transform_f2)
            transform_f2 = (self.glu(transform_f2, self.feature_dim) +
                            transform_f1) * np.sqrt(0.5)

            transform_f3 = tf.keras.layers.Dense(
                self.feature_dim * 2,
                name="Transform_f3" + str(ni),
                use_bias=False)(transform_f2)
            transform_f3 = k.layers.BatchNormalization(
                # training=is_training,
                momentum=self.batch_momentum,
                virtual_batch_size=v_b
            )(transform_f3)
            transform_f3 = (self.glu(transform_f3, self.feature_dim) +
                            transform_f2) * np.sqrt(0.5)

            transform_f4 = tf.keras.layers.Dense(
                self.feature_dim * 2,
                name="Transform_f4" + str(ni),
                use_bias=False)(transform_f3)
            transform_f4 = k.layers.BatchNormalization(
                momentum=self.batch_momentum,
                virtual_batch_size=v_b
            )(transform_f4)
            transform_f4 = (self.glu(transform_f4, self.feature_dim) +
                            transform_f3) * np.sqrt(0.5)

            if ni > 0:
                decision_out = k.layers.ReLU()(transform_f4[:, :self.output_dim])

                # Decision aggregation.
                output_aggregated += decision_out

                # Aggregated masks are used for visualization of the
                # feature importance attributes.
                scale_agg = tf.reduce_sum(
                    decision_out, axis=1, keepdims=True) / (
                                    self.num_decision_steps - 1)
                aggregated_mask_values += mask_values * scale_agg

            features_for_coef = (transform_f4[:, self.output_dim:])

            if ni < self.num_decision_steps - 1:
                # Determines the feature masks via linear and nonlinear
                # transformations, taking into account of aggregated feature use.
                mask_values = k.layers.Dense(
                    self.num_features,
                    name="Transform_coef" + str(ni),
                    use_bias=False)(features_for_coef)
                mask_values = k.layers.BatchNormalization(
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b
                )(mask_values)
                mask_values *= complemantary_aggregated_mask_values
                mask_values = tfa.layers.Sparsemax()(mask_values)

                complemantary_aggregated_mask_values *= (
                        self.relaxation_factor - mask_values)

                # Entropy is used to penalize the amount of sparsity in feature
                # selection.
                total_entropy += tf.reduce_mean(
                    tf.reduce_sum(
                        -mask_values * tf.math.log(mask_values + self.epsilon),
                        axis=1)) / (
                                         self.num_decision_steps - 1)

                # Feature selection.
                masked_features = tf.math.multiply(mask_values, features)

        predictions = k.layers.Dense(self.output_dim, activation='sigmoid', use_bias=False)(output_aggregated)

        model = tf.keras.models.Model(self.input_layer, predictions)

        return model
    def call(self, inputs, training=None, mask=None):

        self.model = keras.Sequential([Input(self.input_shape),BatchNormalization(),self.build()])
        return self.model(inputs,training = training)

t = Tabnet(num_features = input_shape[0]).build()

tabnet = tabnet_classification(num_features= X_train_timeLagged.shape[1]
                                  , output_dim=1
                                  , feature_dim=X_train_timeLagged.shape[1]
                                  , num_decision_steps=3
                                  , relaxation_factor=1.5
                                  , batch_momentum=0.7
                                  , epsilon=0.00001
                                  , BATCH_SIZE=64
                                  , virtual_batch_size=32)



batch_size = 64
tabnet.compile(loss="binary_crossentropy",optimizer='adam',metrics=["accuracy"])
tabnet.fit(X_oversampled[0:int(np.floor(X_oversampled.shape[0]/batch_size)*batch_size),:,:]
            ,y_oversampled[0:int(np.floor(X_oversampled.shape[0]/batch_size)*batch_size)]>0,
             validation_data =
            (X_val_timeLagged[0:int(np.floor(X_val_timeLagged.shape[0]/batch_size)*batch_size),:,:],
             y_val_timeLagged[0:int(np.floor(y_val_timeLagged.shape[0]/batch_size)*batch_size)]) ,
           batch_size=batch_size, epochs= 100, shuffle = True)


tabnet_withBatchNormalizatin = Sequential([Input(input_shape),
            BatchNormalization(),
            tabnet])

tabnet_withBatchNormalizatin.compile(loss="binary_crossentropy",optimizer='adam',metrics=["accuracy"])
tabnet_withBatchNormalizatin.fit(X_oversampled[0:int(np.floor(X_oversampled.shape[0]/batch_size)*batch_size),:,:]
            ,y_oversampled[0:int(np.floor(X_oversampled.shape[0]/batch_size)*batch_size)]>0,
             validation_data =
            (X_val_timeLagged[0:int(np.floor(X_val_timeLagged.shape[0]/batch_size)*batch_size),:,:],
             y_val_timeLagged[0:int(np.floor(y_val_timeLagged.shape[0]/batch_size)*batch_size)]) ,
           batch_size=batch_size, epochs= 100, shuffle = True)





X_val_timeLagged.shape[0]


X_oversampled.shape[0]/64

