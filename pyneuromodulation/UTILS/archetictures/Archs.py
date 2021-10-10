import keras
import tensorflow as tf
import tensorflow.keras as k
import numpy as np
import tensorflow_addons as tfa

from keras.layers import *
from UTILS.archetictures.DeepLearning_base import DeepLearningModel



class MLP(DeepLearningModel):
    def __init__(self,name = "MLP",n_layers = 4,neurons = [32,32,64,128], batch_normalize = True, bias = False,  *args, **kwargs):
        super(MLP,self).__init__(*args,**kwargs)
        self.n_layers = n_layers
        self.neurons = neurons
        self.batch_normalize = batch_normalize
        self.bias = bias
        MLP.name = name
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

        return keras.Model(self.input_layer, self.output_layer, name=self.name)

    def call(self, inputs, training=None, mask=None):

        return self.model(inputs)


class CNN1D(DeepLearningModel):
    def __init__(self, n_conv=3, conv_units=[32, 64, 128], conv_act=ReLU
                 , kernel_sizes=[3, 3, 3]
                 , pool_func=MaxPool1D, pool=[0, 1, 1]
                 , n_dense=2, dense_units=[200, 120], dense_activation=ReLU
                 , batch_norm=False, name = 'CNN1D', *args, **kwargs):

        super(CNN1D, self).__init__(*args, **kwargs)
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
        CNN1D.name = name
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

        self.output_layer = Dense(self.n_out, self.out_act)(x)

        return keras.Model(self.input_layer, self.output_layer, name=self.name)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)


class LSTM1D(DeepLearningModel):
    def __init__(self, n_dense=3, dense_units=[100, 100, 100], dense_activation=ReLU
                 , n_lstm=2, lstm_units=[100, 100], lstm_activation=None
                 , name = 'LSTM1D', *args, **kwargs):
        super(LSTM1D, self).__init__(*args, **kwargs)
        self.n_dense = n_dense
        self.dense_units = dense_units
        self.dense_act = dense_activation
        self.n_lstm = n_lstm
        self.lstm_units = lstm_units
        self.lstm_activation = lstm_activation
        LSTM1D.name = name
        self.model = self.build()

    def build(self):
        x = Reshape((-1,))(self.input_layer)
        x = BatchNormalization()(x)
        for i in range(self.n_dense):
            x = Dense(self.dense_units[i])(x)
            x = self.dense_act()(x)

        x = Reshape((-1, 1))(x)

        for i in range(self.n_lstm):
            if i + 1 == self.n_lstm:
                x = LSTM(self.lstm_units[i], return_sequences=False)(x)

            else:
                x = LSTM(self.lstm_units[i], return_sequences=True)(x)

            if self.lstm_activation:
                x = self.lstm_activation()(x)

            self.output_layer = Dense(self.n_out, activation='sigmoid')(x)

        return keras.Model(self.input_layer, self.output_layer, name=self.name)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)


class Resnet(DeepLearningModel):
    def __init__(self, n_res = 3, name = 'Resnet', *args, **kwargs):
        super(Resnet, self).__init__(*args, **kwargs)
        self.n_res = n_res
        self.residual_x = self.start_block(self.input_layer)
        Resnet.name  = name
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

        return keras.Model(self.input_layer, self.output_layer, name=self.name)

    def call(self, inputs, training=None, mask=None):

        return self.model(inputs)


# This architecture was adopted from Google original Github Repo

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
                              , is_training=True
                              , name = 'Tabnet',  *args, **kwargs):

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
        Tabnet.name = name

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

        model = tf.keras.models.Model(self.input_layer, predictions, name=self.name)

        return model
    def call(self, inputs, training=None, mask=None):

        self.model = keras.Sequential([Input(self.input_shape),BatchNormalization(),self.build()])
        return self.model(inputs,training = training)



