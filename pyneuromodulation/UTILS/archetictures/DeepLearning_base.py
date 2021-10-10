import keras

class DeepLearningModel(keras.Model):
    def __new__(cls, custom_train = False, *args, **kwargs):
        if custom_train:
            return DLModelCustomTraining(*args, **kwargs)
        else:
            return super(DeepLearningModel, cls).__new__(cls, *args, **kwargs)

    def __init__(self,input_shape = (101,1), n_out = 1, out_act = 'sigmoid',
                 start_filters=32, kernel_size=3, activation=keras.layers.ReLU,
                 padding='same', dropout=0.25, custom_train = False,  use_dropout = False,*args,**kwargs):
        super(DeepLearningModel, self).__init__(*args,**kwargs)
        DeepLearningModel.input_shape = input_shape
        self.n_out = n_out
        self.input_layer = keras.Input(shape = self.input_shape)
        self.out_act = out_act
        self.filters = start_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.dropout = dropout
        self.custom_train = custom_train
        self.use_dropout = use_dropout

class DLModelCustomTraining(keras.Model):
    def __init__(self,input_shape = (101,1), n_out = 1, out_act = 'sigmoid',
                 start_filters=32, kernel_size=3, activation=keras.layers.ReLU,
                 padding='same', dropout=0.25, use_dropout = False,*args,**kwargs):
        super(DLModelCustomTraining, self).__init__(*args,**kwargs)
        DeepLearningModel.input_shape = input_shape
        self.n_out = n_out
        self.input_layer = keras.Input(shape = self.input_shape)
        self.out_act = out_act
        self.filters = start_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.dropout = dropout
        self.use_dropout = use_dropout
    # TODO add rhe custom training loop here
    def fit(self, epochs):
        print(epochs)


class TrainingDecider(DeepLearningModel):
    def __new__(cls, custom_train=False, *args, **kwargs):
        if custom_train:
            return CustomTraining(*args, **kwargs)
        else:
            return OriginalTraining(*args, **kwargs)
        
    def __init__(self, *args,**kwargs):
        super(TrainingDecider, self).__init__(*args,**kwargs)


class OriginalTraining(DeepLearningModel):
    def __init__(self, *args, **kwargs):
        super(OriginalTraining, self).__init__(*args, **kwargs)
        pass

class CustomTraining(DeepLearningModel):
    def __init__(self, *args, **kwargs):
        super(CustomTraining, self).__init__(*args, **kwargs)

    def fit(self, epochs):
        print(epochs)


