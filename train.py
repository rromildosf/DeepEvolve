#train
"""
Generic setup of the data sources and the model training. 

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
and also on 
    https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

"""

#import keras
from keras.datasets       import mnist, cifar10
from keras.models         import Sequential
from keras.layers         import Dense, Dropout, Flatten
from keras.utils          import to_categorical
from keras.callbacks      import EarlyStopping, Callback
from keras.layers         import Conv2D, MaxPooling2D
from keras                import backend as K
from skimage import io, color

import logging
import numpy as np
import os


# Helper: Early stopping.
early_stopper = EarlyStopping( monitor='val_loss', min_delta=0.1, patience=2, verbose=0, mode='auto' )

#patience=5)
#monitor='val_loss',patience=2,verbose=0
#In your case, you can see that your training loss is not dropping - which means you are learning nothing after each epoch. 
#It look like there's nothing to learn in this model, aside from some trivial linear-like fit or cutoff value.



def split_dataset( inputs, outpts ):
    # Split dataset
    test = int(len(inputs)*0.1/2)
    train = int((len(inputs)-test)/2)
    m = int( len(inputs)/2 )
    y_test = outpts[:test]#.extend(outpts[m:m+test])
    y_test.extend(outpts[m:m+test])
    x_test = inputs[:test]#.extend( inputs[m:m+test] )
    x_test.extend( inputs[m:m+test]  )


    y_train = outpts[:train]#.extend(outpts[m:m+test])
    y_train.extend(outpts[m:m+train])
    x_train = inputs[:train]#.extend( inputs[m:m+test] )
    x_train.extend( inputs[m:m+train]  )

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)

def load_dataset(config, ext='png'):
    """ Load dataset
    returns: (X, Y)
    """
    def srt(el):
        return int( el.split('.')[-2].split('_')[-1] )
  
    Y = np.loadtxt( os.path.join( config.datadir, config.labels_filename ) )
    
    imgs = [ i for i in os.listdir( config.datadir ) if i.endswith(ext) ]
    imgs.sort(key=srt)
  
    X = []
    for i in imgs:
        img = io.imread(  os.path.join( config.datadir, i ), asGray=True )
        X.append( img )
    
    ## 50% WITH + 50% WITHOUT
    inputs = []
    outpts = []    
    for x, y in zip( X, Y ):
        if y == 1.0:
            inputs.append( x )
            outpts.append( y )
    c = 0;
    l = len(inputs)
    for x, y in zip(X, Y):
        if y == 0.0 and c < l:
            inputs.append( x )
            outpts.append( y )
            c +=1

    (x_train, y_train), (x_test, y_test) = split_dataset( inputs, outpts )
    x_train = x_train.reshape(x_train.shape[0], *config.input_shape)
    x_test  = x_test.reshape(x_test.shape[0], *config.input_shape)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, config.n_classes)
    y_test  = to_categorical(y_test, config.n_classes)
    
    return (x_train, y_train), (x_test, y_test)


def compile_model_mlp(geneparam, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    ann_nb_layers  = genome.geneparam['nb_ann_layers' ]
    ann_nb_neurons = genome.geneparam['nb_ann_neurons']
    ann_activation = genome.geneparam['ann_activation']
    optimizer      = genome.geneparam['optimizer']

    logging.info('Architecture: ann_nb_layers: {}, ann_nb_neurons: {}, ' +
                'ann_activation: {}, optimizer: {}'.format( ann_nb_layers, ann_nb_neurons,
                                                            ann_activation, optimizer) )

    model = Sequential()

    # Add each layer.
    model.add(Dense(ann_nb_neurons, activation=ann_activation, input_shape=input_shape))
    for i in range(cnn_nb_layers):
        model.add(Dense(ann_nb_neurons, activation=ann_activation))
        model.add(Dropout(0.2))  # hard-coded dropout for each layer

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                    optimizer=optimizer,
                    metrics=['acc'])

    return model

def compile_model_cnn(genome, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        genome (dict): the parameters of the genome

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    cnn_nb_layers  = genome.geneparam ['cnn_nb_layers' ]
    cnn_nb_neurons = genome.nb_neurons('cnn')
    ann_nb_layers  = genome.geneparam ['ann_nb_layers' ]
    ann_nb_neurons = genome.nb_neurons('ann')
    
    cnn_activation = genome.geneparam['cnn_activation']
    ann_activation = genome.geneparam['ann_activation']
    optimizer      = genome.geneparam['optimizer']

    message = ('Architecture: cnn_nb_layers: {}, cnn_nb_neurons: {},' \
                 +'ann_nb_layers: {}, ann_nb_neurons: {}, cnn_activation: {}, ' \
                 +'ann_activation: {}, optimizer: {}').format( cnn_nb_layers, str(cnn_nb_neurons),
                                                            ann_nb_layers, str(ann_nb_neurons),
                                                            cnn_activation, ann_activation,
                                                            optimizer) 
    logging.info( message )
    model = Sequential()

    # Add each layer.
    model.add(Conv2D(cnn_nb_neurons[0], kernel_size=(3, 3), 
                activation=cnn_activation, padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(2))
                 
    for i in range(6):
        model.add(Conv2D(cnn_nb_neurons[i], kernel_size = (3, 3), activation=cnn_activation))
        if i < 2: #otherwise we hit zero
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

    ## ANN MLP
    model.add(Flatten())
    # always use last nb_neurons value for dense layer
    for i in range(6):
        model.add(Dense(ann_nb_neurons[i], activation=ann_activation))
        model.add(Dropout(0.333))
    model.add(Dense(nb_classes, activation = 'softmax'))

    #BAYESIAN CONVOLUTIONAL NEURAL NETWORKS WITH BERNOULLI APPROXIMATE VARIATIONAL INFERENCE
    #need to read this paper

    model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['acc'])

    return model

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def train_and_score(genome, config):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    (x_train, y_train), (x_test, y_test) = load_dataset( config )
    
    if config.model_type == 'cnn':
        model = compile_model_cnn(genome, config.n_classes, config.input_shape)
    else:
        model = compile_model_mlp(genome, config.n_classes, config.input_shape)
    history = LossHistory()

    model.fit( x_train, y_train,
              batch_size=config.batch_size,
              epochs=config.epochs,  
              # using early stopping so no real limit - don't want to waste time on horrible architectures
              verbose=1,
              validation_data=(x_test, y_test),
              #callbacks=[history])
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1], end='\n\n')

    K.clear_session()
    #we do not care about keeping any of this in memory - 
    #we just need to know the final scores and the architecture
    
    return score[1]  # 1 is accuracy. 0 is loss.
