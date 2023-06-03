"""
Train a model on the speech_commands_v0.02 dataset.

This script trains a model on the speech_commands_v0.02 dataset. The dataset
is downloaded from the TensorFlow website if it is not already present.

The models are defined in models.py, according to [Sainath15]. The model is
trained using the tf.keras API.

The model is trained using the Adam optimizer and sparse categorical
cross entropy loss. The model is trained for 100 epochs and the accuracy
and loss are plotted using matplotlib.


Usage:
    train.py <model> [--batch_size=<batch_size>] [--epochs=<epochs>] [--loss=<loss>] [--optimizer=<optimizer>] [--metrics=<metrics>]
    train.py (-h | --help)
    train.py --version

Options:
    -h --help                   Show this screen.
    --batch_size=<batch_size>   Batch size [default: 64].
    --epochs=<epochs>           Number of epochs [default: 100]
    --loss=<loss>               Loss function [default: sparse_categorical_crossentropy]
    --optimizer=<optimizer>     Optimizer [default: Adam]
    --metrics=<metrics>         Metrics [default: accuracy].

Example:
    python train.py cnn_trad_fpool3 --batch_size=64 --epochs=100 --loss=sparse_categorical_crossentropy --optimizer=Adam --metrics=accuracy
"""

import pickle
import models
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from DataSource import DataSource
from docopt import docopt


def input_pipeline(path:str='DATA/speech_commands_v0.02',
                   test_ratio:float=0.15,
                   val_ratio:float=0.05,
                   batch_size:int=64,
                   shuffle_buffer_size:int=1000,
                   verbose:int=1):
    """
    Get the data.
    
    Parameters
    ----------
    path : str
        Path to the data.
    test_ratio : float
        Ratio of the data to be used as test set.
    val_ratio : float
        Ratio of the data to be used as validation set.
    batch_size : int
        Batch size.
    shuffle_buffer_size : int
        Shuffle buffer size.
    verbose : int
        Verbosity level.
        
    Returns
    -------
    data_source : DataSource
        Object containing the data.
    """

    # Get the data.
    data_source = DataSource(
        path=path,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        verbose=verbose
    )
    data_source.print_commands()

    data_source.get_data()
    data_source.print_example()
    data_source.train_test_split(
        test_ratio=test_ratio,
        val_ratio=val_ratio,
    )

    #data_source.get_waveform_ds()
    #data_source.get_spectrogram_ds()
    data_source.define_ds()
    data_source.batch_ds()

    return data_source

def training_pipeline(name_model:str,
                      data:DataSource,
                      loss:str,
                      optimizer:str,
                      metrics:str,
                      epochs:int=100):
    """
    Get the model, compile it, train it and evaluate it.
    """

    # Get the model.
    model = getattr(models, name_model)(
        inputs=data
    )
    model.create_model()

    # Print the model parameters.
    model.summary()

    # Compile the model.
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics
    )

    # Define the callbacks.
    model_tensorboad_callback = TensorBoard(log_dir="logs/{}".format(name_model))
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints",
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    # Train the model.
    #model.fit(epochs=epochs, callbacks=[model_tensorboad_callback, model_checkpoint_callback])
    model.fit(
        epochs=epochs,
        callbacks=[
            model_tensorboad_callback,
            model_checkpoint_callback
        ]
    )
    
    evals = [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'confusion_matrix',
        'roc',
        'pr'
    ]

    for ev in evals:
        model.evaluate(
            set='val',
            method=ev,
        )
    
    model.plot_model(
        'figures/{}.png'.format(name_model),
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
        dpi=96
    )

    model.save_fit('history/{}.pkl'.format(name_model))
    model.plot_training()
    model.plot_confusion_matrix()
    model.plot_roc_OvR()
    model.save('models/{}.h5'.format(name_model))

    return model

def main(path='DATA/speech_commands_v0.02',
         batch_size=64,
         shuffle_buffer_size=1000,
         name_model='cnn_trad_fpool3',
         loss='sparse_categorical_crossentropy',
         optimizer='Adam',
         metrics='accuracy',
         epochs=100):
    """
    Main function. Get the data, train the model and evaluate it.
    
    Parameters
    ----------
    path : str
        Path to the data.
    batch_size : int
        Batch size.
    shuffle_buffer_size : int
        Shuffle buffer size.
    name_model : str
        Name of the model.
    loss : str
        Loss function.
    optimizer : str
        Optimizer.
    metrics : str
        Metrics.
    epochs : int
        Number of epochs.
    """
    data = input_pipeline(path=path, batch_size=batch_size, shuffle_buffer_size=shuffle_buffer_size)
    model = training_pipeline(name_model=name_model, data=data, loss=loss, optimizer=optimizer, metrics=metrics, epochs=epochs)

if __name__ == '__main__':

    args = docopt(__doc__, version='Train 0.1')

    name_model = args['<model>']
    batch_size = int(args['--batch_size'])
    epochs = int(args['--epochs'])
    loss = args['--loss']
    optimizer = args['--optimizer']
    metrics = args['--metrics']

    main(
        path='DATA/speech_commands_v0.02',
        batch_size=batch_size,
        shuffle_buffer_size=1000,
        name_model=name_model,
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
        epochs=epochs
    )