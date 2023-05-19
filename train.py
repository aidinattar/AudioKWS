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
    train.py <model> [--batch_size=<batch_size>] [--epochs=<epochs>] [--loss=<loss>]
                     [--optimizer=<optimizer>] [--metrics=<metrics>]
    train.py (-h | --help)
    train.py --version

Options:
    -h --help                   Show this screen.
    --version                   Show version.
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


def input_pipeline(path='DATA/speech_commands_v0.02', batch_size=64, shuffle_buffer_size=1000):
    """
    Get the data.
    """

    # Get the data.
    data_source = DataSource(path=path, batch_size=batch_size, shuffle_buffer_size=shuffle_buffer_size)
    data_source.print_commands()

    data_source.get_data()
    data_source.print_n()
    data_source.train_test_split()
    data_source.print_split()

    data_source.get_waveform_ds()
    data_source.get_spectrogram_ds()
    data_source.define_ds()
    data_source.batch_ds()

    return data_source

def model_pipeline(name_model, data, loss, optimizer, metrics, epochs=100):
    """
    Get the model.
    """

    # Get the model.
    model = getattr(models, name_model)(inputs=data, loss=loss, optimizer=optimizer, metrics=metrics)
    model.define_model()
    # Print the model parameters.
    model.summary()
    # Compile the model.
    model.compile()

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
    model.fit(epochs=epochs, callbacks=[model_tensorboad_callback, model_checkpoint_callback])
    model.evaluate_val()
    model.plot_model('figures/{}.png'.format(name_model))
    model.save_fit('history/{}.pkl'.format(name_model))
    model.plot_training()
    model.plot_confusion_matrix()
    model.plot_roc_OvR()
    model.save('models/{}.h5'.format(name_model))
    return model

def main(path='DATA/speech_commands_v0.02', batch_size=64, shuffle_buffer_size=1000,
         name_model='cnn_trad_fpool3', loss='sparse_categorical_crossentropy',
         optimizer='Adam', metrics='accuracy', epochs=100):
    """
    Main function.
    """
    data = input_pipeline(path=path, batch_size=batch_size, shuffle_buffer_size=shuffle_buffer_size)
    model = model_pipeline(name_model=name_model, data=data, loss=loss, optimizer=optimizer, metrics=metrics, epochs=epochs)

if __name__ == '__main__':

    args = docopt(__doc__, version='Train 0.1')

    name_model = args['<model>']

    if args['--batch_size'] is None:
        batch_size = 64
    else:
        batch_size = int(args['--batch_size'])

    if args['--epochs'] is None:
        epochs = 100
    else:
        epochs = int(args['--epochs'])

    if args['--loss'] is None:
        loss = 'sparse_categorical_crossentropy'
    else:
        loss = args['--loss']

    if args['--optimizer'] is None:
        optimizer = 'Adam'
    else:
        optimizer = args['--optimizer']

    if args['--metrics'] is None:
        metrics = 'accuracy'
    else:
        metrics = args['--metrics']

    main(path='DATA/speech_commands_v0.02', batch_size=batch_size, shuffle_buffer_size=1000,
         name_model=name_model, loss=loss, optimizer=optimizer, metrics=metrics, epochs=epochs)