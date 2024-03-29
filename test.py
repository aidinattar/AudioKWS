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
    train.py <model> [--batch_size=<batch_size>] [--epochs=<epochs>] [--loss=<loss>] [--lr=<lr>] [--metrics=<metrics>]
    train.py (-h | --help)
    train.py --version

Options:
    -h --help                   Show this screen.
    --batch_size=<batch_size>   Batch size [default: 256].
    --epochs=<epochs>           Number of epochs [default: 300]
    --loss=<loss>               Loss function [default: sparse_categorical_crossentropy]
    --lr=<lr>                   learing rate [default: 0.001]
    --metrics=<metrics>         Metrics [default: accuracy].

Example:
    python train.py cnn_trad_fpool3 --batch_size=64 --epochs=100 --loss=sparse_categorical_crossentropy --lr=0.001 --metrics=accuracy
"""
import os
import models
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from dataset import DataLoader, DataVisualizer, DatasetBuilder
from docopt import docopt


def input_pipeline(path:str='DATA/speech_commands_v0.02',
                   method_spectrum:str='log_mel',
                   test_ratio:float=0.15,
                   val_ratio:float=0.05,
                   batch_size:int=64,
                   shuffle_buffer_size:int=1000,
                   shuffle:bool=True,
                   seed:int=42,
                   verbose:int=1,
                   augmentation:bool=True
                   ):
    """
    Get the data.
    
    Parameters
    ----------
    path : str
        Path to the data.
    method_spectrum : str
        Method to compute the spectrum.
    test_ratio : float
        Ratio of the data to be used as test set.
    val_ratio : float
        Ratio of the data to be used as validation set.
    batch_size : int
        Batch size.
    shuffle_buffer_size : int
        Shuffle buffer size.
    shuffle : bool
        Whether to shuffle the data.
    seed : int
        Seed for the random number generator.
    verbose : int
        Verbosity level.
        
    Returns
    -------
    train : tf.data.Dataset
        Training dataset.
    test : tf.data.Dataset
        Test dataset.
    val : tf.data.Dataset
        Validation dataset.
    commands : list
        List of commands.
    """

    # Get the files.
    data = DataLoader(
        path=path
    )
    
    commands = data.get_commands()
    filenames = data.get_filenames()
    train_files, test_files, val_files = data.split_data(
        filenames=filenames,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        shuffle=shuffle,
        seed=seed,
        verbose=verbose
    )

    ds = DatasetBuilder(
        commands=commands,
        train_filenames=train_files,
        test_filenames=test_files,
        val_filenames=val_files,
        batch_size=batch_size,
        buffer_size=shuffle_buffer_size,
        method=method_spectrum
    )
    
    train, test, val = ds.preprocess_dataset_spectrogram(augment=augmentation)

    
    return train, test, val, commands



def evaluation_pipeline(
    model_name:str,
    model:tf.keras.Model,
    test_ds:tf.data.Dataset,
    commands:list,
    verbose:int=1,
):
    """
    Evaluate the model.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    test_ds : tf.data.Dataset
        Test dataset.
    commands : list
        List of commands.
    verbose : int
        Verbosity level.
    """
    methods = [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'roc',
        'confusion_matrix',
        'classification_report'
    ]
    
    # training history
    model.plot_training(
        path=os.path.join(
            'history',
            '{}.png'.format(model_name)
    ))
        
    
    # Evaluate the model.
    metric_test = dict()
    metric_train = dict()
    for method in methods:
        mtest = model.evaluate(
                set='test',
                method=method,
                model_name=model_name,
            )
        mtrain = model.evaluate(
                set='train',
                method=method,
                model_name=model_name,
            )
        metric_test[method] = mtest
        metric_train[method] = mtrain

    # Save the metrics.
    import json
    if not os.path.exists('metrics'):
        os.makedirs('metrics')
    with open('metrics/{}.txt'.format(model_name), 'w') as f:
        for key in metric_test.keys():
            f.write('{}: {}\n'.format(key, metric_test[key]))

def load_model(
    name_model:str,
    train_ds:tf.data.Dataset,
    test_ds:tf.data.Dataset,
    val_ds:tf.data.Dataset,
    commands:list,
    loss:str,
    optimizer:str,
    metrics:str,
    epochs:int=300,
    use_tensorboard:bool=True,
    save_checkpoint:bool=True,
    verbose:int=1,
    path:str='models'
):
    """
    Get the model, compile it, train it and evaluate it.
    """

    # Get the model.
    model = getattr(models, name_model)(
        train_ds=train_ds,
        test_ds=test_ds,
        val_ds=val_ds,
        commands=commands
    )

    if verbose:
        print('Model: {}'.format(name_model))

    model.create_model()
    model.load(filepath=path)
    return model


def saving_pipeline(
    model_name:str,
    model:tf.keras.Model,
    only_weights:bool=False,
    path:str='models',
    verbose:int=1,
    **kwargs
):
    """
    Save the model.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    path : str
        Path to save the model.
    verbose : int
        Verbosity level.
    """
    if only_weights:
        model.save_weights(
            filepath=path,
            **kwargs
        )
    else:
        model.save_model(
            filepath=path,
            **kwargs
        )
        
        if verbose: 
            print('Model saved at {}'.format(path))


def main(
    path='DATA/speech_commands_v0.02',
    method_spectrum='STFT',
    test_ratio=0.15,
    val_ratio=0.05,
    batch_size=128,
    shuffle_buffer_size=1000,
    name_model='CNNOneTStride8',
    loss='sparse_categorical_crossentropy',
    lr=0.001,
    metrics='accuracy',
    epochs=300,
    shuffle=True,
    use_tensorboard:bool=True,
    save_checkpoint:bool=True,
    verbose=1,
    seed=42,
    augmentation:bool=True,
):
    """
    Main function. Get the data, train the model and evaluate it.
    
    Parameters
    ----------
    path : str
        Path to the data.
    method_spectrum : str
        Method to compute the spectrum.
    test_ratio : float
        Ratio of the data to be used as test set.
    val_ratio : float
        Ratio of the data to be used as validation set.
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
    seed : int
        Seed for the random number generator.
    verbose : int
        Verbosity level.
    """

    # print args
    print('path: {}'.format(path))
    print('method_spectrum: {}'.format(method_spectrum))
    print('test_ratio: {}'.format(test_ratio))
    print('val_ratio: {}'.format(val_ratio))
    print('batch_size: {}'.format(batch_size))
    print('shuffle_buffer_size: {}'.format(shuffle_buffer_size))
    print('name_model: {}'.format(name_model))
    print('loss: {}'.format(loss))
    print('lr: {}'.format(lr))
    print('metrics: {}'.format(metrics))
    print('epochs: {}'.format(epochs))
    print('shuffle: {}'.format(shuffle))
    print('use_tensorboard: {}'.format(use_tensorboard))
    print('save_checkpoint: {}'.format(save_checkpoint))


    # optimizer = tf.keras.optimizers.Adam(learning_rate=float(lr), weight_decay=1e-5)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=float(lr), decay=1e-5)

    train, test, val, commands = input_pipeline(
        path=path,
        method_spectrum=method_spectrum,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        shuffle=shuffle,
        seed=seed,
        verbose=verbose,
        augmentation=augmentation
    )
    # img size
    img_size = train.element_spec
    print(img_size)

    # load model
    model = load_model(
        name_model=name_model,
        train_ds=train,
        test_ds=test,
        val_ds=val,
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
        epochs=100,
        use_tensorboard=use_tensorboard,
        save_checkpoint=save_checkpoint,
        verbose=1,
        commands=commands,
        path = 'models/{}.h5'.format(name_model)
    )

    saving_pipeline(
        model_name=name_model,
        model=model,
        only_weights=False,
        path=os.path.join(
            'models',
            '{}.h5'.format(name_model)
        ),
        verbose=1
    )
    
    evaluation_pipeline(
        model_name=name_model,
        model=model,
        test_ds=test,
        commands=commands,
        verbose=1
    )




if __name__ == '__main__':

    args = docopt(__doc__, version='Train 1.0')

    name_model = args['<model>']
    batch_size = int(args['--batch_size'])
    epochs = int(args['--epochs'])
    loss = args['--loss']
    lr = args['--lr']
    metrics = args['--metrics']

    main(
        path='DATA/speech_commands_v0.02',
        batch_size=batch_size,
        shuffle_buffer_size=1000,
        name_model=name_model,
        loss=loss,
        lr=lr,
        metrics=metrics,
        epochs=epochs
    )