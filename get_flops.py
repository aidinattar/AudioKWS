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
                   verbose:int=1):


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
    
    train, test, val = ds.preprocess_dataset_spectrogram()
    
    return train, test, val, commands

def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


def flops(
    name_model:str,
    train_ds:tf.data.Dataset,
    test_ds:tf.data.Dataset,
    val_ds:tf.data.Dataset,
    commands:list,
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


    model.create_model()

    # Print the model parameters.
    model = model.model
    print ('starting flops')
    flops = get_flops(model, )/1e9
    return flops


def process(
    path='DATA/speech_commands_v0.02',
    method_spectrum='mfcc',
    test_ratio=0.15,
    val_ratio=0.05,
    batch_size=128,
    shuffle_buffer_size=1000,
    name_model='cnn_trad_fpool3',
    loss='sparse_categorical_crossentropy',
    lr=0.01,
    metrics='accuracy',
    epochs=300,
    shuffle=True,
    use_tensorboard:bool=True,
    save_checkpoint:bool=True,
    verbose=1,
    seed=42,
):


    train, test, val, commands = input_pipeline(
        path=path,
        method_spectrum=method_spectrum,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        shuffle=shuffle,
        seed=seed,
        verbose=verbose
    )
    
    model = flops(
        name_model=name_model,
        train_ds=train,
        test_ds=test,
        val_ds=val,
        commands=commands,
    )
    print('Model: {}'.format(name_model))
    print ('FLOPS: {}'.format(model))
    





if __name__ == '__main__':

    args = docopt(__doc__, version='Train 1.0')

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