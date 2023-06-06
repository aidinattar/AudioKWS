import os
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from ViT import VisionTransformer
import tensorflow as tf
import sys
sys.path.append('..')
from dataset import DataLoader, DataVisualizer, DatasetBuilder
from docopt import docopt

AUTOTUNE = tf.data.experimental.AUTOTUNE

def input_pipeline(path:str='../DATA/speech_commands_v0.02',
                   method_spectrum:str='log_mel',
                   test_ratio:float=0.15,
                   val_ratio:float=0.05,
                   batch_size:int=64,
                   shuffle_buffer_size:int=1000,
                   shuffle:bool=True,
                   seed:int=42,
                   verbose:int=1):
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
    
    train, test, val = ds.preprocess_dataset_spectrogram()
    
    return train, test, val, commands

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--image-size", default=32, type=int)
    parser.add_argument("--patch-size", default=4, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--d-model", default=64, type=int)
    parser.add_argument("--num-heads", default=4, type=int)
    parser.add_argument("--mlp-dim", default=128, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=4096, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    args = parser.parse_args()

    path='../DATA/speech_commands_v0.02'
    method_spectrum = 'log_mel'
    test_ratio = 0.15
    val_ratio = 0.05
    batch_size = 64
    shuffle_buffer_size = 1000
    shuffle = True
    seed = 42
    verbose = 1

    ds_train, ds_test, val, commands = input_pipeline(
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
    print ("image_size: ", ds_train.element_spec)
    print ("image_size: ", ds_test.element_spec)

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = VisionTransformer(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_layers=args.num_layers,
            num_classes=35,
            d_model=args.d_model,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            channels=3,
            dropout=0.1,
        )
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
            ),

            metrics=["accuracy"],
        )

    model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=args.epochs,
        callbacks=[TensorBoard(log_dir=args.logdir, profile_batch=0),],
    )
    model.save_weights(os.path.join(args.logdir, "vit"))