from flask import Flask, request
import sys
sys.path.append('..')
from dataset import DataLoader, DatasetBuilder
#from models import *
from utils.input import *
import sys 
sys.path.append('..')
from models import *
from model import *
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid", font_scale=1.5, palette='magma')
import matplotlib
matplotlib.use('agg')

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


def predict(N):

    data = DataLoader(path='demo_data')
    commands = data.get_commands()
    filenames = data.get_filenames()
    train_files, val_files, test_files = data.split_data(filenames)

    ds = DatasetBuilder(
        commands=commands,
        train_filenames=train_files,
        val_filenames=val_files,
        test_filenames=test_files,
        batch_size=64,
        buffer_size=1000,
        method='log_mel'
    )
    waveforms = data.get_waveform_ds(train_files)
    ds = data.get_spectrogram_logmel_ds(waveforms, commands)

    imgs = []
    labels = []
    for i in ds.take(N):
        img = i[0].numpy()
        label = i[1].numpy()
        imgs.append(img)
        labels.append(label)
    model_name = 'DNNBaseline'
    train, test, val, commands = input_pipeline(path='demo_data')
    model_path= f'../models/{model_name}.h5'
    model = globals()[model_name](
                                train_ds=train,
                                test_ds=test,
                                val_ds=val,
                                commands=commands
                            )
    model.define_model()
    try:
        model.load_weights(model_path)
    except:
        raise ValueError('Model not found. Please, check the model name.')
    model = model.model

    expimgs = map(lambda x: tf.expand_dims(x, axis=0), imgs)
    preds = model.predict(expimgs)
    pred = [np.argmax(i) for i in preds]

    label2idx = {i:idx for idx, i in enumerate(commands)}
    idx2label = {idx:i for idx, i in enumerate(commands)}
    original_label = [idx2label[i] for i in labels]
    predicted_label = [idx2label[i] for i in pred]

    results = {
        'original_label': original_label,
        'predicted_label': predicted_label,
        'img': imgs
    }
    return results

    
app = Flask(__name__)
@app.route('/<N>', methods=['GET','POST'])
def predict_api(N):
    N = int(N)
    dict_res = predict(N)
    imgs = dict_res['img']
    original_label = dict_res['original_label']
    predicted_label = dict_res['predicted_label']

    if not os.path.exists('static'):
        os.makedirs('static')

    # parse in html the images
    for i in range(N):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(imgs[i])
        ax.set_title(f'Label: {original_label[i]}, Predicted: {predicted_label[i]}')
        plt.savefig(f'static/{i}.png')
        matplotlib.pyplot.close()

    html_text = ''
    for i in range(N):
        html_text += f'<img src="static/{i}.png" alt="img{i}" width="500" height="600">'
    return html_text
        
        
    
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='checkpoint', help='path to directory with model and vocabularies')
args = parser.parse_args()

if __name__ == '__main__':
    MODEL_PATH = args.model_path
    from waitress import serve
    serve(app, host='127.0.0.1', port=5000)
    # app.run(host='127.0.0.1', port=5000)