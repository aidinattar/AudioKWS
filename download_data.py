from dataset import DataLoader, DataVisualizer, DatasetBuilder
#from models import *
import tensorflow as tf
from utils.input import *

if __name__ == '__main__':
    data = DataLoader(path='DATA/speech_commands_v0.02')
    data.download_data()