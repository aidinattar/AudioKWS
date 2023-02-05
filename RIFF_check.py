import os
import tensorflow as tf
from tensorflow.audio import decode_wav

root_dir = 'DATA/speech_commands_v0.02'
filenames = tf.io.gfile.glob(str(root_dir) + '/*/*')
n = 0
k = 0

for file in filenames:
    if file.endswith('.wav'):
        # Load the wave file as a binary string
        wav_binary = tf.io.read_file(file)
        k += 1
        try:
            wav_tensor = decode_wav(wav_binary)
        except Exception as e:
            if 'Header mismatch: Expected RIFF but found' in str(e):
                n += 1
                print(f'Error in file: {file}"\n"')

print(f'Number of files: {k}')
print(f'Number of files with error: {n}')