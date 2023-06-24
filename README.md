# Small-footprint Keyword Spotting with Convolutional Neural Networks
This project focuses on implementing a small-footprint keyword spotting (KWS) engine using convolutional neural networks (CNNs). The goal is to detect specific keywords within speech utterances using machine learning techniques. The project utilizes a reference dataset released by Google called the "Speech Commands Dataset," which consists of 65,000 one-second-long utterances of 30 words collected from thousands of different people. The dataset is released under the Creative Commons 4.0 license.

The project explores different approaches for implementing the KWS engine, including LVCSR-based KWS, Phoneme Recognition-based KWS, and Word Recognition-based KWS. The CNN model proposed by Sainath et al. (2015) is utilized for word recognition, where features are obtained from raw audio data using 40-dimensional log Mel filterbanks coefficients.

## Reference Papers
  - [Sainath15] Tara N. Sainath, Carolina Parada, "Convolutional Neural Networks for Small-footprint Keyword Spotting," INTERSPEECH, Dresden, Germany, September 2015.
  - [Warden18] Pete Warden, "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition," arXiv:1804.03209, April 2018.

## Dataset Description
The reference dataset used for small-footprint keyword spotting is the "Speech Commands Dataset." It was released in August 2017 and contains 65,000 one-second-long utterances of 30 words. The dataset is collected by AYI and released under the Creative Commons 4.0 license. Additional information about the dataset can be found in the Google blog post: Speech Commands Dataset.

The speech dataset can be downloaded from the following link: [Speech Commands Dataset](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) (2.11 GB uncompressed)

Examples of the spectrograms and possible data augmentation tecniques follow:

<p align="center">
<img src="sequential_model/plots/augment.png"  width="800"/> </p>

## Project Developments
The project offers several possible developments and experiments, including:

- Experimenting with different audio features and coefficients.
- Designing custom Mel filterbanks.
- Implementing standard/deep CNN architectures with techniques like dropout and regularization.
- Investigating recent/new artificial neural network (ANN) architectures, such as autoencoder-based models, attention mechanisms, and inception-based CNN networks.
- Conducting a comparison of different architectures based on memory usage and accuracy.

We implement the following pipeline:

## Useful Resources
Recent developments and resources related to keyword spotting and speech recognition:

- [Chorowski15] J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, Y. Bengio, "Attention-Based Models for Speech Recognition," Conference on Neural Information and Processing Systems (NIPS), Montréal, Canada, 2015.
- [Tang18] R. Tang and J. Lin, "Deep residual learning for small-footprint keyword spotting," IEEE ICASSP, Calgary, Alberta, Canada, 2018.
- [Andrade18] D. C. de Andrade, S. Leo, M. L. D. S. Viana, and C. Bernkopf, "A neural attention model for speech command recognition," arXiv:1808.08929, 2018. [PDF Link](https://arxiv.org/pdf/1808.08929.pdf)
- White Paper: "Key-Word Spotting - The Base Technology for Speech Analytics" [PDF Link](https://pdfs.semanticscholar.org/e736/bc0a0cf1f2d867283343faf63211aef8a10c)
