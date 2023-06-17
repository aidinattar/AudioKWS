import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D,\
                                    Flatten, Dense, Input,\
                                    Resizing, Attention

from model import Model
from utils.custom_layers import LowRankDense
import tensorflow.keras.layers as layers
"""

model options:

DNNBaseline
CNNTradFPool3
CNNOneFPool3
CNNOneFStride4
CNNOneFStride8
CNNOneTStride2
CNNOneTStride4
CNNOneTStride8
CNNTPool2
CNNTPool3
vision_transformer
SpatialTransformerCNN
"""

class DNNBaseline(Model):
    """
    A class to manage the model cnn-trad-fpool3 described in [Sainath15].
    """
    def init(self,
             train_ds,
             val_ds,
             test_ds,
             commands):
        """
        Initialize the class.
        
        Parameters
        ----------
        train_ds : tf.data.Dataset
            A dataset of training data.
        val_ds : tf.data.Dataset
            A dataset of validation data.
        test_ds : tf.data.Dataset
            A dataset of test data.
        commands : list
            A list of commands.
        """
        super().__init__(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            commands=commands
        )


    def define_model(self):
        """Define the model."""

        # Create an input layer with the specified input shape
        input = Input(self.input_shape)
        
        # Downsample the input spectrogram to 32x32
        x = Resizing(32, 32)(input)

        # Normalize the input spectrogram
        x = self.norm_layer(x)

        # Flatten the input spectrogram
        x = Flatten()(x)
        
        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a final fully connected layer with the number of output classes and a softmax activation function
        outputs = Dense(units=self.num_classes, activation='softmax')(x)

        # Create a model with the specified inputs and outputs
        self.model = tf.keras.Model(inputs=input, outputs=outputs)


class CNNTradFPool3(Model):
    """A class to manage the model cnn-trad-fpool3 described in [Sainath15]."""
    def init(self,
             train_ds,
             val_ds,
             test_ds,
             commands):
        """
        Initialize the class.
        
        Parameters
        ----------
        train_ds : tf.data.Dataset
            A dataset of training data.
        val_ds : tf.data.Dataset
            A dataset of validation data.
        test_ds : tf.data.Dataset
            A dataset of test data.
        commands : list
            A list of commands.
        """
        super().__init__(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            commands=commands
        )


    def define_model(self):
        """Define the model."""

        # Create an input layer with the specified input shape
        input = Input(self.input_shape)

        # Downsample the input spectrogram to 32x32
        x = Resizing(32, 32)(input)

        # Normalize the input spectrogram
        x = self.norm_layer(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 20x8
        x = Conv2D(filters=64, kernel_size=(20, 8), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x3
        x = MaxPooling2D(pool_size=(1, 3))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Conv2D(filters=64, kernel_size=(10, 4), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x1
        x = MaxPooling2D(pool_size=(1, 1))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Flatten()(x)

        # Create a Low Rank Dense layer with 32 units and a rank of 16
        x = LowRankDense(units=32, rank=3)(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a final fully connected layer with the number of output classes and a softmax activation function
        outputs = Dense(units=self.num_classes, activation='softmax')(x)

        # Create a model with the specified inputs and outputs
        self.model = tf.keras.Model(inputs=input, outputs=outputs)



class CNNOneFPool3(Model):
    """A class to manage the model cnn-one-fpool3 described in [Sainath15]."""
    def init(self,
             train_ds,
             val_ds,
             test_ds,
             commands):
        """
        Initialize the class.
        
        Parameters
        ----------
        train_ds : tf.data.Dataset
            A dataset of training data.
        val_ds : tf.data.Dataset
            A dataset of validation data.
        test_ds : tf.data.Dataset
            A dataset of test data.
        commands : list
            A list of commands.
        """
        super().__init__(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            commands=commands
        )
        

    def define_model(self):
        """Define the model."""

        # Create an input layer with the specified input shape
        input = Input(self.input_shape)
        # Downsample the input spectrogram to 32x32
        x = Resizing(32, 32)(input)

        # Normalize the input spectrogram
        x = self.norm_layer(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 20x8
        x = Conv2D(filters=54, kernel_size=(32, 8), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x3
        x = MaxPooling2D(pool_size=(1, 3))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Flatten()(x)

        # Create a Low Rank Dense layer with 32 units and a rank of 16
        x = LowRankDense(units=32, rank=1)(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a final fully connected layer with the number of output classes and a softmax activation function
        outputs = Dense(units=self.num_classes, activation='softmax')(x)

        # Create a model with the specified inputs and outputs
        self.model = tf.keras.Model(inputs=input, outputs=outputs)


# try to add a conv and delete a dense
class CNNOneFStride4_2conv(Model):
    """A class to manage the model cnn-one-fstride4 described in [Sainath15]."""
    def init(self,
             train_ds,
             val_ds,
             test_ds,
             commands):
        """
        Initialize the class.
        
        Parameters
        ----------
        train_ds : tf.data.Dataset
            A dataset of training data.
        val_ds : tf.data.Dataset
            A dataset of validation data.
        test_ds : tf.data.Dataset
            A dataset of test data.
        commands : list
            A list of commands.
        """
        super().__init__(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            commands=commands
        )
        

    def define_model(self):
        """Define the model."""

        # Create an input layer with the specified input shape
        input = Input(self.input_shape)

        # Downsample the input spectrogram to 32x32
        x = Resizing(32, 32)(input)

        # Normalize the input spectrogram
        x = self.norm_layer(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 20x8
        x = Conv2D(filters=186, kernel_size=(16, 8), strides=(1, 4), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x1
        x = MaxPooling2D(pool_size=(1, 1))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Conv2D(filters=78, kernel_size=(5, 4), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x1
        x = MaxPooling2D(pool_size=(1, 1))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Flatten()(x)

        # Create a Low Rank Dense layer with 32 units and a rank of 16
        x = LowRankDense(units=32, rank=1)(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a final fully connected layer with the number of output classes and a softmax activation function
        outputs = Dense(units=self.num_classes, activation='softmax')(x)

        # Create a model with the specified inputs and outputs
        self.model = tf.keras.Model(inputs=input, outputs=outputs)
        
class CNNOneFStride4(Model):
    """A class to manage the model cnn-one-fstride4 described in [Sainath15]."""
    def init(self,
             train_ds,
             val_ds,
             test_ds,
             commands):
        """
        Initialize the class.
        
        Parameters
        ----------
        train_ds : tf.data.Dataset
            A dataset of training data.
        val_ds : tf.data.Dataset
            A dataset of validation data.
        test_ds : tf.data.Dataset
            A dataset of test data.
        commands : list
            A list of commands.
        """
        super().__init__(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            commands=commands
        )
        

    def define_model(self):
        """Define the model."""

        # Create an input layer with the specified input shape
        input = Input(self.input_shape)

        # Downsample the input spectrogram to 32x32
        x = Resizing(32, 32)(input)

        # Normalize the input spectrogram
        x = self.norm_layer(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 20x8
        x = Conv2D(filters=186, kernel_size=(32, 8), strides=(1, 4), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x1
        x = MaxPooling2D(pool_size=(1, 1))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Flatten()(x)

        # Create a Low Rank Dense layer with 32 units and a rank of 16
        x = LowRankDense(units=32, rank=1)(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a final fully connected layer with the number of output classes and a softmax activation function
        outputs = Dense(units=self.num_classes, activation='softmax')(x)

        # Create a model with the specified inputs and outputs
        self.model = tf.keras.Model(inputs=input, outputs=outputs)


class CNNOneFStride8(Model):
    """A class to manage the model cnn-one-fstride8 described in [Sainath15]."""
    def init(self,
             train_ds,
             val_ds,
             test_ds,
             commands):
        """
        Initialize the class.
        
        Parameters
        ----------
        train_ds : tf.data.Dataset
            A dataset of training data.
        val_ds : tf.data.Dataset
            A dataset of validation data.
        test_ds : tf.data.Dataset
            A dataset of test data.
        commands : list
            A list of commands.
        """
        super().__init__(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            commands=commands
        )
        

    def define_model(self):
        """Define the model."""

        # Create an input layer with the specified input shape
        input = Input(self.input_shape)

        # Downsample the input spectrogram to 32x32
        x = Resizing(32, 32)(input)

        # Normalize the input spectrogram
        x = self.norm_layer(x)

        # Create a 2D Convolutional layer with 336 filters and a kernel size of 32x8
        x = Conv2D(filters=336, kernel_size=(32, 8), strides=(1, 8), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x3
        x = MaxPooling2D(pool_size=(1, 1))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Flatten()(x)

        # Create a Low Rank Dense layer with 32 units and a rank of 16
        x = LowRankDense(units=32, rank=1)(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a final fully connected layer with the number of output classes and a softmax activation function
        outputs = Dense(units=self.num_classes, activation='softmax')(x)

        # Create a model with the specified inputs and outputs
        self.model = tf.keras.Model(inputs=input, outputs=outputs)


"""
Two convs and stride 2, 4, 8
"""
class CNNOneTStride2(Model):
    """A class to manage the model cnn-one-tstride2 described in [Sainath15]."""
    def init(self,
             train_ds,
             val_ds,
             test_ds,
             commands):
        """
        Initialize the class.
        
        Parameters
        ----------
        train_ds : tf.data.Dataset
            A dataset of training data.
        val_ds : tf.data.Dataset
            A dataset of validation data.
        test_ds : tf.data.Dataset
            A dataset of test data.
        commands : list
            A list of commands.
        """
        super().__init__(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            commands=commands
        )
        

    def define_model(self):
        """Define the model."""

        # Create an input layer with the specified input shape
        input = Input(self.input_shape)

        # Downsample the input spectrogram to 32x32
        x = Resizing(32, 32)(input)

        # Normalize the input spectrogram
        x = self.norm_layer(x)


        # Create a 2D Convolutional layer with 78 filters and a kernel size of 16x8
        x = Conv2D(filters=78, kernel_size=(16, 8), strides=(2, 1), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x3
        x = MaxPooling2D(pool_size=(1, 3))(x)

        # Create a 2D Convolutional layer with 78 filters and a kernel size of 10x4
        x = Conv2D(filters=78, kernel_size=(9, 4), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x1
        x = MaxPooling2D(pool_size=(1, 1))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Flatten()(x)

        # Create a Low Rank Dense layer with 32 units and a rank of 16
        x = LowRankDense(units=32, rank=2)(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a final fully connected layer with the number of output classes and a softmax activation function
        outputs = Dense(units=self.num_classes, activation='softmax')(x)

        # Create a model with the specified inputs and outputs
        self.model = tf.keras.Model(inputs=input, outputs=outputs)


class CNNOneTStride4(Model):
    """A class to manage the model cnn-one-tstride4 described in [Sainath15]."""
    def init(self,
             train_ds,
             val_ds,
             test_ds,
             commands):
        """
        Initialize the class.
        
        Parameters
        ----------
        train_ds : tf.data.Dataset
            A dataset of training data.
        val_ds : tf.data.Dataset
            A dataset of validation data.
        test_ds : tf.data.Dataset
            A dataset of test data.
        commands : list
            A list of commands.
        """
        super().__init__(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            commands=commands
        )
        

    def define_model(self):
        """Define the model."""

        # Create an input layer with the specified input shape
        input = Input(self.input_shape)

        # Downsample the input spectrogram to 32x32
        x = Resizing(32, 32)(input)

        # Normalize the input spectrogram
        x = self.norm_layer(x)


        # Create a 2D Convolutional layer with 100 filters and a kernel size of 16x8
        x = Conv2D(filters=100, kernel_size=(16, 8), strides=(4, 1), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x3
        x = MaxPooling2D(pool_size=(1, 3))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Conv2D(filters=78, kernel_size=(5, 4), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x1
        x = MaxPooling2D(pool_size=(1, 1))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Flatten()(x)

        # Create a Low Rank Dense layer with 32 units and a rank of 16
        x = LowRankDense(units=32, rank=4)(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a final fully connected layer with the number of output classes and a softmax activation function
        outputs = Dense(units=self.num_classes, activation='softmax')(x)

        # Create a model with the specified inputs and outputs
        self.model = tf.keras.Model(inputs=input, outputs=outputs)


class CNNOneTStride8(Model):
    """A class to manage the model cnn-one-tstride8 described in [Sainath15]."""
    def init(self,
             train_ds,
             val_ds,
             test_ds,
             commands):
        """
        Initialize the class.
        
        Parameters
        ----------
        train_ds : tf.data.Dataset
            A dataset of training data.
        val_ds : tf.data.Dataset
            A dataset of validation data.
        test_ds : tf.data.Dataset
            A dataset of test data.
        commands : list
            A list of commands.
        """
        super().__init__(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            commands=commands
        )
        

    def define_model(self):
        """Define the model."""

        # Create an input layer with the specified input shape
        input = Input(self.input_shape)

        # Downsample the input spectrogram to 32x32
        x = Resizing(32, 32)(input)

        # Normalize the input spectrogram
        x = self.norm_layer(x)

        # Create a 2D Convolutional layer with 126 filters and a kernel size of 16x8
        x = Conv2D(filters=126, kernel_size=(16, 8), strides=(8, 1), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x3
        x = MaxPooling2D(pool_size=(1, 3))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Conv2D(filters=78, kernel_size=(5, 4), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x1
        x = MaxPooling2D(pool_size=(1, 1))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Flatten()(x)

        # Create a Low Rank Dense layer with 32 units and a rank of 16
        x = LowRankDense(units=32, rank=9)(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a final fully connected layer with the number of output classes and a softmax activation function
        outputs = Dense(units=self.num_classes, activation='softmax')(x)

        # Create a model with the specified inputs and outputs
        self.model = tf.keras.Model(inputs=input, outputs=outputs)

class CNNOneTStride8Attention(Model):
    """A class to manage the model cnn-one-tstride8 described in [Sainath15]."""
    def init(self,
             train_ds,
             val_ds,
             test_ds,
             commands):
        """
        Initialize the class.
        
        Parameters
        ----------
        train_ds : tf.data.Dataset
            A dataset of training data.
        val_ds : tf.data.Dataset
            A dataset of validation data.
        test_ds : tf.data.Dataset
            A dataset of test data.
        commands : list
            A list of commands.
        """
        super().__init__(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            commands=commands
        )
        

    def define_model(self):
        """Define the model."""

        # Create an input layer with the specified input shape
        input = Input(self.input_shape)

        # Downsample the input spectrogram to 32x32
        x = Resizing(32, 32)(input)

        # Normalize the input spectrogram
        x = self.norm_layer(x)

        # Create a 2D Convolutional layer with 126 filters and a kernel size of 16x8
        x = Conv2D(filters=126, kernel_size=(16, 8), strides=(8, 1), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x3
        x = MaxPooling2D(pool_size=(1, 3))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Conv2D(filters=78, kernel_size=(5, 4), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x1
        x = MaxPooling2D(pool_size=(1, 1))(x)

        # attention 
        x = Attention()([x,x,x])

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Flatten()(x)

        # Create a Low Rank Dense layer with 32 units and a rank of 16
        x = LowRankDense(units=32, rank=9)(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a final fully connected layer with the number of output classes and a softmax activation function
        outputs = Dense(units=self.num_classes, activation='softmax')(x)

        # Create a model with the specified inputs and outputs
        self.model = tf.keras.Model(inputs=input, outputs=outputs)

class CNNOneTStride8DoubleAttention(Model):
    """A class to manage the model cnn-one-tstride8 described in [Sainath15]."""
    def init(self,
             train_ds,
             val_ds,
             test_ds,
             commands):
        """
        Initialize the class.
        
        Parameters
        ----------
        train_ds : tf.data.Dataset
            A dataset of training data.
        val_ds : tf.data.Dataset
            A dataset of validation data.
        test_ds : tf.data.Dataset
            A dataset of test data.
        commands : list
            A list of commands.
        """
        super().__init__(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            commands=commands
        )
        

    def define_model(self):
        """Define the model."""

        # Create an input layer with the specified input shape
        input = Input(self.input_shape)

        # Downsample the input spectrogram to 32x32
        x = Resizing(32, 32)(input)
        

        # Normalize the input spectrogram
        x = self.norm_layer(x)
    

        # Create a 2D Convolutional layer with 126 filters and a kernel size of 16x8
        x = Conv2D(filters=126, kernel_size=(16, 8), strides=(8, 1), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x3
        x = MaxPooling2D(pool_size=(1, 3))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Conv2D(filters=78, kernel_size=(5, 4), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x1
        x = MaxPooling2D(pool_size=(1, 1))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Flatten()(x)

        # Create a Low Rank Dense layer with 32 units and a rank of 16
        x = LowRankDense(units=32, rank=9)(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)


        # Create a final fully connected layer with the number of output classes and a softmax activation function
        outputs = Dense(units=self.num_classes, activation='softmax')(x)

        # Create a model with the specified inputs and outputs
        self.model = tf.keras.Model(inputs=input, outputs=outputs)

class CNNTPool2(Model):
    """A class to manage the model cnn-tpool3 described in [Sainath15]."""
    def init(self,
             train_ds,
             val_ds,
             test_ds,
             commands):
        """
        Initialize the class.
        
        Parameters
        ----------
        train_ds : tf.data.Dataset
            A dataset of training data.
        val_ds : tf.data.Dataset
            A dataset of validation data.
        test_ds : tf.data.Dataset
            A dataset of test data.
        commands : list
            A list of commands.
        """
        super().__init__(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            commands=commands
        )
        

    def define_model(self):
        """Define the model."""

        # Create an input layer with the specified input shape
        input = Input(self.input_shape)

        # Downsample the input spectrogram to 32x32
        x = Resizing(32, 32)(input)

        # Normalize the input spectrogram
        x = self.norm_layer(x)

        # Create a 2D Convolutional layer with 94 filters and a kernel size of 21x8
        x = Conv2D(filters=94, kernel_size=(21, 8), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x3
        x = MaxPooling2D(pool_size=(2, 3))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Conv2D(filters=94, kernel_size=(6, 4), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x1
        x = MaxPooling2D(pool_size=(1, 1))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Flatten()(x)

        # Create a Low Rank Dense layer with 32 units and a rank of 16
        x = LowRankDense(units=32, rank=3)(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a final fully connected layer with the number of output classes and a softmax activation function
        outputs = Dense(units=self.num_classes, activation='softmax')(x)

        # Create a model with the specified inputs and outputs
        self.model = tf.keras.Model(inputs=input, outputs=outputs)


class CNNTPool3(Model):
    """A class to manage the model cnn-tpool3 described in [Sainath15]."""
    def init(self,
             train_ds,
             val_ds,
             test_ds,
             commands):
        """
        Initialize the class.
        
        Parameters
        ----------
        train_ds : tf.data.Dataset
            A dataset of training data.
        val_ds : tf.data.Dataset
            A dataset of validation data.
        test_ds : tf.data.Dataset
            A dataset of test data.
        commands : list
            A list of commands.
        """
        super().__init__(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            commands=commands
        )
        

    def define_model(self):
        """Define the model."""

        # Create an input layer with the specified input shape
        input = Input(self.input_shape)

        # Downsample the input spectrogram to 32x32
        x = Resizing(32, 32)(input)

        # Normalize the input spectrogram
        x = self.norm_layer(x)

        # Create a 2D Convolutional layer with 94 filters and a kernel size of 21x8
        x = Conv2D(filters=94, kernel_size=(21, 8), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x3
        x = MaxPooling2D(pool_size=(3, 3))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Conv2D(filters=94, kernel_size=(6, 4), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x1
        x = MaxPooling2D(pool_size=(1, 1))(x)

        # Create a flatten layer
        x = Flatten()(x)

        # Create a Low Rank Dense layer with 32 units and a rank of 3
        x = LowRankDense(units=32, rank=3)(x)

        # Create a fully connected layer with 128 units and a ReLU activation function
        x = Dense(units=128, activation='relu')(x)

        # Create a final fully connected layer with the number of output classes and a softmax activation function
        outputs = Dense(units=self.num_classes, activation='softmax')(x)

        # Create a model with the specified inputs and outputs
        self.model = tf.keras.Model(inputs=input, outputs=outputs)


from ViT import VisionTransformer
class vision_transformer(Model):

    def init(self,
             train_ds,
             val_ds,
             test_ds,
             commands,
             ):
        """
        Initialize the class.
        
        Parameters
        ----------
        train_ds : tf.data.Dataset
            A dataset of training data.
        val_ds : tf.data.Dataset
            A dataset of validation data.
        test_ds : tf.data.Dataset
            A dataset of test data.
        commands : list
            A list of commands.
        """
        super().__init__(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            commands=commands
        )

    def define_model(self):
        self.vit = VisionTransformer(
                                    image_size = 10,
                                    patch_size = 8,
                                    num_layers = 2,
                                    num_classes = 35,
                                    d_model = 64,
                                    num_heads = 4,
                                    mlp_dim = 64,
                                    channels=3,
                                    dropout=0.1,
                                    )

        input = Input(self.input_shape)
        # x = Resizing(40, 40)(input)
        # print (input.shape)
        # outputs = self.vit(x)
        
        # self.model = tf.keras.Model(inputs=input, outputs=outputs)

        # Normalize the input spectrogram
        x = self.norm_layer(input)

        # Create a 2D Convolutional layer with 126 filters and a kernel size of 16x8
        x = Conv2D(filters=32, kernel_size=(16, 8), strides=(4, 1), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x3
        x = MaxPooling2D(pool_size=(1, 3))(x)

        # Create a 2D Convolutional layer with 64 filters and a kernel size of 10x4
        x = Conv2D(filters=3, kernel_size=(5, 4), activation='relu', padding='same')(x)

        # Create a Max Pooling layer with a pool size of 1x1
        x = MaxPooling2D(pool_size=(1, 1))(x)

        # resize to min size
        x_shape = min(x.shape[1], x.shape[2])
        x = Resizing(x_shape, x_shape)(x)

        print (x.shape)
        outputs = self.vit(x)
        
        self.model = tf.keras.Model(inputs=input, outputs=outputs)
        

class SpatialTransformerCNN(Model):
    def init(self,
             train_ds,
             val_ds,
             test_ds,
             commands,
             ):
        """
        Initialize the class.
        
        Parameters
        ----------
        train_ds : tf.data.Dataset
            A dataset of training data.
        val_ds : tf.data.Dataset
            A dataset of validation data.
        test_ds : tf.data.Dataset
            A dataset of test data.
        commands : list
            A list of commands.
        """
        super().__init__(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            commands=commands
        )


    # Spatial transformer localization-network
    def get_localization_network(self, ):
        localization = tf.keras.Sequential([
            layers.Conv2D(8, kernel_size=7, input_shape=(28, 28, 1), 
                        activation="relu", kernel_initializer="he_normal"),
            layers.MaxPool2D(strides=2),
            layers.Conv2D(10, kernel_size=5, activation="relu", kernel_initializer="he_normal"),
            layers.MaxPool2D(strides=2),
        ])
        return localization

    # Regressor for the 3 * 2 affine matrix
    def get_affine_params(self):
        output_bias = tf.keras.initializers.Constant([1, 0, 0, 0, 1, 0])
        fc_loc = tf.keras.Sequential([
            layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
            layers.Dense(3 * 2, kernel_initializer="zeros", bias_initializer=output_bias)
        ])

        return fc_loc
    

    def get_pixel_value(self, img, x, y):
        """
        Utility function to get pixel value for coordinate
        vectors x and y from a  4D tensor image.
        Input
        -----
        - img: tensor of shape (B, H, W, C)
        - x: flattened tensor of shape (B*H*W,)
        - y: flattened tensor of shape (B*H*W,)
        Returns
        -------
        - output: tensor of shape (B, H, W, C)
        """
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, height, width))

        indices = tf.stack([b, y, x], 3)

        return tf.gather_nd(img, indices)
     
    
    def affine_grid_generator(self, height, width, theta):
        """
        This function returns a sampling grid, which when
        used with the bilinear sampler on the input feature
        map, will create an output feature map that is an
        affine transformation [1] of the input feature map.
        Input
        -----
        - height: desired height of grid/output. Used
        to downsample or upsample.
        - width: desired width of grid/output. Used
        to downsample or upsample.
        - theta: affine transform matrices of shape (num_batch, 2, 3).
        For each image in the batch, we have 6 theta parameters of
        the form (2x3) that define the affine transformation T.
        Returns
        -------
        - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
        The 2nd dimension has 2 components: (x, y) which are the
        sampling points of the original image for each point in the
        target image.
        Note
        ----
        [1]: the affine transformation allows cropping, translation,
            and isotropic scaling.
        """
        num_batch = tf.shape(theta)[0]

        # create normalized 2D grid
        x = tf.linspace(-1.0, 1.0, width)
        y = tf.linspace(-1.0, 1.0, height)
        x_t, y_t = tf.meshgrid(x, y)

        # flatten
        x_t_flat = tf.reshape(x_t, [-1])
        y_t_flat = tf.reshape(y_t, [-1])

        # reshape to [x_t, y_t , 1] - (homogeneous form)
        ones = tf.ones_like(x_t_flat)
        sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

        # repeat grid num_batch times
        sampling_grid = tf.expand_dims(sampling_grid, axis=0)
        sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

        # cast to float32 (required for matmul)
        theta = tf.cast(theta, 'float32')
        sampling_grid = tf.cast(sampling_grid, 'float32')

        # transform the sampling grid - batch multiply
        batch_grids = tf.matmul(theta, sampling_grid)
        # batch grid has shape (num_batch, 2, H*W)

        # reshape to (num_batch, H, W, 2)
        batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

        return batch_grids

        
    def bilinear_sampler(self, img, x, y):
        """
        Performs bilinear sampling of the input images according to the
        normalized coordinates provided by the sampling grid. Note that
        the sampling is done identically for each channel of the input.
        To test if the function works properly, output image should be
        identical to input image when theta is initialized to identity
        transform.
        Input
        -----
        - img: batch of images in (B, H, W, C) layout.
        - grid: x, y which is the output of affine_grid_generator.
        Returns
        -------
        - out: interpolated images according to grids. Same size as grid.
        """
        H = tf.shape(img)[1]
        W = tf.shape(img)[2]
        max_y = tf.cast(H - 1, 'int32')
        max_x = tf.cast(W - 1, 'int32')
        zero = tf.zeros([], dtype='int32')

        # rescale x and y to [0, W-1/H-1]
        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
        y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

        # grab 4 nearest corner points for each (x_i, y_i)
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        # clip to range [0, H-1/W-1] to not violate img boundaries
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        # get pixel value at corner coords
        Ia = self.get_pixel_value(img, x0, y0)
        Ib = self.get_pixel_value(img, x0, y1)
        Ic = self.get_pixel_value(img, x1, y0)
        Id = self.get_pixel_value(img, x1, y1)

        # recast as float for delta calculation
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        # calculate deltas
        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        # add dimension for addition
        wa = tf.expand_dims(wa, axis=3)
        wb = tf.expand_dims(wb, axis=3)
        wc = tf.expand_dims(wc, axis=3)
        wd = tf.expand_dims(wd, axis=3)

        # compute output
        out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

        return out

    def stn(self, x):
        localization = self.get_localization_network()
        fc_loc = self.get_affine_params()
        
        xs = localization(x)
        xs = tf.reshape(xs, (-1, 10 * 3 * 3 ))
        theta = fc_loc(xs)
        theta = tf.reshape(theta, (-1, 2, 3))
        
        grid = self.affine_grid_generator(28, 28, theta)
        x_s = grid[:, 0, :, :]
        y_s = grid[:, 1, :, :]
        x = self.bilinear_sampler(x, x_s, y_s)

        return x
    

    def define_model(self):
        # Create an input layer with the specified input shape
        input = Input(self.input_shape)

        # Downsample the input spectrogram to 32x32
        x = Resizing(28, 28)(input)

        # Normalize the input spectrogram
        x = self.norm_layer(x)

        x = self.stn(x)
        print ("after stn", x.shape)
        x = layers.Conv2D(10, (5, 5), activation="relu", kernel_initializer="he_normal")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(20, (5, 5), activation="relu", kernel_initializer="he_normal")(x)
        x = layers.SpatialDropout2D(0.5)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.reshape(x, (-1, 320))
        x = layers.Dense(50, activation="relu", kernel_initializer="he_normal")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(35, activation="softmax")(x)

        self.model = tf.keras.Model(input, outputs)


     



if __name__ == '__main__':
    # test the model
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras.optimizers import Adam



