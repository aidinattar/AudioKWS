import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D,\
                                    Flatten, Dense, Input,\
                                    Resizing

from model import Model
from utils.custom_layers import LowRankDense


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

if __name__ == '__main__':
    # test the model
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras.optimizers import Adam



