import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense
# import tensorflow.keras as keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense
import tensorflow as tf

'''

'''


class TensorProjectionLayer(tf.keras.layers.Layer):
    # class TensorProjectionLayer(keras.layers.Layer):
    '''
    From Juncheng for SVD
    '''

    def __init__(self, q1, q2, q3, e1, e2, e3):
        super(TensorProjectionLayer, self).__init__()
        self.q1 = int(q1)
        self.q2 = int(q2)
        self.q3 = int(q3)
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3

    def build(self, input_shape):
        self.p1 = int(input_shape[1])
        self.p2 = int(input_shape[2])
        self.p3 = int(input_shape[3])
        self.W1 = self.add_weight(
            "W1", shape=[self.p1, self.q1], initializer='normal', trainable=True)
        self.W2 = self.add_weight(
            "W2", shape=[self.p2, self.q2], initializer='normal', trainable=True)
        self.W3 = self.add_weight(
            "W3", shape=[self.p3, self.q3], initializer='normal', trainable=True)
        super(TensorProjectionLayer, self).build(input_shape)

    # suppose that T: n x t1 x t2 x t3
    # k=1,2,3
    def kmode_product(self, T, A, k):

        # number of the observations
        n = tf.shape(T)[0]
        t1 = tf.shape(T)[1]
        t2 = tf.shape(T)[2]
        t3 = tf.shape(T)[3]
        uk = tf.shape(A)[0]

        # Compute the size of the output-tensor
        if k == 1:
            new_shape = [n, uk, t2, t3]
        elif k == 2:
            new_shape = [n, t1, uk, t3]
        elif k == 3:
            new_shape = [n, t1, t2, uk]

        # Compute T xk A
        A = tf.expand_dims(A, 0)
        An = tf.tile(A, [n, 1, 1])
        Tk = self.unfold(T, k)
        ATk = tf.linalg.matmul(An, Tk)
        TxkA = self.fold(ATk, k, new_shape)
        return TxkA

    def fold(self, Tk, k, new_shape):

        a = new_shape

        if k == 1:
            reshape_order = [0, 1, 2, 3]
            permute_order = [0, 1, 2, 3]
        elif k == 2:
            reshape_order = [0, 2, 1, 3]
            permute_order = [0, 2, 1, 3]
        elif k == 3:
            reshape_order = [0, 3, 1, 2]
            permute_order = [0, 2, 3, 1]

        new_shape = [a[reshape_order[0]], a[reshape_order[1]],
                     a[reshape_order[2]], a[reshape_order[3]]]

        T_ = tf.reshape(Tk, new_shape)
        T = tf.transpose(T_, perm=permute_order)
        return T

    def unfold(self, T, k):
        n = tf.shape(T)[0]
        t1 = tf.shape(T)[1]
        t2 = tf.shape(T)[2]
        t3 = tf.shape(T)[3]

        if k == 1:
            new_shape = [n, t1, t2 * t3]
            A = T
        elif k == 2:
            new_shape = [n, t2, t3 * t1]
            A = tf.transpose(T, perm=[0, 2, 1, 3])
        elif k == 3:
            new_shape = [n, t3, t1 * t2]
            A = tf.transpose(T, perm=[0, 3, 1, 2])

        Tk = tf.reshape(A, new_shape)

        return Tk

    def call(self, X):
        n = tf.shape(X)[0]

        Iq1 = tf.eye(self.q1)
        Iq2 = tf.eye(self.q2)
        Iq3 = tf.eye(self.q3)

        W1T = tf.transpose(self.W1, perm=[1, 0])
        W2T = tf.transpose(self.W2, perm=[1, 0])
        W3T = tf.transpose(self.W3, perm=[1, 0])

        M1 = tf.math.add(tf.linalg.matmul(W1T, self.W1), Iq1 * self.e1)
        M2 = tf.math.add(tf.linalg.matmul(W2T, self.W2), Iq2 * self.e2)
        M3 = tf.math.add(tf.linalg.matmul(W3T, self.W3), Iq3 * self.e3)

        # without penalty
        #M1 = tf.linalg.matmul(W1T,self.W1);
        #M2 = tf.linalg.matmul(W2T,self.W2);
        #M3 = tf.linalg.matmul(W3T,self.W3);

        sqrtM1 = tf.linalg.sqrtm(M1)
        sqrtM2 = tf.linalg.sqrtm(M2)
        sqrtM3 = tf.linalg.sqrtm(M3)

        G1 = tf.linalg.inv(sqrtM1)
        G2 = tf.linalg.inv(sqrtM2)
        G3 = tf.linalg.inv(sqrtM3)

        U1 = tf.linalg.matmul(self.W1, G1)  # p1 x q1
        U2 = tf.linalg.matmul(self.W2, G2)  # p2 x q2
        U3 = tf.linalg.matmul(self.W3, G3)  # p3 x q3

        U1T = tf.transpose(U1, perm=[1, 0])  # q1 x p1
        U2T = tf.transpose(U2, perm=[1, 0])  # q2 x p2
        U3T = tf.transpose(U3, perm=[1, 0])  # q3 x p3

        XU1T = self.kmode_product(X, U1T, 1)
        XU1TU2T = self.kmode_product(XU1T, U2T, 2)
        XU1TU2TU3T = self.kmode_product(XU1TU2T, U3T, 3)

        # This is necessary. But I don't know why...
        Z = tf.reshape(XU1TU2TU3T, [n, self.q1, self.q2, self.q3])

        return Z


def simple_cnn_sigmoid_SVD(input_shape):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu',
                     input_shape=input_shape, padding='same'))
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(TensorProjectionLayer(6, 6, 32, 0.01, 0.01, 0.01))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    return model


def simple_cnn_sigmoid(input_shape):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu',
                     input_shape=input_shape, padding='same'))
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    return model


def simple_cnn_regression(input_shape):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu',
                     input_shape=input_shape, padding='same'))
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    return model


def simple_cnn_softmax(input_shape, num_classes=2):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def alexnet(input_shape):
    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=(1, 1), input_shape=input_shape, padding='same', activation='relu',
                     kernel_initializer='uniform'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model


def VGG(input_shape):
    from keras.applications.vgg16 import VGG16
    import tensorflow as tf
    from keras.models import Model

    original_dim = input_shape
    target_size = (224, 224)
    input = keras.layers.Input(original_dim)
    x = keras.layers.Lambda(
        lambda image: tf.image.resize_images(image, target_size))(input)

    base_model = VGG16(weights='imagenet', input_tensor=x, include_top=False)
    predictions = Dense(2, activation='softmax')(base_model.output)
    model = Model(inputs=input, outputs=predictions)
    # for layer in base_model.layers:
    #     layer.trainable = False

    return model
