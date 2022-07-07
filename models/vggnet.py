import tensorflow as tf
from keras import Sequential, Model, layers


# 构建方式二：函数式API
def maxpool2(input_shape):
    inputs = layers.Input(shape=input_shape)
    conv = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    outputs = layers.MaxPool2D(pool_size=2, strides=2)(conv)
    model = Model(inputs, outputs)
    return model


# 构建方式三：Sequential()
def dense():
    model = Sequential([
        layers.Dense(4096),
        layers.Activation(tf.nn.relu),
        layers.Dropout(0.5),
        layers.Dense(4096),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(1),
        layers.Activation('sigmoid')
    ])
    return model


# 构建方式一：继承keras.Model
class VGG11(Model):
    def __init__(self, img_size=224):
        super(VGG11, self).__init__()
        self.maxpool1 = Sequential([
            layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', input_shape=(img_size, img_size, 3)),
            layers.ReLU(),
            layers.MaxPool2D(pool_size=2, strides=2)
        ])

        self.maxpool2 = maxpool2((img_size//2, img_size//2, 64))

        self.maxpool3 = Sequential([
            layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=2, strides=2)
        ])

        self.maxpool4 = Sequential([
            layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=2, strides=2)
        ])

        self.maxpool5 = Sequential([
            layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=2, strides=2)
        ])

        self.dense = dense()

    def call(self, inputs, training=False, mask=None):
        pool1 = self.maxpool1(inputs)
        pool2 = self.maxpool2(pool1)
        pool3 = self.maxpool3(pool2)
        pool4 = self.maxpool4(pool3)
        pool5 = self.maxpool5(pool4)
        flatten = layers.Flatten()(pool5)
        out = self.dense(flatten)
        return out


if __name__ == "__main__":
    net = VGG11()
    net.build((None, 224, 224, 3))
    net.summary()
