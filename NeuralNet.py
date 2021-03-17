from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, Dense, Dropout, Flatten, Concatenate, Input, GlobalAveragePooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.vgg16 import VGG16

NN_SHAPE = (128, 128, 1)

def get_pretrained_DenseNet121():
    densenet_base = DenseNet121(input_shape = NN_SHAPE, include_top = False, weights = 'imagenet')
    densenet_base.trainable = False

    inputs = Input(shape = NN_SHAPE)

    x = densenet_base(inputs)
    x = GlobalAveragePooling2D()(x)

    outputs = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(optimizer = Adam(lr = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()

    return model

def get_pretrained_ResNet50V2():
    resnet_base = ResNet50(input_shape = NN_SHAPE, include_top = False, weights = 'imagenet')
    resnet_base.trainable = False 

    inputs = Input(shape = NN_SHAPE)

    x = resnet_base(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    outputs = Dense(1, activation = 'sigmoid')(x)
    
    model = Model(inputs, outputs)

    model.compile(optimizer = Adam(lr = 0.00001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()

    return model

def get_pretrained_VGG16():
    vgg16_base = VGG16(input_shape = NN_SHAPE, include_top = False, weights = None)
    vgg16_base.trainable = False

    inputs = Input(shape = NN_SHAPE)

    x = vgg16_base(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation = 'relu')(x)
    x = Dense(4096, activation = 'relu')(x)
    
    outputs = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(optimizer = Adam(lr = 0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()

    return model

def get_neural_net_Arch1():
    """NN used for true/false identification of nodules"""
    #PROOF OF CONCEPT MODEL
    model = Sequential()
    
    model.add(Conv2D(32, (3,3), input_shape = NN_SHAPE, padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(64, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(128, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(256, (3,3), padding = 'same'))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(4096))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(4096))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer = Adam(lr = 0.000005), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    
    return model