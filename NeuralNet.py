from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, Dense, Dropout, Flatten, Concatenate, Input, GlobalAveragePooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet169
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

NN_SHAPE = (128, 128, 1)
NN_SHAPE_RGB = (224, 224, 3)

def get_pretrained_DenseNet169_FC():
    base = get_pretrained_DenseNet169()

    input = Input(shape = (1664,))
    next_input = input

    next_input = base.layers[-3](next_input)
    next_input = base.layers[-2](next_input)
    next_input = base.layers[-1](next_input)

    model = Model(input, next_input)
    model.compile(optimizer = Adam(lr = 0.000001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()

    return model

def get_pretrained_DenseNet169():
    densenet_base = DenseNet169(input_shape = NN_SHAPE_RGB, include_top = False, weights = 'imagenet')
    densenet_base.trainable = False

    inputs = Input(shape = NN_SHAPE_RGB)

    x = densenet_base(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation = 'relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(optimizer = Adam(lr = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()

    return model

def get_pretrained_ResNet50_FC():
    base = get_pretrained_ResNet50()

    input = Input(shape = (2048,))
    next_input = input

    next_input = base.layers[-3](next_input)
    next_input = base.layers[-2](next_input)
    next_input = base.layers[-1](next_input)

    model = Model(input, next_input)
    model.compile(optimizer = Adam(lr = 0.0000001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()

    return model

def get_pretrained_ResNet50():
    resnet_base = ResNet50(input_shape = NN_SHAPE_RGB, include_top = False, weights = 'imagenet')
    resnet_base.trainable = False 

    inputs = Input(shape = NN_SHAPE_RGB)

    x = resnet_base(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation = 'relu')(x)
    x = Dropout(0.75)(x)

    outputs = Dense(1, activation = 'sigmoid')(x)
    
    model = Model(inputs, outputs)

    model.compile(optimizer = Adam(lr = 0.000001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()

    return model

def get_pretrained_VGG16():
    vgg16_base = VGG16(input_shape = NN_SHAPE_RGB, include_top = False, weights = None)
    vgg16_base.trainable = False

    inputs = Input(shape = NN_SHAPE_RGB)

    x = vgg16_base(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation = 'relu')(x)
    x = Dense(4096, activation = 'relu')(x)
    
    outputs = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(optimizer = Adam(lr = 0.00005), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()

    return model

def get_pretrained_VGG19_FC():
    base = get_pretrained_VGG16()

    input = Input(shape = (512,))
    next_input = input

    next_input = base.layers[-3](next_input)
    next_input = Dropout(0.75)(next_input)
    next_input = base.layers[-2](next_input)
    next_input = Dropout(0.75)(next_input)
    next_input = base.layers[-1](next_input)

    model = Model(input, next_input)
    model.compile(optimizer = Adam(lr = 0.0000001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()

    return model


def get_pretrained_VGG19():
    vgg19_base = VGG19(input_shape = NN_SHAPE_RGB, include_top = False, weights = 'imagenet')
    vgg19_base.trainable = False

    inputs = Input(shape = NN_SHAPE_RGB)

    x = vgg19_base(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation = 'relu')(x)
    #x = Dropout(0.5)(x)
    x = Dense(4096, activation = 'relu')(x)
    #x = Dropout(0.5)(x)

    output = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs, output)

    model.compile(optimizer = Adam(lr = 0.000001), loss = 'binary_crossentropy', metrics = ['accuracy'])
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
    model.add(Dropout(0.75))

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.75))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer = Adam(lr = 0.000005), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    
    return model