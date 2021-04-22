#3rd party imports
import medpy
import medpy.io
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import math
import gc
from scipy import ndimage

#KERAS
from keras.callbacks import EarlyStopping
from keras.applications.vgg19 import preprocess_input
#from keras.applications.densenet import preprocess_input
#from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.layers import Input

#My imports
import NeuralNet

#=============== GLOBAL VARIABLES ===============
#Saves training stats after every epoch into file
TRAINING_HISTORY = True
TRAINING_HISTORY_VAL_ACC = 'val_acc_training.csv'
TRAINING_HISTORY_ACC = 'acc_training.csv'
TRAINING_HISTORY_VAL_LOSS = 'val_loss_training.csv'
TRAINING_HISTORY_LOSS = 'loss_training.csv'

#Label set implementation
ONE_HOT = 0
HEATMAP = 1

#Data augumentation
NONE = 0
UNDERSAMPLING = 1
OVERSAMPLING = 2

#CNN Model
SAVE_MODELS = True
MODEL_SAVE_PATH = '.old_models'                     #Folder used to store trained models
EXTRACTED_FEATURES_PATH = '.extracted_features'     #Folder used to store extracted features

#Dataset
DATASETS_FOLDER = 'D:/LungCancerCTScans/LUNA16/augumented_extended' #Folder holding '.p' patient files. 


class NeuralNetManager:
    """Manager used to create and train new neural networks using CTScanModules"""

    def __init__(self):
        self._ct_scan_list = []
        self._last_used_scan = -1 #K-Fold validation
        self._number_of_folds = 5


    def extract_features(self):
        start_fold = 0
        
        #Misc
        if (EXTRACTED_FEATURES_PATH not in os.listdir('.')):
            os.mkdir(EXTRACTED_FEATURES_PATH)

        ##Loading and creating an extractor
        #DenseNet169
        #model = NeuralNet.get_pretrained_DenseNet169()
        #extractor = Model(model.inputs, model.layers[-3].output) #Input -> GlobalAveragePooling2D
        
        #VGG19
        #model = NeuralNet.get_pretrained_VGG19()
        #extractor = Model(model.inputs, model.layers[-4].output) #Input -> GlobalAveragePooling2D

        #ResNet50
        model = NeuralNet.get_pretrained_ResNet50()
        extractor = Model(model.inputs, model.layers[-4].output) #Input -> GlobalAveragePooling2D
        extractor.summary()
        
        dataset_filenames = os.listdir(DATASETS_FOLDER)
        dataset_count = len(os.listdir(DATASETS_FOLDER))
        
        features_name = "VGG19_test"
        counter = 1
        patient_count = len(dataset_filenames)
        for patient_file in dataset_filenames:
            print("**Patient data: {} / {}".format(counter, patient_count))
            patient_data = self.load_dataset_single(os.path.join(DATASETS_FOLDER, patient_file))

            X = np.array(patient_data[0])
            Y = np.array(patient_data[1])

            patient_data_enhanced = self.random_sampler(X, Y, OVERSAMPLING)
            #If a patient doesnt have any nodules, dont oversample it.
            if patient_data_enhanced is None:
                X = np.array(patient_data[0])
                Y = np.array(patient_data[1])
            else:
                X = patient_data_enhanced[0]
                Y = patient_data_enhanced[1]

            #Data augumentation - Random image rotation
            random_rotation = True
            if (random_rotation):
                #Iterates over all images in feature sets and rotates them by random degree
                print_counter = 0
                print("**Rotating feature set.")
                for train_img in X:
                    print("\r**Rotating image: {}/{}".format(print_counter + 1, X.shape[0]), end = '')
                    if (print_counter + 1 == X.shape[0]):
                        print("...DONE")
                    random_angle = random.randint(0, 360)
                    train_img = ndimage.rotate(train_img, random_angle, order = 0)
                    print_counter += 1

            #In case the model needs RGB images!
            rgb_input = True
            if (rgb_input):
                #training
                X = np.repeat(X[..., np.newaxis], 3, -1)
            else:
                X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))

            #PREPROCESS FOR 3RD PARTY NEURAL NETS
            X = preprocess_input(X)

            unique, counts = np.unique(Y, return_counts=True)
            print(dict(zip(unique, counts)))

            extracted_features = extractor.predict(X, batch_size = 2, verbose = 1)

            #Saving the extracted features along with label set.
            feature_file = open(os.path.join(EXTRACTED_FEATURES_PATH, "{}_{}.feat".format(features_name, patient_file)), 'wb')
            #(X_extracted, Y)
            data_to_save = (extracted_features, Y)
            pickle.dump(data_to_save, feature_file)
            feature_file.close()
            counter += 1
                

    def train_model_extracted_features(self):
        start_fold = 0
        #for current_fold in range(start_fold, self._number_of_folds):
        for current_fold in range(start_fold, self._number_of_folds):
            model_name = 'LUNA16_ResNet50_Experiment_11'
            data_name = 'ResNet50'
            data = self.load_extracted_features(data_name, current_fold)

            unique, counts = np.unique(data[0][1], return_counts=True)
            print(dict(zip(unique, counts)))

            undersampled_train = self.random_sampler(data[0][0], data[0][1], UNDERSAMPLING)            
            undersampled_validation = self.random_sampler(data[1][0], data[1][1], UNDERSAMPLING)
            
            X_train = undersampled_train[0]
            Y_train = undersampled_train[1]

            X_validation = undersampled_validation[0]
            Y_validation = undersampled_validation[1]

            unique, counts = np.unique(Y_train, return_counts=True)
            print(dict(zip(unique, counts)))
            
            #Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

            #Training
            model = NeuralNet.get_pretrained_ResNet50_FC()
            history = model.fit(X_train, Y_train,
                                batch_size = 64,
                                epochs = 100,
                                verbose = 1,
                                validation_data = (X_validation, Y_validation),
                                shuffle = True,
                                callbacks = [early_stopping])
            
            if (SAVE_MODELS and MODEL_SAVE_PATH not in os.listdir('.')):
                os.mkdir(MODEL_SAVE_PATH)
                model.save(os.path.join(MODEL_SAVE_PATH, "{}_fold{}.h5".format(model_name, current_fold)))
            elif (SAVE_MODELS):
                model.save(os.path.join(MODEL_SAVE_PATH, "{}_fold{}.h5".format(model_name, current_fold)))

            #Saving
            if (TRAINING_HISTORY):
                self.save_training_statistics(TRAINING_HISTORY_VAL_ACC, history.history['val_acc'], model_name, current_fold)
                self.save_training_statistics(TRAINING_HISTORY_ACC, history.history['acc'], model_name, current_fold)
                self.save_training_statistics(TRAINING_HISTORY_VAL_LOSS, history.history['val_loss'], model_name, current_fold)
                self.save_training_statistics(TRAINING_HISTORY_LOSS, history.history['loss'], model_name, current_fold)

    def train_model(self):
        model_name = 'LUNA16_VGG19_0255'
        #Used for statistics
        start_fold = 4
        for current_fold in range(start_fold, self._number_of_folds):
            #Data
            data = self.load_dataset(current_fold, DATASETS_FOLDER, OVERSAMPLING)
            
            feature_set_training = data[0][0]
            label_set_training = data[0][1]

            feature_set_validation = data[1][0]
            label_set_validation = data[1][1]

            #In case the model needs RGB images!
            rgb_input = True
            if (rgb_input):
                #training
                print(feature_set_training.shape)  # (x, 128, 128, 1)
                feature_set_training = np.squeeze(feature_set_training, axis = 3)
                print(feature_set_training.shape)  # (x, 128, 128)
                feature_set_training = np.repeat(feature_set_training[..., np.newaxis], 3, -1)
                print(feature_set_training.shape)  # (x, 128, 128, 3)
                
                #validation
                print(feature_set_validation.shape)  # (x, 128, 128, 1)
                feature_set_validation = np.squeeze(feature_set_validation, axis = 3)
                print(feature_set_validation.shape)  # (x, 128, 128)
                feature_set_validation = np.repeat(feature_set_validation[..., np.newaxis], 3, -1)
                print(feature_set_validation.shape)  # (x, 128, 128, 3)

            
            #PREPROCESS FOR 3RD PARTY NEURAL NETS
            feature_set_training = preprocess_input(feature_set_training)
            feature_set_validation = preprocess_input(feature_set_validation)
                
            unique, counts = np.unique(label_set_training, return_counts=True)
            print(dict(zip(unique, counts)))

            #DEBUG
            #Checking if the dataset is alright
            #for i in range(0, len(label_set_training)):
            #    if label_set_training[i] == True:
            #        plt.imshow(feature_set_training[i], cmap = 'gray')
            #        plt.show()
            #for i in range(0, len(label_set_training)):
            #    if label_set_training[i] == True:
            #        print(label_set_training[i])
            #        print(label_set_training[i - 1])

            #Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

            #Training
            model = NeuralNet.get_pretrained_VGG19()
            history = model.fit(feature_set_training, label_set_training,
                                batch_size = 64,
                                epochs = 100,
                                verbose = 1,
                                validation_data = (feature_set_validation, label_set_validation),
                                shuffle = True,
                                callbacks = [early_stopping])
            
            if (SAVE_MODELS and MODEL_SAVE_PATH not in os.listdir('.')):
                os.mkdir(MODEL_SAVE_PATH)
                model.save(os.path.join(MODEL_SAVE_PATH, "{}_fold{}.h5".format(model_name, current_fold)))
            elif (SAVE_MODELS):
                model.save(os.path.join(MODEL_SAVE_PATH, "{}_fold{}.h5".format(model_name, current_fold)))

            #Saving
            if (TRAINING_HISTORY):
                self.save_training_statistics(TRAINING_HISTORY_VAL_ACC, history.history['val_acc'], model_name, current_fold)
                self.save_training_statistics(TRAINING_HISTORY_ACC, history.history['acc'], model_name, current_fold)
                self.save_training_statistics(TRAINING_HISTORY_VAL_LOSS, history.history['val_loss'], model_name, current_fold)
                self.save_training_statistics(TRAINING_HISTORY_LOSS, history.history['loss'], model_name, current_fold)


    def random_sampler(self, feature_set, label_set, mode = 1):
        """Implementation of random over and undersampler"""
        #Modes: 1 - undersampling
        #       2 - oversampling
        #Return (feature_set, label_set)
        positive_indexes = [] #For oversampler
        negative_indexes = [] #For undersampler

        positive = 0
        #[!] WARNING: Random sampler for heatmaps not implemented yet!
        for i in range(0, len(label_set)):
            if (label_set[i] == 1.0):
                positive += 1
                positive_indexes.append(i)

        if positive == 0:
            return None


        if (mode == UNDERSAMPLING): #Random undersmapler implementation
            copies_fset = []
            copies_lset = []
            negative = 0
            while(negative != positive):
                rand_index = random.randint(0, label_set.shape[0] - 1)
                if (rand_index not in positive_indexes and rand_index not in negative_indexes):
                    negative_indexes.append(rand_index)
                    negative += 1
                print('\r**Undersampling {}/{}'.format(negative, positive), end = '')
                if (negative == positive):
                    print('...DONE')

            for index in positive_indexes:
                copies_fset.append(np.copy(feature_set[index]))
                copies_lset.append(np.copy(label_set[index]))
                   
            for index in negative_indexes:
                copies_fset.append(np.copy(feature_set[index]))
                copies_lset.append(np.copy(label_set[index]))

            copies_fset = np.array(copies_fset)
            if (len(feature_set.shape) > 2):
                copies_fset = np.reshape(copies_fset, (len(copies_fset), feature_set.shape[1], feature_set.shape[2]))
            else:
                copies_fset = np.reshape(copies_fset, (len(copies_fset), feature_set.shape[1]))

            copies_lset = np.array(copies_lset)

            return (copies_fset, copies_lset)

        elif (mode == OVERSAMPLING): #Random oversampler implementation
            negative = label_set.shape[0] - positive
            copies_fset = []
            copies_lset = []
            while (positive != negative):
                rand_index = random.randint(0, len(positive_indexes) - 1)

                copies_fset.append(np.copy(feature_set[positive_indexes[rand_index]]))
                copies_lset.append(np.copy(label_set[positive_indexes[rand_index]]))

                positive += 1

                print('\r**Oversampling {}/{}'.format(positive, negative), end = '')
                if (positive == negative): 
                    print('...DONE')


            copies_fset = np.array(copies_fset)
            copies_fset = np.reshape(copies_fset, (len(copies_fset), feature_set.shape[1], feature_set.shape[2]))

            copies_lset = np.array(copies_lset)

            feature_set = np.concatenate((feature_set, copies_fset), axis = 0)
            label_set = np.concatenate((label_set, copies_lset), axis = 0)

            return (feature_set, label_set)

    def load_extracted_features_old(self, name, fold):
        path = os.path.join(EXTRACTED_FEATURES_PATH, "{}_fold_{}.feat".format(name, fold))
        file = open(path, 'rb')
        data = pickle.load(file)
        file.close()

        return data

    def load_extracted_features(self, name, fold_to_use):
        dataset_filenames = os.listdir(EXTRACTED_FEATURES_PATH)
        dataset_filenames = [x for x in dataset_filenames if name in x]
        dataset_count = len(dataset_filenames)
        dataset_count_to_use = math.ceil(dataset_count / self._number_of_folds) 
        
        #Select indexes supposed to be in training and validation folds
        validation_scans = [x for x in range(fold_to_use * dataset_count_to_use, fold_to_use * dataset_count_to_use + dataset_count_to_use)]
        print(validation_scans)

        training_fset = None
        training_lset = np.array([])

        validation_fset = None
        validation_lset = np.array([])

        for i in range(0, dataset_count):
            feature_set, label_set = self.load_dataset_single(os.path.join(EXTRACTED_FEATURES_PATH, dataset_filenames[i]))
            print('\r**Processing dataset: {}/{}'.format(i + 1, dataset_count), end = '')
            if (i not in validation_scans):
                if (training_fset is None): 
                    training_fset = feature_set
                else:
                    training_fset = np.concatenate((training_fset, feature_set), 0)
                training_lset = np.concatenate((training_lset, label_set))
            else:
                if (validation_fset is None):
                    validation_fset = feature_set
                else:
                    validation_fset = np.concatenate((validation_fset, feature_set), 0)
                validation_lset = np.concatenate((validation_lset, label_set))
        print('')

        return ((training_fset, training_lset), (validation_fset, validation_lset))
        
        
    def load_dataset_single(self, filename):
        #Format - (np.array(f_set), np_array(l_set))
        file = open(filename, 'rb')
        data = pickle.load(file)
        file.close()
        return data
        
    def load_dataset(self, fold_to_use, dataset_folder, data_aug_options = NONE):
        #DATA AUGUMENTATION
        #0 - do nothing
        #1 - undersampling
        #2 - oversampling

        training_fset = None
        training_lset = np.array([])

        validation_fset = None
        validation_lset = np.array([])

        print("**Loading validation Feature and Label set")
        #How many sets will be in a fold
        dataset_filenames = os.listdir(dataset_folder)
        dataset_count = len(os.listdir(dataset_folder))
        dataset_count_to_use = math.ceil(dataset_count / self._number_of_folds) 
        
        #Select indexes supposed to be in training and validation folds
        validation_scans = [x for x in range(fold_to_use * dataset_count_to_use, fold_to_use * dataset_count_to_use + dataset_count_to_use)]
        print(validation_scans)

        #Iterate over all sets and add them to corresponding sets
        for i in range(0, dataset_count):
            feature_set, label_set = self.load_dataset_single(os.path.join(dataset_folder, dataset_filenames[i]))
            print('\r**Processing dataset: {}/{}'.format(i + 1, dataset_count), end = '')
            if (i not in validation_scans):
                if (training_fset is None): 
                    training_fset = feature_set
                else:
                    training_fset = np.concatenate((training_fset, feature_set), 0)
                training_lset = np.concatenate((training_lset, label_set))
            else:
                if (validation_fset is None):
                    validation_fset = feature_set
                else:
                    validation_fset = np.concatenate((validation_fset, feature_set), 0)
                validation_lset = np.concatenate((validation_lset, label_set))
        print('')


        #Data augumentation - Random sampling
        if (data_aug_options != 0):
            temp_training = None
            temp_validation = None

            if (data_aug_options == UNDERSAMPLING):
                temp_training = self.random_sampler(training_fset, training_lset, UNDERSAMPLING)
                temp_validation = self.random_sampler(validation_fset, validation_lset, UNDERSAMPLING)
            elif (data_aug_options == OVERSAMPLING):
                temp_training = self.random_sampler(training_fset, training_lset, OVERSAMPLING)
                temp_validation = self.random_sampler(validation_fset, validation_lset, OVERSAMPLING)
            
            training_fset = temp_training[0]
            training_lset = temp_training[1]

            validation_fset = temp_validation[0]
            validation_lset = temp_validation[1]

        #Data augumentation - Random image rotation
        random_rotation = True
        if (random_rotation):
            #Iterates over all images in feature sets and rotates them by random degree
            print_counter = 0
            print("**Rotating training feature set.")
            for train_img in training_fset:
                print("\r**Rotating image: {}/{}".format(print_counter + 1, training_fset.shape[0]), end = '')
                if (print_counter + 1 == training_fset.shape[0]):
                    print("...DONE")
                random_angle = random.randint(0, 360)
                train_img = ndimage.rotate(train_img, random_angle, order = 0)
                print_counter += 1

            print_counter = 0
            print("**Rotating validation feature set.")
            for validate_img in validation_fset:
                print("\r**Rotating image: {}/{}".format(print_counter + 1, validation_fset.shape[0]), end = '')
                if (print_counter + 1 == validation_fset.shape[0]):
                    print("...DONE")
                random_angle = random.randint(0, 360)
                validate_img = ndimage.rotate(validate_img, random_angle, order = 0)
                print_counter += 1

        #Reshaping for CNN
        print("**Reshaping validation sets")
        validation_fset = validation_fset.reshape(len(validation_fset), validation_fset.shape[1], validation_fset.shape[2], 1)
        print('**Reshaping training sets')
        training_fset = training_fset.reshape(len(training_fset), training_fset.shape[1], training_fset.shape[2], 1)
        print("**Shapes: Training {} {} \n          Validation: {} {}".format(training_fset.shape, training_lset.shape, validation_fset.shape, validation_lset.shape))

        return ((training_fset, training_lset), (validation_fset, validation_lset))


    def save_training_statistics(self, file_path, data_to_save, model_name, model_fold):
        save_file = open(file_path, 'a+')
        save_file.write(model_name + ' fold_{}'.format(model_fold))
        for value in data_to_save:
            save_file.write(';{}'.format(value))
        save_file.write('\n')
        save_file.close()
    