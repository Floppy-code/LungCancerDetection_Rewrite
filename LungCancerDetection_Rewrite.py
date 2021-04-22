import os
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

from NeuralNetManager import NeuralNetManager

def main():
    manager = NeuralNetManager()
    
    #Select one of three modes
    #manager.train_model()
    #manager.extract_features()
    manager.train_model_extracted_features()


main()