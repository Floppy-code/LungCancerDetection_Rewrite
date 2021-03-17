import os
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

from NeuralNetManager import NeuralNetManager

def main():
    manager = NeuralNetManager()
    manager.train_model()


main()