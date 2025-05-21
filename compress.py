from ngram import Model
import os

def compress(model_name):
    Model(6, model_name).compress()

def uncompress():
    for model_name in os.listdir('model/'):
        Model(6, model_name).uncompress()

if __name__ == '__main__':
    compress('wikipedia-ca')