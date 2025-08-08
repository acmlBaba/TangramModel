#ライブラリインポート
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image 
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from keras.utils import np_utils
import numpy as np
import sys
import json
import glob
import os
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pickle
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping

base ="/home/acml/.cache/huggingface/datasets/downloads/extracted/b91722fb368c1bfb2778b08b5c6440efc47e408f05a54a7a8abe404a585e3348/sketch/"

folders = os.listdir(base)

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


#前学習画像ファイルの読み込み
image_file = './learn/input_tangram'+str(sys.argv[1])+'.npy'
#ベクトルファイルの読み込み
vector_file = './learn/testdata'+str(sys.argv[1])+'.npy'
#前学習画像ファイルの読み込み
image_file_r = './learn/input_tangram_r'+str(sys.argv[1])+'.npy'
#ベクトルファイルの読み込み
vector_file_r = './learn/testdata_r'+str(sys.argv[1])+'.npy'


# 画像ファイルを読み込んでリサイズし、リストに追加
def resized_list (image_file):
    file_paths = np.load(image_file)
    image_list = []
    for file_path in file_paths:
        img = Image.open(file_path)
        img_resized = img.resize((64, 64))
        img_array = np.array(img_resized)
        image_list.append(img_array)
    return image_list
        
# リストをNumPy配列に変換
image_array = np.array(resized_list(image_file))
image_array_r = np.array(resized_list(image_file_r))

num_classes = len(folders)
image_size = 64

def load_data(image_array,vector_file):
    X = image_array
    Y = np.load(vector_file, allow_pickle=True)

    return X, Y


"""
モデルを学習する関数
"""

def train(X, Y,modelfile):
    model = load_model(modelfile)
      
    batch_size = len(X)
       # 早期終了コールバックを定義
    early_stopping = EarlyStopping(monitor='loss', 
                                   patience=10, restore_best_weights=True)    
        # 早期終了コールバックを含めてモデルをトレーニング
    model.fit(X, Y, epochs=5, batch_size=batch_size, callbacks=[early_stopping])  
    # 適切なエポック数、バッチサイズ、検証データの使用などを設定
    # HDF5ファイルにKerasのモデルを保存
    model.save(modelfile)    
    return model

"""
メイン関数
データの読み込みとモデルの学習を行います。
"""
def main():
    # データの読み込み

    X, Y = load_data(image_array,vector_file)
    X_r, Y_r = load_data(image_array_r,vector_file_r)
    
    # モデルの学習
    model = train(X,Y,'./cnn2sender.h5')
    #model_r = train(X_r,Y_r,'./cnn2receiver.h5')    
    
main()