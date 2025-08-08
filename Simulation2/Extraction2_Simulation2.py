import tensorflow as tf  # TensorFlowをインポート
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
import keras.backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

base ="/home/acml/.cache/huggingface/datasets/downloads/extracted/b91722fb368c1bfb2778b08b5c6440efc47e408f05a54a7a8abe404a585e3348/sketch/"
folders = os.listdir(base)


def load_image(path,angle):
    img = Image.open(path)
    img = img.convert('RGB')
    img = img.rotate(angle)
    # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
    img = img.resize((64,64))
    # 画像データをnumpy配列の形式に変更
    img = np.asarray(img)
    img = img / 255.0
    return img

def output_lab (id):
    labels=json.load(open("/home/acml/.keras/models/imagenet_class_index.json"))
    lst=list(labels.values())
    for j in lst:
        if j[0]==id:
            return j[1]
        
def rmse_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))        
# モデルを読み込む
model_path = "./cnn2sender.h5"
model = tf.keras.models.load_model(model_path, custom_objects={'rmse_loss': rmse_loss})
        
def top_K(model,testpic,K,angle):
    img = load_image(testpic,angle)
    prd = model.predict(np.array([img]))
    # ソートはされていない上位k件のインデックス
    unsorted_max_indices = np.argpartition(-prd[0], K)[:K]
    # 上位k件の値
    y = prd[0][unsorted_max_indices]
    # 大きい順にソートし、インデックスを取得
    indices = np.argsort(-y)
    # 類似度上位k件のインデックス
    max_k_indices = unsorted_max_indices[indices]
    return [output_lab(folders[i]) for i in max_k_indices]

angles=[0,135,180,225,270,315,45,90]
out=[[top_K (model,'./out_img2img/rotate/rotatedtangram_'+str(j)+'_'+str(angles[i])+'.png',1,0) for i in range(0,8)] for j in range(1,7)] 
np.save("./prediction/top5_"+str(sys.argv[1])+"_"+str(sys.argv[2])+".npy", out)