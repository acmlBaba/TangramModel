from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image 
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop # TensorFlow1系
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
from io import BytesIO
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

keras_param_r = "./cnn2receiver.h5"
model_r = load_model(keras_param_r)

keras_param_s = "./cnn2sender.h5"
model_s = load_model(keras_param_s)

imsize = (64, 64)

def softmax(vec):
    exponential = np.exp(vec)
    probabilities = exponential / np.sum(exponential)
    return probabilities

def load_image(path,angle):
    img = Image.open(path)
    img = img.convert('RGB')
    img = img.rotate(angle)
    # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
    img = img.resize(imsize)
    # 画像データをnumpy配列の形式に変更
    img = np.asarray(img)
    img = img / 255.0
    return img

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

#OpenCVとosをインポート
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pickle
IMG_SIZE = (64, 64)

base ="/home/acml/.cache/huggingface/datasets/downloads/extracted/b91722fb368c1bfb2778b08b5c6440efc47e408f05a54a7a8abe404a585e3348/sketch/"
folders = os.listdir(base)
angles=[0,135,180,225,270,315,45,90]
ou=[]

for j in range(1,7):
    ou2=[]
    for i in angles:
        tag = './out_img2img/rotate/rotatedtangram_'+str(j)+'_'+str(i)+'.png'
        print (tag+'_start')
        img=load_image(tag,0)
        tgv=model_s.predict(np.array([img]))
        sk="./tangram_image_img2img/"+str(j)+'_'+str(i)+'_'+str(sys.argv[1])+'_'+str(sys.argv[2])+".jpg"
        sk_img=load_image(sk,0)
        bs=model_r.predict(np.array([sk_img]))
        ds=["f",0,int(j),int(i),0,0,[tgv[0]],sk,[bs[0]]] 
        for k in range(1,7):
            for l in angles:
                #tag="./tangrams/"+str(k)+".jpg"
                t_img=load_image('./out_img2img/rotate/rotatedtangram_'
                                 +str(k)+'_'+str(l)+'.png',0)
                tg=model_r.predict(np.array([t_img]))
                sm=cos_sim(tg[0],bs[0])
                
                #print (sm)
                if ds[1] < sm:
                    ds=[tag,sm,int(j),int(i),int(k),
                        int(l),[tgv[0]],sk,[tg[0]]] 
                    #print(ds)            
        if ds[0]!="f":
            print(ds[0]+'_finish')
            #imgo=load_image(ds[0],ds[5])
        ou2.append(ds)
        #print(ds[0])
    ou.append(ou2)   

#print (ou.shape)
np.save("./output/out_"+str(sys.argv[1])+'_'+str(sys.argv[2])+".npy", ou)

#事例を貯めるか，リセットするか．以下は貯める場合
#if int(sys.argv[2])>1:
    #pas_s = np.load('./learn/input_tangram'+str(sys.argv[1])+'.npy')
    #prd_s = np.load('./learn/testdata'+str(sys.argv[1])+'.npy')
    #pas_r = np.load('./learn/input_tangram_r'+str(sys.argv[1])+'.npy')
    #prd_r = np.load('./learn/testdata_r'+str(sys.argv[1])+'.npy')
#else:
#    prd = []
#    pas =[]

#リセットする場合
prd_s = []
pas_s =[]
prd_r = []
pas_r =[]
image_s=[]
caption_s=[]

base = "./out_img2img/rotate/rotatedtangram_"
cnnoutput_caption = np.load("./caption/cap_"+str(sys.argv[1])+'_'+str(sys.argv[2])+".npy")
new_angles = [0, 45, 90, 135, 180, 225, 270, 315]
for i in range(0,6):
    for j in range(0,8):
        if ou[i][j][2]==ou[i][j][4]:
                print(str(i+1)+'-'+str(angles[j]))
                #img = load_image('./out_img2img/rotate/rotatedtangram_'+str(i+1)+'_0.png',angles[j])
                #img_path = './out_img2img/rotate/rotatedtangram_'+str(i+1)+'_'+str(angles[j])+'.png'
                #prd.append(model.predict(np.array([img]))) 
                image_s.append(base+ str(ou[i][j][2]) + "_" + str(ou[i][j][3])+".png")
                target_index=new_angles.index(ou[i][j][3])
                caption_s.append(cnnoutput_caption[0][i][target_index])
                pas_s.append(ou[i][j][0])
                prd_s.append(ou[i][j][6]) 
                pas_r.append(ou[i][j][7])
                prd_r.append(ou[i][j][8]) 
                #else:
                #pas_s.append(ou[i][j][0])
                #prd_s.append(softmax(np.ones(1000) - ou[i][j][6]))
                #pas_r.append(ou[i][j][7])
                #prd_r.append(softmax(np.ones(1000)  - ou[i][j][8])) 


prd_s = np.array(prd_s)
prd_r = np.array(prd_r)
new_array = np.squeeze(prd_s, axis=1)
# 1次元目を削除(n,1,m)になるため
np.save('./learn/testdata'+str(sys.argv[1])+'.npy', new_array)
np.save('./learn/input_tangram'+str(sys.argv[1])+'.npy',pas_s)
new_array = np.squeeze(prd_r, axis=1)
np.save('./learn/testdata_r'+str(sys.argv[1])+'.npy', new_array)
np.save('./learn/input_tangram_r'+str(sys.argv[1])+'.npy',pas_r)
np.save('./learn/sender_tangram'+str(sys.argv[1])+'.npy',image_s)
np.save('./learn/sender_caption'+str(sys.argv[1])+'.npy',caption_s)

