import base64
import os
import requests
from PIL import Image
import numpy as np
import requests
import base64
from io import BytesIO
import sys
import json
import io
from PIL import Image, PngImagePlugin
from pprint import pprint
import random
url =  "http://10.70.175.144:7861"


sd_models = requests.get(f"{url}/sdapi/v1/sd-models").json()
sd_models = [i["title"] for i in sd_models]
with open("sd_model.txt", '+w', encoding='UTF-8') as f:
    f.write('\n'.join(sd_models))
    
model = "v1-5-pruned-emaonly.safetensors [6ce0161689]"
option_payload = {
    "sd_model_checkpoint": model,
    # "CLIP_stop_at_last_layers": 2
}
response = requests.post(url=f'{url}/sdapi/v1/options', json=option_payload)
seeds=np.load("../random_senders.npy")
seed=seeds[int(sys.argv[1])-1]

def tangram_img2img(image,caption,angle,ID):
    

# バイナリデータからテキスト変換
    with io.BytesIO() as img_bytes:
        image.save(img_bytes, format='PNG')
        img_bytes = base64.b64encode(img_bytes.getvalue()).decode()
    
    png_payload = {}
    png_payload["image"] = [img_bytes]

    payload = {
        "init_images": png_payload["image"],
        "prompt":caption,
        "steps": 100,
        "image_strength":0.35,
        "init_image_mode": "IMAGE_STRENGTH",
        "init_images": png_payload["image"],
        "cfg_scale": 7,
        "samples": 1,
        "seed": int(seed)
    }

    response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)

    r = response.json()

    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

        png_payload = {
            "image": "data:image/png;base64," + i
        }
        response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))
        image.save('./out_img2img/img2imgResult/img2img_pic_'+str(ID)+'_'+str(angle)+'_'+str(sys.argv[1])+'_'+str(sys.argv[2])+'.png', pnginfo=pnginfo)
        
cnnoutput_caption = np.load("./prediction/top5_"+str(sys.argv[1])+"_"+str(sys.argv[2])+".npy") #CNNにたんグラムを入れて出てきたラベル.shape=(1,6,8,1)

# 画像が保存されているフォルダのパス
img_folder_result = './out_img2img/img2imgResult/'

angles=[0,135,180,225,270,315,45,90]
for ID in range(1,7):
    print("tangrams"+str(ID))

    for angle_i,angle in enumerate(angles):
        image = Image.open('./out_img2img/rotate/rotatedtangram_'+str(ID)+'_'+str(angle)+'.png')
        caption = cnnoutput_caption[ID-1][angle_i][0]

        tangram_img2img(image,caption,angle,ID)

        print(str(ID)+'_'+str(angle)+"finish")
        print("caption: "+ caption)    
        
