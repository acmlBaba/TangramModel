import requests
import io
import base64
from PIL import Image, PngImagePlugin
import numpy as np
import os
from PIL import Image
import requests
from pprint import pprint
import sys
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
seeds=np.load("../random_receivers.npy")
seed=seeds[int(sys.argv[1])-1]

# テキストからの画像生成

def output_img (pmt, name):
    
    payload = {
    #"init_images": png_payload["image"],
    "prompt": "a monochrome sketch of" + pmt,
    "steps": 100,
    "seed": int(seed)
    }

    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    r = response.json()

    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
 
        png_payload = {
            "image": "data:image/png;base64," + i
        }
        response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))
        image.save(name, pnginfo=pnginfo)
    
#npyはhugtestに応じて変える
inp=np.load("./caption/cap_"+str(sys.argv[1])+'_'+str(sys.argv[2])+".npy", allow_pickle=True)


for tan in range(1,7):
    for an in range(0,8):
        output_img(str(inp[0][tan-1][an][0]),"./tangram_image_img2img/"+str(tan)+"_"+str(an*45)+'_'+str(sys.argv[1])+'_'+str(sys.argv[2])+".jpg")
        print('txt2img_'+str(tan)+'_'+str(an*45)+'finish')
