from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import numpy as np
import pickle
import sys


model = VisionEncoderDecoderModel.from_pretrained("./fine_tuned_model")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_paths):
    images = []
    i_image = Image.open(image_paths)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    print (preds)
    return preds

base = "./out_img2img/rotate/rotatedtangram_"

out=[[[predict_step(base+str(j)+"_"+str(45*i)+".png") for i in range(0,8)] for j in range(1,7) ]]
np.save("./caption/cap_"+str(sys.argv[1])+'_'+str(sys.argv[2])+".npy", out)
print (out)
