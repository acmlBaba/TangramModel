from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import numpy as np
import pickle
import sys

model = VisionEncoderDecoderModel.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cpu")
model.to(device)
from torch.utils.data import Dataset

class ImageCaptionDataset(Dataset):
    def __init__(self, image_paths, captions, feature_extractor, tokenizer, max_length=16):
        self.image_paths = image_paths
        self.captions = captions
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]

        # 画像をロードし、特徴量を抽出
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        pixel_values = self.feature_extractor(images=[image], return_tensors="pt").pixel_values.squeeze()

        # キャプションをトークン化
        tokens = self.tokenizer(caption, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)

        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids.squeeze(),
            "attention_mask": tokens.attention_mask.squeeze()
        }
from torch.utils.data import DataLoader
image_np   = np.load('./learn/sender_tangram'+str(sys.argv[1])+'.npy')
caption_np =np.load('./learn/sender_caption'+str(sys.argv[1])+'.npy')

#image_paths = [base + str(j) + "_" + str(45 * i) + "_1_1.png" for i in range(0, 8) for j in range(1, 7)]
image_paths=list(image_np)
# 対応するキャプション作成
captions = [caption[0] for caption in caption_np]
#captions = [out[0][j][i] for i in range(0, 8) for j in range(0, 6)]

dataset = ImageCaptionDataset(image_paths, captions, feature_extractor, tokenizer)
dataloader = DataLoader(dataset, batch_size=len(image_paths), shuffle=True)

from transformers import AdamW

# AdamWを設定
optimizer = AdamW(model.parameters(), lr=5e-5)

from torch.nn import CrossEntropyLoss

model.train()  # モデルを学習モードに切り替える

epochs = 5
loss_fn = CrossEntropyLoss()

for epoch in range(epochs):
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=pixel_values, labels=input_ids, decoder_attention_mask=attention_mask)


        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

