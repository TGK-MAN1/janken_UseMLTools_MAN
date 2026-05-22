#YOLOで使うため画像の拡張子は.jpgのみとします
#元データ(.jpg)をimages, アノテーションファイル(.txt)をlabelsに格納している前提とします

import os
import random
import shutil

image_dir = "images"
label_dir = "labels"
train_ratio = 0.8

new_dirs_name = [
    os.path.join(image_dir, "train"),
    os.path.join(image_dir, "val"),
    os.path.join(label_dir, "train"),
    os.path.join(label_dir, "val")
]

for d in new_dirs_name:
    os.makedirs(d, exist_ok=True)

all_files = [
    os.path.splitext(f)[0]
    for f in os.listdir(image_dir)
    if f.lower().endswith(".jpg")
]

random.seed(42)
random.shuffle(all_files)

split_point = int(len(all_files) * train_ratio)

train_files = all_files[:split_point]
val_files = all_files[split_point:]

def move_files(file_list, subset):
    for base_name in file_list:
        img_src = os.path.join(image_dir, base_name + ".jpg")
        if os.path.exists(img_src):
            shutil.move(
                img_src, os.path.join(
                    image_dir, 
                    subset, 
                    base_name + ".jpg"
                )
            )

        txt_src = os.path.join(label_dir, base_name + ".txt")
        if os.path.exists(txt_src):
            shutil.move(
                txt_src, os.path.join(
                    label_dir, 
                    subset, 
                    base_name + ".txt"
                )
            )

print(f"全体のデータ数 : {len(all_files)}枚")
move_files(train_files, "train")
move_files(val_files, "val")
print(f"分割完了 -> Train : {len(train_files)}枚、Val : {len(val_files)}枚")