import uuid 
import json
import math
import os
import random

def crop_output_image(clusters, rendered_image, output_path: str, filename: str, all_annotations: list):
    for idx, cluster in enumerate(clusters):
        crop_img = rendered_image.crop(cluster["bbox"])
        save_path = f"{output_path}/{filename}_{idx}.png"
        crop_img.save(save_path)
        _text = cluster["text"]
        anno_str = f"{save_path}"
        anno_pair = {anno_str: _text}
        all_annotations.append(anno_pair)

def export_label_txt_det(label_dict_list, txt_type: str):
    """
    this shit is for PaddlePaddle text detector
    """
    if txt_type.lower() == "train":
        with open("det_gt_train.txt", "w", encoding='utf-8') as train_txt:
            for _a in label_dict_list:
                train_txt.write(f"{_a[0]}\t{_a[1]}\n")
    if txt_type.lower() == "eval":
        with open("det_gt_eval.txt", "w", encoding='utf-8') as train_txt:
            for _a in label_dict_list:
                train_txt.write(f"{_a[0]}\t{_a[1]}\n")

def convert_to_rectangle_coordinates(coords):
    x1, y1, x2, y2 = coords
    # Top-left, Top-right, Bottom-right, Bottom-left
    rectangle = [
        [x1, y1],  # Top-left
        [x2, y1],  # Top-right
        [x2, y2],  # Bottom-right
        [x1, y2]   # Bottom-left
    ]

    return rectangle

def export_label_txt(label_dict_list, txt_type: str):
    """
    takes a list of annotation pairs that looks like 
    `Image_path`\t`Label`\n
    """
    if txt_type.lower() == "train":
        with open("rec_gt_train.txt", "w", encoding='utf-8') as train_txt:
            for _a in label_dict_list:
                for key, value in _a.items():
                    train_txt.write(f"{key}\t{value}\n")
    elif txt_type.lower() == "eval":
        with open("rec_gt_eval.txt", "w", encoding='utf-8') as eval_txt:
            for _a in label_dict_list:
                for key, value in _a.items():
                    eval_txt.write(f"{key}\t{value}\n")
    else:
        print(f"invalid export type {txt_type}!")
        exit()

def separate_train_eval(all_anno: list,train_sep: float = 0.8):
    _all_anno = all_anno.copy() # copy so we dont fuck up the original shit homie
    _train_num = math.floor(len(_all_anno) * train_sep) 
    train_set = []
    _sel_count = _train_num
    for _ in range(_train_num):
        random_index = random.randint(0, (_sel_count - 1))
        _sel_count -= 1
        _sel  = _all_anno[random_index]
        train_set.append(_sel)
        # remove from the copied array and use remaining as eval, avoid copying twice
        _all_anno.pop(_all_anno.index(_sel)) 
    return train_set, _all_anno 

def load_names():
    with open("km_5m.txt", "r", encoding="utf-8") as readfile:
        lines = readfile.read()
    return lines.split()

def gen_filename():
    return uuid.uuid4().hex


