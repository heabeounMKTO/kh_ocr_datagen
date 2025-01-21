from kh_bbox_gen import KhmerTextClusterGenerator 
import json
import math
import os
import random
import uuid 
from tqdm import tqdm
from argparse import ArgumentParser
from common_utils import convert_to_rectangle_coordinates,export_label_txt, load_names, crop_output_image, gen_filename, separate_train_eval, export_label_txt_det

def xyxy2xywh(xyxy, w, h):
    x1, y1 = xyxy[0]
    x2, y2 = xyxy[1]
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    _w = (x2 - x1) / w
    _h = (y2 - y1) / h
    return x_center, y_center, _w, _h
    
def writeYOLOtoFile(yololabels, filename):
    # print("Writing annoations to File...")
    with open(filename, "w") as f:
        for annotations in yololabels:
            f.write(str(annotations[0]))
            f.write(" ")
            f.write(str(annotations[1]))
            f.write(" ")
            f.write(str(annotations[2]))
            f.write(" ")
            f.write(str(annotations[3]))
            f.write(" ")
            f.write(str(annotations[4]))
            f.write("\n")

## change name IMMEDIEATELY
def generate_bbox(result, save_path: str,save_fname: str):
    width = result["rendered_image"].width
    height = result["rendered_image"].height
    all_bboxes = []
    for cluster in result["cluster_info"]:
        x1, y1, x2, y2 = [int(x) for x in cluster["bbox"]]
        # x, y, w, h = xyxy2xywh(([x1, y1], [x2, y2]), width, height)
        # x, y, w, h = abs(x), abs(y), abs(w), abs(h)
        coord = convert_to_rectangle_coordinates([x1,y1,x2, y2])
        fuck = {"transcription": cluster["text"], "points":coord}
        # _fjson = json.dumps(fuck,  ensure_ascii=False)
        all_bboxes.append(fuck)
    img_pth = f"{save_path}/{save_fname}.png"
    result["rendered_image"].save(img_pth)
    return img_pth, json.dumps(all_bboxes, ensure_ascii=False)
    # writeYOLOtoFile(all_bboxes, txt_path)


def generate_one(input_text: str, output_dir: str,save_debug: bool = False):
    img_filename = gen_filename()
    output = os.path.join(output_dir, f"{img_filename}.png")
    return renderer.render_text(input_text, output, save_debug, augment=False)

if __name__ == "__main__":
    _ALL_FONTS = ["./Hanuman-Regular.ttf", "./fonts/Battambang-Regular.ttf", "./fonts/Bayon.ttf", "./fonts/Kantumruy-Regular.ttf", "./fonts/KHMERMEF2.ttf", "./fonts/KhmerMoul.ttf", "./fonts/KhmerMuol.ttf", "./fonts/KhmerOS.ttf"]
    names = load_names()
    parser = ArgumentParser()
    parser.add_argument("--count", default=10,type=int,help="amount to generate")
    parser.add_argument("--export", default="bbox", type=str, help="export folder")
    opts = parser.parse_args()
    label_pairs = [] # i am too lazy to rename to train pairs bro
    eval_pairs = [] 
    print("generating train set")
    for _ in tqdm(range(0, math.ceil(opts.count))):
        for _font in _ALL_FONTS:
            renderer = KhmerTextClusterGenerator(font_path=_font, font_size=18)
            if not os.path.exists(opts.export):
                os.makedirs(opts.export, exist_ok=True)
            length = random.randrange(5, 20)
            final_string = []
            for _a in range(0, length):
                random_idx = random.randrange(0, (len(names) - 1))
                final_string.append(names[random_idx])    
            final_str = "".join(final_string)
            result = generate_one(final_str, opts.export, False)
            annotation = generate_bbox(result, opts.export, gen_filename())
            label_pairs.append(annotation)

    print("generating eval set")
    # i should make the ratio a arg , maybe one day.
    for _ in tqdm(range(0, math.ceil(opts.count * 0.3))):
        for _font in _ALL_FONTS:
            renderer = KhmerTextClusterGenerator(font_path=_font, font_size=18)
            if not os.path.exists(opts.export):
                os.makedirs(opts.export, exist_ok=True)
            length = random.randrange(5, 20)
            final_string = []
            for _a in range(0, length):
                random_idx = random.randrange(0, (len(names) - 1))
                final_string.append(names[random_idx])    
            final_str = "".join(final_string)
            result = generate_one(final_str, opts.export, False)
            annotation = generate_bbox(result, opts.export, gen_filename())
            eval_pairs.append(annotation)
    _z = export_label_txt_det(label_pairs,"train")
    _a = export_label_txt_det(eval_pairs,"eval")

