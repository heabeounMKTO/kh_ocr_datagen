from kh_bbox_gen import KhmerTextClusterGenerator 
from pprint import pprint
import numpy as np
import os
import random
import uuid 
from tqdm import tqdm
from argparse import ArgumentParser
from common_utils import export_label_txt, load_names, crop_output_image, gen_filename, separate_train_eval



def generate_one(input_text: str, output_dir: str,save_debug: bool = False):
    img_filename = gen_filename()
    output = os.path.join(output_dir, f"{img_filename}.png")
    augmentation_params = {
        'noise_factor': 0.08,
        'max_blur': 0.125,
        'use_background': True,
        'rotation_range': (-0.125, 0.125),
        'emboss': 0.0,
        'invert': 0.1
    }
    return renderer.render_text(input_text, output, save_debug, augment=True, augmentation_params=augmentation_params)

if __name__ == "__main__":
    ALL_FONTS = []
    for file in os.listdir("fonts"):
        if file.endswith(".ttf"):
            ALL_FONTS.append(os.path.join("fonts",file))
    # _ALL_FONTS=ALL_FONTS[0:30]
    _ALL_FONTS = ["./Hanuman-Regular.ttf", "./fonts/Battambang-Regular.ttf", "./fonts/Bayon.ttf", "./fonts/Kantumruy-Regular.ttf", "./fonts/KHMERMEF2.ttf", "./fonts/KhmerMoul.ttf", "./fonts/KhmerMuol.ttf", "./fonts/KhmerOS.ttf"]
    names = load_names()
    parser = ArgumentParser()
    parser.add_argument("--count", default=10,type=int,help="amount to generate")
    parser.add_argument("--export", default="results", type=str, help="export folder")
    opts = parser.parse_args()


    all_data = []
    for _font in tqdm(_ALL_FONTS):
        renderer = KhmerTextClusterGenerator(font_path=_font, font_size=22)
        if not os.path.exists(opts.export):
            os.makedirs(opts.export, exist_ok=True)

        for i in range(0, opts.count):
            length = random.randrange(5, 15)
            final_string = []
            for _a in range(0, length):
                random_idx = random.randrange(0, (len(names) - 1))
                final_string.append(names[random_idx])    
            final_str = "".join(final_string)
            result = generate_one(final_str, opts.export ,False)
            pprint(result)
            crop_output_image(result["cluster_info"], 
                              result["rendered_image"],
                              opts.export,
                              gen_filename(), 
                              all_data)
    train_set , eval_set = separate_train_eval(all_data) 
    # print(len(train_set), len(eval_set))
    export_label_txt(train_set, "train")
    export_label_txt(train_set, "eval")
