import os
from synthgen import *
from common import *
from functools import reduce
import re
from time import time
from data_provider import DateProvider
import cv2
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.cm as cm

# Define some configuration variables:
# number of images to use for generation (-1 to use all available):
NUM_IMG = 1
INSTANCE_PER_IMAGE = 1  # number of times to use the same image
# SECS_PER_IMG = 5  # max time per image in seconds


def get_datalist(data_path):
    f = open(data_path, 'r')
    data_list = []
    for line in f.readlines():
        paths = line.strip().split()
        data_list.append(tuple(paths))
    return data_list


def main(data_list, output):
    """
    Entry point.

    Args:
        viz: display generated images. If this flag is true, needs user input to continue with every loop iteration.
        output_masks: output masks of text, which was used during generation
    """
    renderer = RendererV3('data')
    for i, (img_path, seg_path, dep_path) in enumerate(data_list):
        try:
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            dep = pickle.load(open(dep_path, 'rb'))
            seg = pickle.load(open(seg_path, 'rb'))

            sz = dep.shape[:2][::-1]
            img = Image.fromarray(img)
            img = np.array(img.resize(sz, Image.ANTIALIAS))
            seg = np.array(Image.fromarray(seg).resize(
                sz, Image.NEAREST)).astype(np.float32)
            
            label, area = np.unique(seg.astype('int0'), return_counts=True)

            res = renderer.render_text(img, dep, seg, area, label,
                                       ninstance=INSTANCE_PER_IMAGE)

            img_save = img_path.split("\\")[-1]
            img_save = os.path.join(output, img_save)
            try:
                imageio.imwrite(img_save, res[0]['img'])
            except: pass

        except:
            traceback.print_exc()
            print(colorize(Color.GREEN, '>>>> CONTINUING....', bold=True))
            continue


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Genereate Synthetic Scene-Text Images')
    parser.add_argument("--data-list", type=str, dest='data_path', default='../background/data_list.txt',
                        help="absolute path to data list containing images, segmaps and depths")
    parser.add_argument("--output-mask", type=str, dest='output_mask', default='../synth_image',
                        help="absolute path to data save path")
    args = parser.parse_args()
    data_list = get_datalist(args.data_path)
    main(data_list, args.output_mask)
