import os
import gc
import math
import errno
import torch
import pandas
import yaml
from tqdm import tqdm
from zipfile import ZipFile
from itertools import chain

# Blank Slate labels
blank_slate_names = [ 
        'person', 'hand', 'face', 'held toy', 'tabu', 'cube', 'box', 'pyramid', 'ball',
        'wooden toy', 'real food', 'soft toy', 'picture book', 'tablet', 'phone',
]

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
dataset_zip = os.path.join(os.path.dirname(__file__), "blank_slate.zip")
images_dir = os.path.join(os.path.dirname(__file__), "images")
labels = [label for label in os.listdir(images_dir) if (not label.startswith(".")) and os.path.isdir(os.path.join(images_dir, label))]
label_dirs = [[os.path.join(images_dir, label, f) for f in os.listdir(os.path.join(images_dir, label)) if (not f.startswith(".")) and os.path.isfile(os.path.join(images_dir, label, f)) ] for label in labels]

def autolabel():
    print("Labelling...")
    for label in tqdm(labels, unit="label"):
        if True:
            files = [f for f in os.listdir(os.path.join(images_dir, label)) if os.path.isfile(os.path.join(images_dir, label, f))]
            txts = set([txt.replace(".txt", "") for txt in files if txt.endswith(".txt")])
            unlabelled = [os.path.join(images_dir, label, f) for f in files if (f.endswith(".JPG") or f.endswith(".jpg")) and f.replace(".JPG", "").replace(".jpg", "") not in txts ]
            auto = {}
            if os.path.isfile(os.path.join(images_dir, label, "auto.yaml")):
                with open(os.path.join(images_dir, label, "auto.yaml")) as f:
                    auto = yaml.safe_load(f)

            for needed in unlabelled:
                # Inference
                results = model([needed])
                with open(needed.replace(".JPG",".txt").replace(".jpg", ".txt"), "w") as file:
                    file.writelines(list(chain.from_iterable([
                        translate_to_lines(auto, x, y, w, h, results.names[int(class_idx)], confidence)
                          for x, y, w, h, confidence, class_idx in results.xywhn[0]
                        ])))
        gc.collect(generation=2)
    print("done.")

def translate_to_lines(auto, x, y, w, h, external_label, confidence):
    def make_line(bs_label):
        try:
            idx = int(blank_slate_names.index(bs_label))
        except ValueError:
            idx = ''
        if confidence < 0.6:
            idx = ''
        return "{},{},{},{},{},{},{},{}\n".format(idx, x, y, w, h, external_label, bs_label, confidence)

    if external_label not in auto:
        return [make_line('')]

    result = auto[external_label]

    if type(result) is list:
        return [make_line(r) for r in result]
        # return [make_line(r) for r in result if r in blank_slate_names]

    return [make_line(result)]
    # return result in blank_slate_names and [make_line(result)] or []

def isindex(idx):
    if math.isnan(idx):
        return False
    if idx == "":
        return False
    if type(idx) is int:
        return True
    if type(idx) is float:
        return True
    return False

def prep_dataset_zip():
    try:
        os.makedirs(os.path.join(dataset_dir, "images", "all"))
    except OSError as exc: 
        pass
    try:
        os.makedirs(os.path.join(dataset_dir, "labels", "all"))
    except OSError as exc: 
        pass

    with ZipFile(dataset_zip, mode="w") as zip:
        for label in ["apple"]: # labels:
            label_path = os.path.join(images_dir, label)
            files = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]
            txts = set([txt.replace(".txt", "") for txt in files if txt.endswith(".txt")])
            labelled = [os.path.join(label_path, f) for f in files if (f.endswith(".JPG") or f.endswith(".jpg")) and f.replace(".JPG", "").replace(".jpg", "") in txts ]
            auto = {}
            if os.path.isfile(os.path.join(images_dir, label, "auto.yaml")):
                with open(os.path.join(images_dir, label, "auto.yaml")) as f:
                    auto = yaml.safe_load(f)

            for lf in labelled:
                arc_path = os.path.join("images", label + "_" + lf.replace(label_path + "/", ""))
                zip.write(lf, arcname=arc_path)

                txt = lf.replace(".JPG", ".txt").replace(".jpg", ".txt")
                csv = pandas.read_csv(txt, header=None, names=["idx", "x", "y", "w", "h", "external_label", "bs_label", "confidence"])
                names = [
                    "{},{},{},{},{}\n".format(row.get("idx"), row.get("x"), row.get("y"), row.get("w"), row.get("h"))
                      for idx, row in csv.iterrows() if isindex(row.get("idx"))]
                # if len(names) > 0:
                #     print("Found {label} picture({pic}) with {l} objects({names}) to use.".format(label=label, pic=arc_path, l=len(names), names=names))                

                data = "".join(names)
                txt_arc_path = os.path.join("images", label + "_" + txt.replace(label_path + "/", ""))
                zip.writestr(txt_arc_path, str(data))

autolabel()
prep_dataset_zip()

import code
code.InteractiveConsole(locals=globals()).interact()