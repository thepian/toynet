#! /usr/local/bin/python
import os
import json
import cv2

try:
    os.makedirs("dataset")
except Exception:
    pass

for entry in os.scandir(path="annotations"):
    if entry.is_file() and entry.name.endswith("-asset.json"):
        with open(entry.path) as f:
            data = json.load(f)
            regions = data["regions"]
            if len(regions) > 0:
                tag = regions[0]["tags"][0] or "unknown"
                boundingBox = regions[0]["boundingBox"]
                height = int(boundingBox["height"])
                width = int(boundingBox["width"])
                left = int(boundingBox["left"])
                top = int(boundingBox["top"])

                # print(data["asset"]["path"].replace("file:", ""), left, top, width, height)
                img = cv2.imread(data["asset"]["path"].replace("file:", ""))
                crop_img = img[top:top+height, left:left+width]
                cropped_path = os.path.join("dataset", tag, data["asset"]["name"])
                try:
                    os.makedirs(os.path.join("dataset", tag))
                except Exception:
                    pass
                # print("writing", cropped_path, len(crop_img))
                cv2.imwrite(cropped_path, crop_img)


"""
import cv2
img = cv2.imread("lenna.png")
crop_img = img[y:y+h, x:x+w]
cv2.imshow("cropped", crop_img)
"""
