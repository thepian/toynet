#! /usr/bin/env python3
import argparse
import os
from io import BytesIO
import pandas as pd
# from oauth2client.service_account import ServiceAccountCredentials
from google.cloud import storage # https://googleapis.dev/python/storage/latest/client.html  https://cloud.google.com/storage/docs/json_api/v1

def parser():
    parser = argparse.ArgumentParser(description="Toynet Upload")
    parser.add_argument("--path", type=str, default="../toynet",
                        help="path to toynet folder"
                        "")
    return parser.parse_args()

def load_csvs_and_decorate(images):
    for image in images:
        # map it with URL
        Nil

    # As a pd.DataFrame

    label_files = os.listdir(os.path.join(args.path, "labels"))
    csvs = [fn for fn in label_files if fn.endswith(".csv")]
    for csv in csvs:
        lines = pd.read_csv(os.path.join(args.path, "labels", csv), header = 0, delimiter = ",")  # header = 1

    return images # updated version

def collate_and_upload_images(labels):
    images = []

    for label in labels:
        label_path = os.path.join(args.path, "images", label)
        if (label != ".DS_Store") and os.path.isdir(label_path):
            files = [f for f in os.listdir(label_path) if (f != ".DS_Store") and os.path.isfile(os.path.join(label_path, f))]
            print("Uploading", label, "TAG", label_path, len(files), "files.")
            for file in files:
                images.append([
                    "UNASSIGNED", 
                    os.path.join("gs://src-media.ignorethegap.com", label, file),
                    label,
                    "","", #str(0),str(0),
                    "","", #str(0),str(0),
                    "","", #str(0),str(0),
                    "","", #str(0),str(0),
                    ])
                blob = bucket.blob(os.path.join(label, file))
                if not blob.exists():
                    blob.upload_from_filename(os.path.join(args.path, "images", label, file))

    return images

args = parser()

storage_client = storage.Client.from_service_account_json(os.path.join("./google-cloud-f4e569864d3e.json")) # , project="Ignore the Gap")
bucket = storage_client.get_bucket("src-media.ignorethegap.com") # https://googleapis.dev/python/storage/latest/buckets.html#google.cloud.storage.bucket.Bucket

labels = os.listdir(os.path.join(args.path, "images"))
classes = len(labels)
images = collate_and_upload_images(labels)
images = load_csvs_and_decorate(images)

# print(images)
with open("toynet-untagged-files-cloud.csv", mode="wt", encoding="utf-8") as f:
    f.write("[set,]image_path[,label,x1,y1,,,x2,y2,,]\n")
    f.writelines([",".join(line) + "\n" for line in images])

# print("Saving", "toynet-untagged-files-cloud.csv")
# filesCSV = bucket.blob("toynet-untagged-files-cloud.csv")
# filesCSV.upload_from_filename(os.path.join(args.path, "toynet-untagged-files-cloud.csv"))

# print("Done.")
