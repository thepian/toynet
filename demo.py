import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
# dir = 'https://github.com/ultralytics/yolov5/raw/master/data/images/'
dir = 'https://github.com/thepian/toynet/raw/master/images/pear/'
imgs = [dir + f for f in ('IMG_1514.JPG', 'IMG_1515.JPG')]  # batched list of images

# Inference
results = model(imgs)

# Results
results.print()  
results.save()  # or .show()

# Data
print(results.xyxy[0]) 
