
"""
Created on Fri Jan 19 19:23:00 2024

@author: yoge
"""

import ultralytics
import streamlit as st
import cv2
from PIL import Image,ImageEnhance
import numpy as np
import pandas as pd
import glob
import random
import os
import torch
from ultralytics import YOLO
from shutil import rmtree
from pathlib import Path
from io import BytesIO
from PIL import ImageOps
model_path =r"C:\Users\yoges\Downloads\besst.pt"
if torch.cuda.is_available():
    torch.device('cuda')
else:
    torch.device('cpu')
model = YOLO(model_path)
 
class_names = ['Biodegradable ', 'Non-Biodegradable']
all_track_ids = []
st.title ("waste detection")
uploaded_image= st.file_uploader("Upload an image(jpg.jpeg,png):",type = ["jpg", "jpeg","png"])
if st.button("DetecT waste"):
    if uploaded_image:
        image = Image.open(uploaded_image)
        image = ImageOps.exif_transpose(image)
        image = image.convert('RGB')
        image = image.resize((640,640))
        results = model.predict(source= image, save = True, conf = 0.4)
        object_count = sum(len(r) for r in results)
        detection_results_dir = "./runs/detect"
        subfolders = [f.path for f in os.scandir(detection_results_dir) if f.is_dir()]
        latest_subfolder = max(subfolders, key=os.path.getctime)
        
        result_image_filename = 'image0.jpg'
        result_image_path = os.path.join(latest_subfolder,result_image_filename)
        
        detected_image = Image.open(result_image_path)
        detected_image = np.array(detected_image)
        detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
        
        st.subheader("Image Results:")
        st.image(detected_image, caption="Annotated image", use_column_width=True)
        st.write(f"Biodegradable Non-Biodegradable: {object_count}")
        