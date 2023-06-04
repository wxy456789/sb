import streamlit as st
import os
import fastai.vision.all
from fastai.vision.all import *

import pathlib
temp=pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath

path=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(path,'export.pkl')

learn_inf=load_learner(model_path)

pathlib.PosixPath=temp

uploaded_file=st.file_uploader("Choose an image...",type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img=PILImage.create(uploaded_file)
    st.image(img.to_thumb(500,500),caption='Your Image')
    pred,pred_idx,probs=learn_inf.predict(img)
  
    st.write(f"Prediction：{pred};Probability:{probs[pred_idx]:.04f}'")
   
