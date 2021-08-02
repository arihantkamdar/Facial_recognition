# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 23:25:59 2021

@author: Lenovo
"""

from deepface import DeepFace
import streamlit as st
import PIL
import os

st.title("Facial Recognition sys")
directory = os.listdir("Dataset")
for i in directory:
    if os.path.isdir(i)==False:
        directory.remove(i)
st.write("The available Faces are",directory)
"Upload the Image to be recognized"
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib","Ensemble"]
metrics = ["cosine", "euclidean", "euclidean_l2"]
option = st.sidebar.selectbox('Select Model',models)
option2 = st.sidebar.selectbox('Select metrics',metrics)
temp_img = st.file_uploader("Give the file image")
if temp_img is not None:
    img = PIL.Image.open(temp_img).convert('RGB')
    img = img.resize((300,300))
    st.image(img)
    path = os.path.join(os.getcwd(),'img.jpg')
    with open(path,"wb") as f: 
      f.write(temp_img.getbuffer())  
#    try:
    df = DeepFace.find(path,'Dataset',distance_metric = option2,model_name = option,)
    if len(df.index) != 0:
        "Match Found"
        best_match = df['identity'][0]
        parent = os.path.dirname(best_match)
        pro_parent = os.path.dirname(parent) 
        classs = parent.replace(pro_parent , '')
        classs = classs.replace("/Dataset", ": ")
        st.write("### Class ", classs)
        "Best Matches :"
        df["Smilarity"] = 1/df[option + '_' +option2]
        st.dataframe(df.head())
        
    else:
        "No Match Found"
#"""    except:
#        st.write("## No face detected, Please insert a better picture")
#"""
