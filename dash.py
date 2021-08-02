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
st.write("""
         > Best Backend for face detection is default opencv
         
         > Best Metrics is Cosine
         
         > Best Model in Facenet512""")



os.chdir("D:/AI-ML_Trainee/Facial_recog")
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
    df = DeepFace.find(path,'/Dataset/',distance_metric = option2,model_name = option,)
    if len(df.index) != 0:
        "Match Found"
        best_match = df['identity'][0]
        parent = os.path.dirname(best_match)
        pro_parent = os.path.dirname(parent) 
        classs = parent.replace(pro_parent , '')
        classs = classs.replace("\\", "")
        st.write("### Class :", classs)
        "Best Matches :"
        df["Smilarity"] = 1/df[option + '_' +option2]
        df.drop(option + '_' +option2,axis = 1)
        st.dataframe(df.head())
        "Select to open image from database"
        names = df.head()["identity"]
        name_options = st.selectbox("Choose image to open",names)
        if st.button("Open"):
            new_img = PIL.Image.open(name_options).convert('RGB')
            new_img = new_img.resize((300,300))
            st.image(new_img)
    else:
        "No Match Found"
#"""    except:
#        st.write("## No face detected, Please insert a better picture")
#"""
