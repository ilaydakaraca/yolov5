__author__ = "Mehmet Değirmenci"


import streamlit as st
from PIL import Image
from urllib.request import urlretrieve
import numpy as np
import torch
import requests




#url = ("http://dl.dropboxusercontent.com/s/puam7atr8213pks/colony.pt?raw=1")
url = ("http://dl.dropboxusercontent.com/s/drg59pqp7b56sf8/best.pt?raw=1")

filename = "best.pt"
urlretrieve(url,filename)



urll = ("http://dl.dropboxusercontent.com/s/ecl4tj6q2u8s4q3/fig-03_5.png?raw=1")
filenamee = "1.png"
urlretrieve(urll,filenamee)
st.image(filenamee)
st.write('# COLONY COUNTER V.1')




uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])



if uploaded_file is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=filename)
    model.conf = 0.40
    model.conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    
    model.results = model(img_array, size=512)
    #model.results.save("yolov5/results")
    model.results.save()
    st.image("runs/detect/exp/image0.jpg")
    ######
   
        
        
  
        


        

        
  

 #######
       
 
    liste = []
    liste1 = []
    liste2 = []
    liste0 = []
    liste3 = []
    liste4 = []
    liste5 = []
    liste6 = []


    for i in model.results.xywh:
        for j in i:
            for k in j:
                liste.append(k)
            if k ==2:
                liste2.append(k)
            elif k == 1:
                liste1.append(k)
            elif k == 3:
                liste3.append(k)
            elif k == 4:
                liste4.append(k)
            elif k == 5:
                liste5.append(k) 
            elif k == 6:
                liste6.append(k)
            elif k == 0:
                liste0.append(k)
            

    st.write("Number of colony detected:",len(liste0) +len(liste1) +len(liste2) +len(liste3) +len(liste4) +len(liste5) +len(liste6))
   
    
    
