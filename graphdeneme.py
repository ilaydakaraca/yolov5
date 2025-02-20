__author__ = "Mehmet Değirmenci"


import streamlit as st
from PIL import Image
from urllib.request import urlretrieve
from fastai.vision.widgets import *
from fastai.vision.all import *


st.set_page_config(
    page_title="# GRAPHICS",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)


url = ("http://dl.dropboxusercontent.com/s/9jmlqe43bxv3p7y/best.pt?raw=1")
filename = "best.pt"
urlretrieve(url,filename)




urll = ("http://dl.dropboxusercontent.com/s/ecl4tj6q2u8s4q3/fig-03_5.png?raw=1")
filenamee = "1.png"
urlretrieve(urll,filenamee)
st.image(filenamee)
st.write('#')




uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])



if uploaded_file is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=filename)
    model.conf = 0.50
    model.conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    
    model.results = model(img_array, size=512)
    model.results.save("yolov5/results")
    model.results.save()
    st.image("yolov5/results/image0.jpg")
    ######
   
        
        
  
        


        

        
  

 #######
       
 
    liste = []
    liste1 = []
    liste2 = []
    liste0 = []


    for i in model.results.xywh:
        for j in i:
            for k in j:
                liste.append(k)
            if k ==2:
                liste2.append(k)
            elif k == 1:
                liste1.append(k)
            elif k == 0:
                liste0.append(k)


    st.write("Number of ----- detected:")
    st.write("Inhale:",len(liste0))
    st.write("Exhale:",len(liste1))
    st.write(":",len(liste2))
    
