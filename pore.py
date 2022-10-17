__author__ = "Ilayd Karaca"


import streamlit as st
from PIL import Image
from urllib.request import urlretrieve
import requests



url = ("http://dl.dropboxusercontent.com/s/drg59pqp7b56sf8/best.pt?raw=1")
filename = "best.pt"
urlretrieve(url,filename)






uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg', 'tiff', 'tif'])



if uploaded_file is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=filename)
    model.conf = 0.75
    #model.conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    

    
    model.results = model(img_array, size=512)
    #model.results.save("yolov5/results")
    model.results.save()
    st.image("runs/detect/exp/image0.jpg")
    ######
