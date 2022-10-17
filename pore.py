__author__ = "Ilayd Karaca"


import streamlit as st
from PIL import Image
from urllib.request import urlretrieve
from fastai.vision.widgets import *
from fastai.vision.all import *





url = ("https://www.dropbox.com/s/drg59pqp7b56sf8/best%20%281%29.pt?dl=0")
filename = "best (1).pt"
urlretrieve(url,filename)






uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg', 'tiff', 'tif'])



if uploaded_file is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    model = torch.hub.load('ilaydakaraca/pore_detection', 'custom', path=filename)
    model.conf = 0.75
    #model.conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    
    model.results = model(img_array, size=512)
    model.results.save("pore_detection/results")
    model.results.save()
    st.image("pore_detection/results/image0.jpg")
    ######