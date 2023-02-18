import streamlit as st
import dataviz_heart
import home_heart
import classif_heart
#import utils
#import forecast_stream
import plotly.io as pio
from PIL import Image


pio.templates.default = "none"


img = 'heart.jpg'

## Put the logo image
st.sidebar.image(Image.open(img))

PAGES = {
 "Home": home_heart,
 "Data Visualization": dataviz_heart,
 "Prediction": classif_heart
}
st.sidebar.subheader("About")
st.sidebar.markdown("This application can be used "
            "to predict heart failure.")

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.write()

st.sidebar.markdown('''
    -------
    By Danielle Taneyo''')
