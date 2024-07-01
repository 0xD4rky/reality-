from os import path
from random import randint

import streamlit as st
from PIL import Image

# Set Streamlit page configuration
st.set_page_config(page_title="DeepFake Image",
                   page_icon=path.join('assets', 'icons', 'logo.png'))

# Load and apply custom CSS
with open(r'C:\Users\DELL\OneDrive\Documents\reality\assets\style.css', "r") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Create columns for layout
col1, col2, col3 = st.columns([3, 2, 3])

# Display the banner image in the middle column
with col2:
    st.image(r'C:\Users\DELL\OneDrive\Documents\reality\assets\icon\robot.png', use_column_width=True)

# Set the title and subtitle
st.title('Real Fake Face Image Classifier "DeepFakes"')
st.subheader('')
