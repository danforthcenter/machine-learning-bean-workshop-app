# Import libraries
from plantcv import plantcv as pcv
import streamlit as st
from PIL import Image 
import pandas as pd
import numpy as np
import cv2
import io

# Title the app 
st.title("Bean/Seed Analyzer")

pcv.outputs.clear()

# User file uploader 
uploaded_files = st.file_uploader(
    "Upload image(s)", accept_multiple_files=True, type=["jpg", "png"]
)

# Analysis workflow for each image uploaded 
# Note: image name will be used for the sample_label parameter in analysis functions 
i = 0
for uploaded_file in uploaded_files:
    i += 1 
    img = Image.open(uploaded_file)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    #st.markdown("Image shape:" + str(np.shape(img)))
    st.image(uploaded_file)

    # Use the image filename to use for labeling data
    img_name = uploaded_file.name.split('.')[0]

    # Converts the input color image into the LAB colorspace and returns the B (blue-yellow) channel as a grayscale image
    gray = pcv.rgb2gray_lab(rgb_img=img, channel="b")
    
    # Threshold the grayscale image
    auto_mask = pcv.threshold.otsu(gray_img=gray)
    # Optional intermediate outputs plotted
    #st.image(auto_mask)
       
    # Remove color card from mask
    cc_mask = pcv.transform.mask_color_card(img)
    cc_mask = pcv.dilate(cc_mask, ksize=20, i=1)
    auto_mask = pcv.image_subtract(auto_mask, cc_mask)
    # Optional intermediate outputs plotted
    #st.image(auto_mask)
    
    # Remove small background noise
    fill = pcv.fill(bin_img=auto_mask, size=1000)
    
    # Flood fill "pepper" noise
    clean_mask = pcv.fill_holes(bin_img=fill)
    
    # Create labeled mask
    labeled_mask, num = pcv.create_labels(mask=clean_mask)
    
    # Extract seed shape and color traits
    pcv.params.text_size = 1.5
    pcv.params.text_thickness = 5
    shape_img = pcv.analyze.size(img=img, labeled_mask=labeled_mask, n_labels=num, label=img_name)
    # Optional, plot the annotated shape image 
    st.image(shape_img, channels="BGR")
    
    # Detect the color card in the image
    img = pcv.transform.auto_correct_color(rgb_img=img)
    
    # Extract color traits from each replicate
    
    color_img = pcv.analyze.color(rgb_img=img, labeled_mask=labeled_mask, n_labels=num, colorspaces="hsv",
                                  label=img_name)
    # Optional, plot the color trait histograms
    st.altair_chart(color_img)
    
    csv_filename = f"{img_name}_csv.csv"
    pcv.outputs.save_results(csv_filename, "csv")
    pcv.outputs.clear()
    # Read CSV back in to allow download 
    csv_data = pd.read_csv(csv_filename)
    # Download button to save output data 
    st.download_button(
    label="Download data as CSV", # The text on the button
    data=csv_data.to_csv().encode("utf-8"), # The data to be downloaded
    file_name=csv_filename, # Suggested file name
    mime='text/csv', # MIME type for CSV files
    # Use an icon (optional)
    icon=":material/download:", 
)



