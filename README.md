# DeepFake Detection

## Project Overview
This project implements multiple deep learning architectures so as to detect fake images or ai generated images. The application is built with PyTorch and Streamlit, providing an interactive web interface for users to upload and analyze images.

REFERENCES:

1. [siamese net](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
2. [xceptionet fine-tuning](https://discuss.pytorch.org/t/xception-finetune/19195)

## Features
- Utilizes a fine-tuned VGG16 model for high-accuracy deepfake detection
- Interactive web interface for easy image upload and analysis
- Real-time prediction with probability scores for real and fake classifications
- Model architecture summary display

## Project Structure
'''Certainly! Here's a comprehensive README file for your deep fake image detection project:
markdownCopy# DeepFake Image Detection

## Project Overview
This project implements a deep learning model for detecting deepfake images using a pre-trained VGG16 network. The application is built with PyTorch and Streamlit, providing an interactive web interface for users to upload and analyze images.

## Features
- Utilizes a fine-tuned VGG16 model for high-accuracy deepfake detection
- Interactive web interface for easy image upload and analysis
- Real-time prediction with probability scores for real and fake classifications
- Model architecture summary display

## Project Structure
![Screenshot (591)](https://github.com/user-attachments/assets/c5425031-b072-4488-96d0-49d941d99023)

## Installation

1. Clone the repository:

   `git clone https://github.com/0xD4rky/reality-`

2. Install the required packages:

   `pip install -r requirements.txt`

3. Run the Streamlit app:

   `cd app`
   `streamlit run app.py`
