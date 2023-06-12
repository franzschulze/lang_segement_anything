import streamlit as st
import tempfile

# import leafmap
from PIL import Image
from lang_sam import LangSAM
from lang_sam.utils import load_image
from lang_sam.utils import draw_image

import numpy as np
import matplotlib.pyplot as plt
model = LangSAM()
import os
import torch
import torchvision
import torchvision.transforms as T

def clip_image(image_path, masks, segment):
    image_pil = load_image(image_path)
    image = np.asarray(image_pil)
    image_t = torch.from_numpy(image).permute(2, 0, 1)

    mask_tensor = torch.where(masks[segment], torch.tensor(1), torch.tensor(0))
    image_1 = image_t[0].unsqueeze(0)
    image_2 = image_t[1].unsqueeze(0)
    image_3 = image_t[2].unsqueeze(0)

    masked_tensor1 = image_1 * mask_tensor
    masked_tensor2 = image_2 * mask_tensor
    masked_tensor3 = image_3 * mask_tensor

    combined_tensor = torch.cat((masked_tensor1, masked_tensor2, masked_tensor3))
    return combined_tensor

def predicter(box_threshold, text_threshold, image_path, text_prompt):

    image_pil = load_image(image_path)
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold, text_threshold)

    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    image_array = np.asarray(image_pil)
    predicted_img = draw_image(image_array, masks, boxes, labels)
    predicted_img = Image.fromarray(np.uint8(predicted_img))
    plt.imshow(predicted_img)

    with tempfile.NamedTemporaryFile() as tmp:
        plt.savefig(tmp.name + "img.jpg", dpi=300)

        #######
        image_name = "ff"
        #######

        with open(tmp.name + "img.jpg", "rb") as fp:
            btn = st.download_button(
                label = "Download Predicted Image",
                data = fp,
                file_name = image_name + str(box_threshold) + "_" + str(text_threshold) + "_" + text_prompt+".jpg",
                mime = "image/jpeg"
            )
            plt.axis('off')
            st.pyplot()
        


def predicter_mask(box_threshold, text_threshold, image_path, text_prompt):

    image_pil = load_image(image_path)
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold, text_threshold)

    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    image_array = np.asarray(image_pil)
    predicted_img = draw_image(image_array, masks, boxes, labels)
    predicted_img = Image.fromarray(np.uint8(predicted_img))
    plt.imshow(predicted_img)

    with tempfile.NamedTemporaryFile() as tmp:
        plt.savefig(tmp.name + "img.jpg", dpi=300)

        #######
        image_name = "ff"
        #######

        with open(tmp.name + "img.jpg", "rb") as fp:
            btn = st.download_button(
                label = "Download Predicted Image",
                data = fp,
                file_name = image_name + str(box_threshold) + "_" + str(text_threshold) + "_" + text_prompt+".jpg",
                mime = "image/jpeg"
            )
            plt.axis('off')
            st.pyplot()
        #################################
        dim = masks.shape[0]
        print("*************************+++++++++++++++++++++++++++++************************************")
        print(dim)
        for i in range((dim-1)):
            masked_image = clip_image(image_path, masks, i)
            print(masked_image)
            plt.imshow(masked_image.permute(1, 2, 0) )
            plt.savefig(tmp.name + "img" + text_prompt + str(i) + ".jpg", dpi=300)   
            plt.axis('off')

            with open(tmp.name + "img" + text_prompt + str(i) + ".jpg", "rb") as fp:
                st.download_button(
                    label="Download Cliped IMAGE",
                    data=fp,
                    file_name = image_name + str(box_threshold) + "_" + str(text_threshold) + "_" + text_prompt+"_clip" + str(i) + ".jpg",
                    mime="image/jpeg"
                    )
            st.pyplot()

        # masking
        print(masks.shape)

st.set_option('deprecation.showPyplotGlobalUse', False)

#############################################################################################################################################
def main():
    st.title("Text Promt Image Segmentation with Segment Anything")
    st.markdown("""---""")

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    print(uploaded_file)
    # Text input
    prompt = st.sidebar.text_input("Enter a prompt", "")

    box_threshold  = st.sidebar.slider('Select a box threshold', 0.0, 1.0, 0.2)
    text_threshold = st.sidebar.slider('Select a text threshold', 0.0, 1.0, 0.8)

    if prompt is not None and uploaded_file is not None:
        plot_download = st.button("Predict and Plot")
        ##################################
        with st.spinner("Predicting and Plotting..."):
            if plot_download:
                predicter(box_threshold, text_threshold, uploaded_file, prompt)

        plot_download_mask = st.button("Predict, Plot and get all masks")
        st.markdown("""---""")
        ##################################
        with st.spinner("Predicting and Plotting..."):
            if plot_download_mask:
                predicter_mask(box_threshold, text_threshold, uploaded_file, prompt)

    st.write("Sources: Luca Medeiros (2023): Language Segment-Anything, https://github.com/luca-medeiros/lang-segment-anything/tree/main")

if __name__ == "__main__":
    main()



