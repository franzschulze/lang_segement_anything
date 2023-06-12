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

def clip_image(image_path, masks, segment):
    image_pil = Image.open(image_path)
    image = np.array(image_pil)
    image_t = np.transpose(image, (2, 0, 1))

    mask_array = np.where(masks[segment], 1, 0)

    image_1 = image_t[0].reshape(1, image_t.shape[1], image_t.shape[2])
    image_2 = image_t[1].reshape(1, image_t.shape[1], image_t.shape[2])
    image_3 = image_t[2].reshape(1, image_t.shape[1], image_t.shape[2])

    masked_array1 = image_1 * mask_array
    masked_array2 = image_2 * mask_array
    masked_array3 = image_3 * mask_array

    combined_array = np.concatenate((masked_array1, masked_array2, masked_array3))

    return combined_array

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
            st.download_button(
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
        for i in range((dim)):
            masked_image = clip_image(image_path, masks, i)
            print(masked_image)
  
            combined_array = np.transpose(masked_image, (1, 2, 0))  # Transpose dimensions to (height, width, channels)
            combined_array = np.clip(combined_array, 0, 255).astype(np.uint8)  # Clip values to 0-255 and convert to uint8
            plt.imshow(combined_array)
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



