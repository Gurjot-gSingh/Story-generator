import streamlit as st
from PIL import Image
from transformers import pipeline
import numpy as np

# Set the app title
st.set_page_config(page_title="Image-to-Story Generator", layout="centered")



# Load pipelines only once
@st.cache_resource
def load_pipelines():
    img2caption = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text_to_story = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    return img2caption, text_to_story

# Generate image caption
def generate_image_caption(image, img2caption):
    return img2caption(image)[0]['generated_text']

# Generate story (limit to ~100 words)
def text2story(caption, text_to_story, word_limit=250):
    result = text_to_story(caption, max_new_tokens=150)[0]['generated_text']
    words = result.split()
    return ' '.join(words[:word_limit])


# Main logic
def main():

    # App title and description
    st.title("📖 Image to Story with Voice")
    st.markdown("Upload an image and get a short story based on it!")

    uploaded_file = st.file_uploader("Choose an image")
    
    if uploaded_file is not None:
        file_ext = uploaded_file.name.lower().split('.')[-1]
        if file_ext not in ["jpg", "jpeg", "png"]:
            st.error("🚫 Unsupported file type. Please upload a JPG, JPEG, or PNG image.")
            return

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img2caption, text_to_story = load_pipelines()

        with st.spinner("🔍 Generating caption..."):
            caption = generate_image_caption(image, img2caption)
        st.subheader("📝 Generated Caption")
        st.write(caption)

        with st.spinner("✍️ Creating story..."):
            story = text2story(caption, text_to_story)
        st.subheader("📖 Generated Story")
        st.write(story)

          

    else:
        st.info("👈 Please upload a JPG, JPEG, or PNG image to get started.")

# Entry point
if __name__ == "__main__":
    main()
