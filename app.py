import streamlit as st
from PIL import Image
import numpy as np
import result
import model
import os

def main():   

    st.title("Image & Text Retrieval System")

    # Sidebar
    search_option = st.sidebar.selectbox("Search option", ["Text query", "Image query"])
    
    if search_option == "Text query":
        query = st.text_input("Enter search query:")
        text_embedding = model.input_text_embedding(query)
        resulting_imgs = result.search_similar_vectors(result.database,text_embedding,10)
        if st.button("Search"):
            image_paths = []
            for result1 in resulting_imgs['Name']:
                path=r'C:\Users\smitr\Downloads\IRSModified_compressed\images_compressed'
                image_path = os.path.join(path, f'{result1}')
                image_paths.append(image_path)
            cols = st.columns(3) # Adjust the number of columns as desired

            for index, path in enumerate(image_paths):
                cols[index % 3].image(path) 
    
    elif search_option == "Image query":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

        if uploaded_image is not None:
            img = Image.open(uploaded_image)
            st.write("This is the Input Image")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                col1.image(img, width=300)
            with col3:
                button1 = st.button("Search similar")
            with col4:
                button2 = st.button("Fetch Information")    

            img_embedding = model.input_img_embedding(img)
            if button1:
                resulting_imgs = result.search_similar_vectors(result.database,img_embedding,10)
                image_paths = []
                for result1 in resulting_imgs['Name']:
                    path=r'C:\Users\smitr\Downloads\IRSModified_compressed\images_compressed'
                    image_path = os.path.join(path, f'{result1}')
                    image_paths.append(image_path)
                
                cols = st.columns(3) # Adjust the number of columns as desired

                for index, path in enumerate(image_paths):
                    cols[index % 3].image(path) 

            if button2:
                label = result.search_similar_vectors_text(img_embedding, 1)
                st.title(label)
                try:
                    info = model.fetch_wikipedia_description(label)
                    st.write(info)

                    urls = model.get_google_search_results(label, 5)
                
                    if urls:
                        st.write(f"Top URLs related to '{label}':")
                    for idx, url in enumerate(urls, start=1):
                        st.write(f"{idx}. {url}")
                        print(f"{idx}. {url}")
                    else:
                        pass

                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
