import clip
import os
import numpy as np
import torch
import json
import wikipedia
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from PIL import Image

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

#labels for images
# image_directory = 'C:/Users/smitr/Downloads/Image & Text Retrieval System/imgclass'
# labels = [folder for folder in os.listdir(image_directory) if os.path.isdir(os.path.join(image_directory, folder))]
# text_descriptions = [f"{label}" for label in labels]

#creating embeddings
def create_image_embedding(images):  # Ensure that image is kind of [ img1 ] 
    image_input = torch.tensor(np.stack(images)).cuda()
    with torch.no_grad():
        image_embeddings = model.encode_image(image_input).float() 
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    return image_embeddings

def create_text_embedding(texts):  # Ensure that text is kind of [ txt1 ] 
    texts=clip.tokenize(texts)
    with torch.no_grad():
        text_embeddings = model.encode_text(texts.cuda()).float()
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    
    return text_embeddings

#for image to image
def input_img_embedding(img):
    images = []
    images.append(preprocess(img.convert("RGB")))
    query_vector = create_image_embedding(images)
    query_vector = query_vector.cpu()
    return query_vector

#for text to image
def input_text_embedding(txt):
    text = []
    text.append(txt)
    query_text=clip.tokenize(text)
    with torch.no_grad():
        query_text_embedding = model.encode_text(query_text.cuda()).float()
        query_text_embedding /= query_text_embedding.norm(dim=-1, keepdim=True)
    query_text_embedding = query_text_embedding.cpu()
    return query_text_embedding

#for image to text
def fetch_wikipedia_description(label):
    try:
        summary = wikipedia.summary(label, sentences=2)  # Fetch the first few sentences
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple Wikipedia entries found. Please be more specific. Options: {e.options}"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for this label."

def get_google_search_results(query, num_results=5):
    ua = UserAgent()
    user_agent = ua.random

    base_url = "https://www.google.com/search"
    params = {
        "q": query,
        "num": num_results
    }
    headers = {
        "User-Agent": user_agent
    }
    response = requests.get(base_url, params=params, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        search_results = soup.find_all("div", class_="tF2Cxc")
        urls = [result.find("a")['href'] for result in search_results]
        return urls
    else:
        print("Failed to fetch search results.")
        return []

#necessary for csv to embeddings
def serialize_embedding(embedding):
    return json.dumps(embedding.tolist())  # Convert to a list and then to JSON string

def deserialize_embedding(serialized_embedding):
    return np.array(json.loads(serialized_embedding))  # Convert back to NumPy array