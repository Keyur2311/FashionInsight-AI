import clip
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import model
import os

model1, preprocess = clip.load("ViT-B/32")
model1.cuda().eval()
input_resolution = model1.visual.input_resolution
context_length = model1.context_length
vocab_size = model1.vocab_size


def search_similar_vectors(database, query_vector, top_k=5):

    # Calculate cosine similarity between query_vector and vectors in the database
    similarity_scores = cosine_similarity(loaded_embeddings, query_vector)

    # Create a DataFrame to store similarity scores along with their indices
    similarity_df = pd.DataFrame({'similarity': similarity_scores.flatten()})

    # Combine similarity scores with the original dataframe
    database_with_similarity = pd.concat([database, similarity_df], axis=1)

    # Sort the dataframe by similarity scores in descending order and retrieve top k
    top_k_similar = database_with_similarity.sort_values(by='similarity', ascending=False).head(top_k)

    return top_k_similar

def search_similar_vectors_text(query_vector, top_k=1):

    #labels for images
    # image_directory = 'C:/Users/smitr/Downloads/Image & Text Retrieval System/imgclass'
    # labels = [folder for folder in os.listdir(image_directory) if os.path.isdir(os.path.join(image_directory, folder))]
    # text_descriptions = [f"{label.title()}" for label in labels]

    df = pd.read_csv('images.csv')

    # Select the desired column
    column_name = 'label'

    # Get unique values as a list
    text_descriptions = df[column_name].unique().tolist()

    text_embeddings = model.create_text_embedding(text_descriptions)
    text_embeddings = text_embeddings.cpu()

    # Calculate cosine similarity between query_vector and vectors in the text embeddings
    similarity_scores = cosine_similarity(text_embeddings, query_vector)

    # Create a DataFrame to store similarity scores along with their indices
    similarity_df = pd.DataFrame({'similarity': similarity_scores.flatten()})
    
    text_descriptions = pd.DataFrame({'label': text_descriptions})
    
    # Combine similarity scores with the original dataframe
    label_with_similarity = pd.concat([text_descriptions, similarity_df], axis=1)

    # Sort the dataframe by similarity scores in descending order and retrieve top k
    top_k_similar = label_with_similarity.sort_values(by='similarity', ascending=False).head(top_k)

    #get useful string
    label = top_k_similar['label'].astype(str).iloc[0]

    return label


database = pd.read_csv('data.csv')

loaded_embeddings = [model.deserialize_embedding(emb) for emb in database['Serialized_Embeddings']]