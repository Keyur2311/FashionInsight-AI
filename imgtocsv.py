import clip
import os
from PIL import Image
import pandas as pd
import model

model1, preprocess = clip.load("ViT-B/32")
model1.cuda().eval()

path=r'C:\Users\smitr\Downloads\IRSModified_compressed\images_compressed'
format_='jpg'
num_images=4200
images=[];names=[]
for i in range(1,num_images):
    image_path = os.path.join(path, f'img ({i}).{format_}')
    image = Image.open(image_path).convert("RGB")
    images.append(preprocess(image))
    names.append(f'img ({i}).{format_}')

embeddings = model.create_image_embedding(images)
embeddings = embeddings.cpu()  # Convert to NumPy array on the CPU 

data={
    'Name':names,
}

database = pd.DataFrame(data)

serialized_embeddings = [model.serialize_embedding(emb) for emb in embeddings]
database['Serialized_Embeddings'] = serialized_embeddings

path_to_csv = r'C:\Users\smitr\Downloads\IRSModified_compressed\data.csv'

database.to_csv(path_to_csv)

# loaded_embeddings = [model.deserialize_embedding(emb) for emb in database['Serialized_Embeddings']]
# to load embeddings from .csv