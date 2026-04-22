import sentence_transformers
from PIL import Image
from sentence_transformers import SentenceTransformer

def verify_image_embedding(img_path):
    multi_modal = MultiModalSearch()
    image_embedding = multi_modal.embed_image(img_path)
    print(f"Embedding shape: {image_embedding.shape[0]} dimensions")



class MultiModalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, img_path):
        img = Image.open(img_path)
        embedding = self.model.encode([img])
        return embedding[0]
