from .search_utils import load_movies
from PIL import Image
from sentence_transformers import SentenceTransformer
from .semantic_search import cosine_similarity

def verify_image_embedding(img_path):
    multi_modal = MultiModalSearch()
    image_embedding = multi_modal.embed_image(img_path)
    print(f"Embedding shape: {image_embedding.shape[0]} dimensions")

def image_search_command(img_path):
    multi_modal = MultiModalSearch()
    results = multi_modal.search_with_image(img_path)
    for i, result in enumerate(results):
        print(f"""
        {i+1}. {result['title']} (similarity: {result['similarity']:.3f})
        {result['description']}
        """)



class MultiModalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.docs = load_movies()
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.docs]
        self.text_embeddings = self.model.encode(self.texts)


    def embed_image(self, img_path):
        img = Image.open(img_path)
        embedding = self.model.encode([img])
        return embedding[0]

    def search_with_image(self, img_path):
        image_embedding = self.embed_image(img_path)
        cos_sim_list = []
        for doc, text_embedding in zip(self.docs, self.text_embeddings):
            cos_sim_list.append({
                "id": doc['id'],
                "title": doc['title'],
                "description": doc['description'],
                "similarity": cosine_similarity(image_embedding, text_embedding)
            })
        cos_sim_list.sort(key=lambda x: x["similarity"], reverse=True)
        return cos_sim_list[:5]


