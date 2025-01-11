import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import faiss
import json
from PIL import Image

class Blip:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "Salesforce/blip2-opt-2.7b"
        self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_id, load_in_8bit=True, 
                                                                   device_map={"": 0}, torch_dtype=torch.float16)
        self.processor = Blip2Processor.from_pretrained(self.model_id)
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

        # Set pad token
        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token

        kbase_file = open("eateries_knowledge_base.json", "r")
        self.kbase_config = json.load(kbase_file)
        kbase_file.close()

    def create_knowledge_base(self):
        # Step 2: Generate Embeddings for the Knowledge Base
        knowledge_embeddings = []
        for entry in self.kbase_config:
            embedding = self.sentence_model.encode(entry["description"], convert_to_tensor=True)
            knowledge_embeddings.append(embedding)

        knowledge_embeddings = torch.stack(knowledge_embeddings).cpu().numpy()
        return knowledge_embeddings

    def store_embeddings(self, knowledge_embeddings):
        # Create a FAISS Index and Store Embeddings
        dimension = knowledge_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(knowledge_embeddings)

    # Step 3: Query Processing
    def process_query(self, image, query_text):
        # 3.1 Generate Image Caption using BLIP-2
        # image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        image_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(image_caption)

        # 3.2 Extract Text Features using Sentence Transformer
        combined_query_text = f"{query_text} {image_caption}"
        query_embedding = self.sentence_model.encode(combined_query_text, convert_to_tensor=True).cpu().numpy()

        # Reshape query_embedding to a 2D array with a single row
        query_embedding = query_embedding.reshape(1, -1)


        # 3.3 Perform Similarity Search
        distances, indices = self.index.search(query_embedding, k=2)
        top_results = [self.kbase_config[i] for i in indices[0]]
        print("TOP RESULTS: ",top_results)
        return image_caption, top_results
    
    def chunk_text(self, text, max_length):
        """
        Split text into chunks of tokens with length <= max_length.
        """
        tokens = self.gpt_tokenizer.encode(text, add_special_tokens=False)
        return [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]

    
    def get_final_result(self, image_caption, query_text, top_results):
        """
        Generate a response using GPT-2 for the given input prompt and top results.
        """
        # Create the RAG prompt
        rag_prompt = f"""
        Here are the top 2 similar places nearby:
        1. {top_results[0]['name']}: {top_results[0]['description']}
        Address: {top_results[0]['address']}
        Rating: {top_results[0]['metadata']['rating']}
        2. {top_results[1]['name']}: {top_results[1]['description']}
        Address: {top_results[1]['address']}
        Rating: {top_results[1]['metadata']['rating']}
        """

        max_length = 1024
        # Chunk the text to handle large inputs
        chunks = self.chunk_text(rag_prompt, max_length - 50)

        responses = []
        for chunk in chunks:
            input_chunk = torch.tensor([chunk]).to(self.gpt_model.device)
            attention_mask = (input_chunk != self.gpt_tokenizer.pad_token_id).long()

            # Generate response for each chunk
            output = self.gpt_model.generate(
                input_chunk,
                attention_mask=attention_mask,
                max_new_tokens=200,
                num_return_sequences=1,
                pad_token_id=self.gpt_tokenizer.pad_token_id,
                eos_token_id=self.gpt_tokenizer.eos_token_id,
            )
            decoded_output = self.gpt_tokenizer.decode(output[0], skip_special_tokens=True)
            responses.append(decoded_output)

        # Combine responses from all chunks
        final_response = " ".join(responses)
        return final_response



if __name__ == "__main__":
    blip = Blip()
