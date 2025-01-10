import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Union
import logging

class PatientRecordEmbedder:
    """
    A class to generate embeddings for patient records using PubMedBERT.
    """
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
        """
        Initialize the embedder with PubMedBERT model and tokenizer.
        
        Args:
            model_name (str): The name of the PubMedBERT model to use
        """
        logging.info(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on the token embeddings.
        
        Args:
            model_output: Output from the BERT model
            attention_mask: Attention mask from tokenizer
            
        Returns:
            torch.Tensor: Mean pooled embeddings
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def generate_embeddings(self, texts: Union[str, List[str]], batch_size: int = 8) -> np.ndarray:
        """
        Generate embeddings for one or more patient records.
        
        Args:
            texts: Single text string or list of text strings
            batch_size: Number of texts to process at once
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize the batch
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                batch_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                embeddings.append(batch_embeddings.cpu().numpy())
                
        return np.vstack(embeddings)

# Example usage
def main():
    # Sample patient records
    patient_records = [
        "Patient presents with severe chest pain, shortness of breath, and diaphoresis. History of hypertension.",
        "Follow-up visit for diabetes management. Blood sugar levels stable with current medication regimen.",
        "New onset migraine headaches with visual aura. No previous history of migraines."
    ]
    
    # Initialize embedder
    embedder = PatientRecordEmbedder()
    
    # Generate embeddings
    embeddings = embedder.generate_embeddings(patient_records)
    
    # Print embedding shapes and sample similarities
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Calculate similarities between records
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings)
    
    print("\nCosine similarities between patient records:")
    for i in range(len(patient_records)):
        for j in range(i+1, len(patient_records)):
            print(f"Record {i+1} vs Record {j+1}: {similarities[i][j]:.4f}")

if __name__ == "__main__":
    main()
