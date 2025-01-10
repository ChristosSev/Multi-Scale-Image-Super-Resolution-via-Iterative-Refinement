import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List, Dict, Tuple
import seaborn as sns
from pathlib import Path
import json
import logging
import torch

class MedicalEmbeddingAnalyzer:
    """
    Utility class for analyzing and visualizing medical record embeddings.
    """
    
    def __init__(self, embeddings_path: str = None):
        """
        Initialize the analyzer with optional pre-computed embeddings.
        
        Args:
            embeddings_path: Path to saved embeddings (optional)
        """
        self.embeddings = None
        self.labels = None
        logging.basicConfig(level=logging.INFO)
        
        if embeddings_path:
            self.load_embeddings(embeddings_path)
    
    def load_embeddings(self, path: str) -> None:
        """
        Load embeddings and metadata from disk.
        
        Args:
            path: Path to the saved embeddings file
        """
        data = np.load(path, allow_pickle=True)
        self.embeddings = data['embeddings']
        self.labels = data['labels'] if 'labels' in data else None
        logging.info(f"Loaded embeddings with shape: {self.embeddings.shape}")
    
    def save_embeddings(self, path: str, embeddings: np.ndarray, labels: List[str] = None) -> None:
        """
        Save embeddings and metadata to disk.
        
        Args:
            path: Path to save the embeddings
            embeddings: The embedding vectors
            labels: Optional labels for the embeddings
        """
        save_dict = {'embeddings': embeddings}
        if labels is not None:
            save_dict['labels'] = labels
        np.savez(path, **save_dict)
        logging.info(f"Saved embeddings to {path}")
    
    def visualize_embeddings(self, 
                           embeddings: np.ndarray,
                           labels: List[str] = None,
                           perplexity: int = 30,
                           title: str = "Medical Record Embeddings Visualization") -> plt.Figure:
        """
        Create t-SNE visualization of embeddings.
        
        Args:
            embeddings: The embedding vectors to visualize
            labels: Optional labels for the points
            perplexity: t-SNE perplexity parameter
            title: Plot title
        
        Returns:
            matplotlib figure object
        """
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        
        if labels:
            for i, label in enumerate(labels):
                ax.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
        
        ax.set_title(title)
        ax.set_xlabel("t-SNE dimension 1")
        ax.set_ylabel("t-SNE dimension 2")
        
        return fig
    
    def cluster_analysis(self, 
                        embeddings: np.ndarray,
                        n_clusters: int = 5) -> Tuple[np.ndarray, KMeans]:
        """
        Perform clustering analysis on embeddings.
        
        Args:
            embeddings: The embedding vectors to cluster
            n_clusters: Number of clusters to create
            
        Returns:
            Tuple of cluster labels and KMeans model
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        return clusters, kmeans
    
    def similarity_matrix(self, 
                         embeddings: np.ndarray,
                         labels: List[str] = None) -> plt.Figure:
        """
        Create a similarity matrix visualization.
        
        Args:
            embeddings: The embedding vectors
            labels: Optional labels for the matrix
            
        Returns:
            matplotlib figure object
        """
        similarities = cosine_similarity(embeddings)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(similarities, 
                   annot=True, 
                   cmap='YlOrRd', 
                   xticklabels=labels if labels else False,
                   yticklabels=labels if labels else False,
                   ax=ax)
        ax.set_title("Record Similarity Matrix")
        
        return fig
    
    def find_similar_records(self, 
                           query_embedding: np.ndarray,
                           embeddings: np.ndarray,
                           labels: List[str],
                           top_k: int = 5) -> List[Dict]:
        """
        Find the most similar records to a query embedding.
        
        Args:
            query_embedding: The embedding to compare against
            embeddings: The embedding vectors to search
            labels: Labels for the embeddings
            top_k: Number of similar records to return
            
        Returns:
            List of dictionaries containing similar records and scores
        """
        similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'label': labels[idx],
                'similarity_score': float(similarities[idx]),
                'index': int(idx)
            })
        
        return results

def main():
    """Example usage of the MedicalEmbeddingAnalyzer"""
    # Create sample data
    sample_embeddings = np.random.rand(10, 768)  # Example dimensionality
    sample_labels = [f"Record_{i}" for i in range(10)]
    
    # Initialize analyzer
    analyzer = MedicalEmbeddingAnalyzer()
    
    # Save and load embeddings
    analyzer.save_embeddings('sample_embeddings.npz', sample_embeddings, sample_labels)
    analyzer.load_embeddings('sample_embeddings.npz')
    
    # Create visualizations
    tsne_fig = analyzer.visualize_embeddings(sample_embeddings, sample_labels)
    tsne_fig.savefig('embeddings_tsne.png')
    
    # Perform clustering
    clusters, kmeans = analyzer.cluster_analysis(sample_embeddings)
    print(f"Cluster assignments: {clusters}")
    
    # Create similarity matrix
    sim_fig = analyzer.similarity_matrix(sample_embeddings, sample_labels)
    sim_fig.savefig('similarity_matrix.png')
    
    # Find similar records
    query_embedding = sample_embeddings[0]
    similar_records = analyzer.find_similar_records(
        query_embedding, sample_embeddings, sample_labels
    )
    print("\nMost similar records:")
    for record in similar_records:
        print(f"Label: {record['label']}, Score: {record['similarity_score']:.3f}")

if __name__ == "__main__":
    main()
