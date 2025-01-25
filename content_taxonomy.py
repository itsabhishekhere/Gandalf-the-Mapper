from typing import Dict, List, NamedTuple, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass
import torch
import csv

@dataclass
class CategoryMatch:
    id: str
    parent: str
    name: str
    tier1: str
    tier2: Optional[str] = None
    tier3: Optional[str] = None
    tier4: Optional[str] = None
    name_similarity: float = 0.0
    tier1_similarity: float = 0.0
    tier2_similarity: float = 0.0
    tier3_similarity: float = 0.0
    tier4_similarity: float = 0.0

    def get_path_with_similarities(self) -> str:
        """Returns the hierarchical path of the category with similarity scores."""
        parts = []
        parts.append(f"{self.id}")
        parts.append(f"{self.parent}")
        parts.append(f"{self.name} ({self.name_similarity:.1f}%)")
        parts.append(f"{self.tier1} ({self.tier1_similarity:.1f}%)")
        if self.tier2:
            parts.append(f"{self.tier2} ({self.tier2_similarity:.1f}%)")
        if self.tier3:
            parts.append(f"{self.tier3} ({self.tier3_similarity:.1f}%)")
        if self.tier4:
            parts.append(f"{self.tier4} ({self.tier4_similarity:.1f}%)")
        return " -> ".join(filter(None, parts))

    @property
    def max_similarity(self) -> float:
        """Returns the highest similarity score across all tiers."""
        return max(
            self.name_similarity,
            self.tier1_similarity,
            self.tier2_similarity if self.tier2 else 0,
            self.tier3_similarity if self.tier3 else 0,
            self.tier4_similarity if self.tier4 else 0
        )

class TaxonomyEntry(NamedTuple):
    id: str
    parent: str
    name: str
    tier1: str
    tier2: str
    tier3: str
    tier4: str

def load_taxonomy_data() -> Dict[str, TaxonomyEntry]:
    """Load taxonomy data from the TSV file."""
    taxonomy_data = {}
    try:
        with open('Content Taxonomy 3.1.tsv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # Skip header row
            next(reader)  # Skip subheader row
            for row in reader:
                if len(row) >= 7:  # Ensure row has all required fields
                    taxonomy_data[row[0]] = TaxonomyEntry(
                        id=row[0],
                        parent=row[1],
                        name=row[2],
                        tier1=row[3],
                        tier2=row[4] if row[4] else "",
                        tier3=row[5] if row[5] else "",
                        tier4=row[6] if row[6] else ""
                    )
    except Exception as e:
        print(f"Error loading taxonomy data: {e}")
        raise
    return taxonomy_data

# Load taxonomy data from file
TAXONOMY_DATA = load_taxonomy_data()

class TaxonomyMatcher:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        self.category_embeddings = {}
        self.initialize_embeddings()

    def initialize_embeddings(self):
        """Pre-compute embeddings for all category names and tiers."""
        texts = []
        entries = []
        
        for entry in TAXONOMY_DATA.values():
            # Add name embedding
            texts.append(entry.name)
            entries.append((entry, "name"))
            
            # Add tier embeddings if they exist
            if entry.tier1:
                texts.append(entry.tier1)
                entries.append((entry, "tier1"))
            if entry.tier2:
                texts.append(entry.tier2)
                entries.append((entry, "tier2"))
            if entry.tier3:
                texts.append(entry.tier3)
                entries.append((entry, "tier3"))
            if entry.tier4:
                texts.append(entry.tier4)
                entries.append((entry, "tier4"))
        
        # Compute embeddings in batch
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        if self.device != "cpu":
            embeddings = embeddings.cpu()  # Move to CPU for numpy operations
        
        # Store embeddings with their corresponding entries
        for (entry, field), embedding in zip(entries, embeddings):
            key = (entry.id, field)
            self.category_embeddings[key] = embedding.numpy()

    def compute_similarity(self, text_embedding, entry_id: str, field: str) -> float:
        """Compute similarity score between text and a specific field of an entry."""
        key = (entry_id, field)
        if key in self.category_embeddings:
            if self.device != "cpu":
                text_embedding = text_embedding.cpu()  # Move to CPU for numpy operations
            sim = float(np.dot(text_embedding.numpy(), self.category_embeddings[key]))
            return sim * 100  # Convert to percentage
        return 0.0

    def find_matching_categories(self, text: str, threshold: float = 30.0, max_matches: int = 5) -> List[CategoryMatch]:
        """Find matching categories using semantic similarity."""
        # Get embedding for the input text
        text_embedding = self.model.encode(text, convert_to_tensor=True)
        
        matches = []
        for entry in TAXONOMY_DATA.values():
            # Compute similarities for each field
            name_sim = self.compute_similarity(text_embedding, entry.id, "name")
            tier1_sim = self.compute_similarity(text_embedding, entry.id, "tier1")
            tier2_sim = self.compute_similarity(text_embedding, entry.id, "tier2") if entry.tier2 else 0.0
            tier3_sim = self.compute_similarity(text_embedding, entry.id, "tier3") if entry.tier3 else 0.0
            tier4_sim = self.compute_similarity(text_embedding, entry.id, "tier4") if entry.tier4 else 0.0
            
            # If any similarity is above threshold, create a match
            max_sim = max(name_sim, tier1_sim, tier2_sim, tier3_sim, tier4_sim)
            if max_sim > threshold:
                match = CategoryMatch(
                    id=entry.id,
                    parent=entry.parent,
                    name=entry.name,
                    tier1=entry.tier1,
                    tier2=entry.tier2 if entry.tier2 else None,
                    tier3=entry.tier3 if entry.tier3 else None,
                    tier4=entry.tier4 if entry.tier4 else None,
                    name_similarity=name_sim,
                    tier1_similarity=tier1_sim,
                    tier2_similarity=tier2_sim,
                    tier3_similarity=tier3_sim,
                    tier4_similarity=tier4_sim
                )
                matches.append(match)
        
        # Sort by maximum similarity score and return top matches
        matches.sort(key=lambda x: x.max_similarity, reverse=True)
        return matches[:max_matches]

# Initialize the matcher as a global instance
matcher = TaxonomyMatcher()

def find_matching_categories(text: str) -> List[CategoryMatch]:
    """Wrapper function to use the global matcher instance."""
    return matcher.find_matching_categories(text) 