"""
Core taxonomy module for content classification.
"""
from typing import Dict, List, NamedTuple, Optional
from dataclasses import dataclass
import csv
import logging
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from gandalf.config.settings import get_config

# Configure logging
logger = logging.getLogger(__name__)

config = get_config()

@dataclass
class CategoryMatch:
    """Represents a matched category with similarity scores."""
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
    """Represents a single taxonomy entry."""
    id: str
    parent: str
    name: str
    tier1: str
    tier2: str
    tier3: str
    tier4: str

class TaxonomyMatcher:
    """Handles content matching against taxonomy categories."""
    
    def __init__(self, taxonomy_file: str = None, model_name: str = None):
        """
        Initialize the TaxonomyMatcher.
        
        Args:
            taxonomy_file: Path to the taxonomy TSV file
            model_name: Name of the sentence transformer model to use
        """
        self.taxonomy_file = taxonomy_file or config.TAXONOMY_FILE
        self.model_name = model_name or config.MODEL_NAME
        self.taxonomy_data = self._load_taxonomy_data()
        
        # Set device
        self.device = self._get_device()
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        try:
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            self.category_embeddings = {}
            self._initialize_embeddings()
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def _get_device(self) -> str:
        """Determine the best available device for computation."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_taxonomy_data(self) -> Dict[str, TaxonomyEntry]:
        """Load taxonomy data from the TSV file."""
        taxonomy_data = {}
        try:
            file_path = Path(self.taxonomy_file)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Taxonomy file not found: {self.taxonomy_file}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                next(reader)  # Skip header row
                next(reader)  # Skip subheader row
                for row in reader:
                    if len(row) >= 7:
                        taxonomy_data[row[0]] = TaxonomyEntry(
                            id=row[0],
                            parent=row[1],
                            name=row[2],
                            tier1=row[3],
                            tier2=row[4] if row[4] else "",
                            tier3=row[5] if row[5] else "",
                            tier4=row[6] if row[6] else ""
                        )
            logger.info(f"Loaded {len(taxonomy_data)} taxonomy entries")
            return taxonomy_data
        except Exception as e:
            logger.error(f"Error loading taxonomy data: {e}")
            raise

    def _initialize_embeddings(self) -> None:
        """Pre-compute embeddings for all category names and tiers."""
        try:
            texts = []
            entries = []
            
            for entry in self.taxonomy_data.values():
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
                embeddings = embeddings.cpu()
            
            # Store embeddings with their corresponding entries
            for (entry, field), embedding in zip(entries, embeddings):
                key = (entry.id, field)
                self.category_embeddings[key] = embedding.numpy()
            
            logger.info(f"Initialized {len(self.category_embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise

    def compute_similarity(self, text_embedding: torch.Tensor, entry_id: str, field: str) -> float:
        """
        Compute similarity score between text and a specific field of an entry.
        
        Args:
            text_embedding: Embedding of the input text
            entry_id: ID of the taxonomy entry
            field: Field to compare against (name, tier1, etc.)
            
        Returns:
            Similarity score as a percentage
        """
        try:
            key = (entry_id, field)
            if key in self.category_embeddings:
                if self.device != "cpu":
                    text_embedding = text_embedding.cpu()
                sim = float(np.dot(text_embedding.numpy(), self.category_embeddings[key]))
                return max(0, min(100, sim * 100))  # Clamp between 0 and 100
            return 0.0
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def find_matching_categories(self, text: str, threshold: float = None, max_matches: int = None) -> List[CategoryMatch]:
        """
        Find matching categories using semantic similarity.
        
        Args:
            text: Input text to match
            threshold: Minimum similarity score to consider a match
            max_matches: Maximum number of matches to return
            
        Returns:
            List of CategoryMatch objects sorted by similarity
        """
        try:
            # Use config values if not provided
            threshold = threshold if threshold is not None else config.SIMILARITY_THRESHOLD
            max_matches = max_matches if max_matches is not None else config.MAX_MATCHES
            
            # Get embedding for the input text
            text_embedding = self.model.encode(text, convert_to_tensor=True)
            
            matches = []
            for entry in self.taxonomy_data.values():
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
        except Exception as e:
            logger.error(f"Error finding matching categories: {e}")
            return []

# Initialize global matcher instance
matcher = None

def initialize_matcher(taxonomy_file: str = None) -> None:
    """Initialize the global matcher instance."""
    global matcher
    try:
        matcher = TaxonomyMatcher(taxonomy_file=taxonomy_file)
        logger.info("Global matcher initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize global matcher: {e}")
        raise

def find_matching_categories(text: str) -> List[CategoryMatch]:
    """Wrapper function to use the global matcher instance."""
    global matcher
    if matcher is None:
        initialize_matcher()
    return matcher.find_matching_categories(text) 