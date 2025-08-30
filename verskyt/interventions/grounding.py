"""
Feature grounding for Tversky Neural Networks.

Provides tools for grounding features and prototypes to semantic concepts,
enabling human-interpretable understanding of TNN internals.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from .manager import InterventionManager, FeatureInfo, PrototypeInfo


@dataclass
class ConceptGrounding:
    """Associates a TNN parameter with a semantic concept."""
    
    layer_name: str
    parameter_type: str  # 'feature' or 'prototype'
    parameter_index: int
    concept_name: str
    concept_description: str
    confidence: float  # Confidence in the grounding (0-1)
    
    # Supporting evidence
    activation_correlation: Optional[float] = None
    visual_similarity: Optional[float] = None
    semantic_coherence: Optional[float] = None
    
    # Metadata
    grounding_method: str = "manual"  # 'manual', 'activation_based', 'similarity_based'
    validation_samples: Optional[List[Any]] = None


@dataclass
class ConceptLibrary:
    """Library of semantic concepts for grounding."""
    
    concepts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def add_concept(
        self, 
        name: str, 
        description: str, 
        examples: Optional[List[Any]] = None,
        properties: Optional[Dict[str, Any]] = None
    ):
        """Add a concept to the library."""
        self.concepts[name] = {
            "description": description,
            "examples": examples or [],
            "properties": properties or {},
            "groundings": []  # Will track which parameters are grounded to this concept
        }
    
    def get_concept(self, name: str) -> Optional[Dict[str, Any]]:
        """Get concept information."""
        return self.concepts.get(name)
    
    def list_concepts(self) -> List[str]:
        """List all concept names."""
        return list(self.concepts.keys())


class FeatureGrounder:
    """
    Ground TNN features and prototypes to semantic concepts.
    
    Provides various methods for associating learned parameters with
    human-interpretable concepts.
    """
    
    def __init__(self, intervention_manager: InterventionManager):
        """
        Initialize FeatureGrounder.
        
        Args:
            intervention_manager: InterventionManager to ground features for
        """
        self.manager = intervention_manager
        self.model = intervention_manager.model
        self.concept_library = ConceptLibrary()
        self.groundings: Dict[str, ConceptGrounding] = {}
    
    def add_concept(
        self, 
        name: str, 
        description: str, 
        examples: Optional[List[Any]] = None,
        properties: Optional[Dict[str, Any]] = None
    ):
        """Add a concept to the concept library."""
        self.concept_library.add_concept(name, description, examples, properties)
    
    def ground_feature_manually(
        self, 
        layer_name: str, 
        feature_index: int, 
        concept_name: str,
        confidence: float = 1.0
    ) -> ConceptGrounding:
        """
        Manually ground a feature to a concept.
        
        Args:
            layer_name: Name of layer containing the feature
            feature_index: Index of the feature
            concept_name: Name of the concept to ground to
            confidence: Confidence in the grounding
            
        Returns:
            ConceptGrounding object
        """
        if concept_name not in self.concept_library.concepts:
            raise ValueError(f"Concept '{concept_name}' not found in library")
        
        concept = self.concept_library.get_concept(concept_name)
        
        grounding = ConceptGrounding(
            layer_name=layer_name,
            parameter_type="feature",
            parameter_index=feature_index,
            concept_name=concept_name,
            concept_description=concept["description"],
            confidence=confidence,
            grounding_method="manual"
        )
        
        # Store grounding
        key = f"{layer_name}.feature.{feature_index}"
        self.groundings[key] = grounding
        
        # Update concept library
        concept["groundings"].append(grounding)
        
        return grounding
    
    def ground_prototype_manually(
        self, 
        layer_name: str, 
        prototype_index: int, 
        concept_name: str,
        confidence: float = 1.0
    ) -> ConceptGrounding:
        """
        Manually ground a prototype to a concept.
        
        Args:
            layer_name: Name of layer containing the prototype
            prototype_index: Index of the prototype
            concept_name: Name of the concept to ground to
            confidence: Confidence in the grounding
            
        Returns:
            ConceptGrounding object
        """
        if concept_name not in self.concept_library.concepts:
            raise ValueError(f"Concept '{concept_name}' not found in library")
        
        concept = self.concept_library.get_concept(concept_name)
        
        grounding = ConceptGrounding(
            layer_name=layer_name,
            parameter_type="prototype",
            parameter_index=prototype_index,
            concept_name=concept_name,
            concept_description=concept["description"],
            confidence=confidence,
            grounding_method="manual"
        )
        
        # Store grounding
        key = f"{layer_name}.prototype.{prototype_index}"
        self.groundings[key] = grounding
        
        # Update concept library
        concept["groundings"].append(grounding)
        
        return grounding
    
    def ground_features_by_activation(
        self, 
        layer_name: str,
        concept_inputs: Dict[str, torch.Tensor],
        confidence_threshold: float = 0.7
    ) -> List[ConceptGrounding]:
        """
        Ground features by analyzing their activation patterns on concept examples.
        
        Args:
            layer_name: Name of layer to analyze
            concept_inputs: Dict mapping concept names to input tensors
            confidence_threshold: Minimum confidence for automatic grounding
            
        Returns:
            List of ConceptGrounding objects
        """
        if layer_name not in self.manager.layer_names:
            raise ValueError(f"Layer '{layer_name}' not found")
        
        layer = self.manager._tnn_layers[layer_name]
        if not hasattr(layer, 'feature_bank'):
            raise ValueError(f"Layer '{layer_name}' has no feature bank")
        
        groundings = []
        
        # Compute feature activations for each concept
        concept_activations = {}
        for concept_name, inputs in concept_inputs.items():
            if concept_name not in self.concept_library.concepts:
                self.add_concept(concept_name, f"Auto-discovered concept: {concept_name}")
            
            # Compute feature membership for these inputs
            with torch.no_grad():
                # Get feature activations (input · feature > 0)
                feature_activations = torch.matmul(inputs, layer.feature_bank.T)  # [batch, num_features]
                feature_activations = (feature_activations > 0).float()
                
                # Average activation per feature for this concept
                avg_activations = feature_activations.mean(dim=0)  # [num_features]
                concept_activations[concept_name] = avg_activations
        
        # Find best concept match for each feature
        for feat_idx in range(layer.feature_bank.shape[0]):
            best_concept = None
            best_correlation = 0.0
            
            feature_activations_across_concepts = []
            concept_names = list(concept_activations.keys())
            
            for concept_name in concept_names:
                activation = concept_activations[concept_name][feat_idx].item()
                feature_activations_across_concepts.append(activation)
            
            # Find concept with highest activation for this feature
            if feature_activations_across_concepts:
                max_idx = np.argmax(feature_activations_across_concepts)
                max_activation = feature_activations_across_concepts[max_idx]
                
                if max_activation >= confidence_threshold:
                    best_concept = concept_names[max_idx]
                    best_correlation = max_activation
            
            # Create grounding if confidence is high enough
            if best_concept and best_correlation >= confidence_threshold:
                grounding = ConceptGrounding(
                    layer_name=layer_name,
                    parameter_type="feature",
                    parameter_index=feat_idx,
                    concept_name=best_concept,
                    concept_description=self.concept_library.get_concept(best_concept)["description"],
                    confidence=best_correlation,
                    activation_correlation=best_correlation,
                    grounding_method="activation_based"
                )
                
                # Store grounding
                key = f"{layer_name}.feature.{feat_idx}"
                self.groundings[key] = grounding
                groundings.append(grounding)
                
                # Update concept library
                concept = self.concept_library.get_concept(best_concept)
                concept["groundings"].append(grounding)
        
        return groundings
    
    def ground_prototypes_by_similarity(
        self, 
        layer_name: str, 
        concept_prototypes: Dict[str, torch.Tensor],
        confidence_threshold: float = 0.8
    ) -> List[ConceptGrounding]:
        """
        Ground prototypes by similarity to concept prototype vectors.
        
        Args:
            layer_name: Name of layer to analyze
            concept_prototypes: Dict mapping concept names to prototype vectors
            confidence_threshold: Minimum similarity for automatic grounding
            
        Returns:
            List of ConceptGrounding objects
        """
        if layer_name not in self.manager.layer_names:
            raise ValueError(f"Layer '{layer_name}' not found")
        
        layer = self.manager._tnn_layers[layer_name]
        if not hasattr(layer, 'prototypes'):
            raise ValueError(f"Layer '{layer_name}' has no prototypes")
        
        groundings = []
        
        # Compute similarities between learned prototypes and concept prototypes
        for proto_idx in range(layer.prototypes.shape[0]):
            learned_prototype = layer.get_prototype(proto_idx)
            
            best_concept = None
            best_similarity = 0.0
            
            for concept_name, concept_vector in concept_prototypes.items():
                if concept_name not in self.concept_library.concepts:
                    self.add_concept(concept_name, f"Concept with provided prototype: {concept_name}")
                
                # Compute cosine similarity
                similarity = F.cosine_similarity(
                    learned_prototype.unsqueeze(0), 
                    concept_vector.unsqueeze(0)
                ).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_concept = concept_name
            
            # Create grounding if similarity is high enough
            if best_concept and best_similarity >= confidence_threshold:
                grounding = ConceptGrounding(
                    layer_name=layer_name,
                    parameter_type="prototype",
                    parameter_index=proto_idx,
                    concept_name=best_concept,
                    concept_description=self.concept_library.get_concept(best_concept)["description"],
                    confidence=best_similarity,
                    visual_similarity=best_similarity,
                    grounding_method="similarity_based"
                )
                
                # Store grounding
                key = f"{layer_name}.prototype.{proto_idx}"
                self.groundings[key] = grounding
                groundings.append(grounding) 
                
                # Update concept library
                concept = self.concept_library.get_concept(best_concept)
                concept["groundings"].append(grounding)
        
        return groundings
    
    def get_grounding(
        self, 
        layer_name: str, 
        parameter_type: str, 
        parameter_index: int
    ) -> Optional[ConceptGrounding]:
        """
        Get grounding for a specific parameter.
        
        Args:
            layer_name: Name of the layer
            parameter_type: 'feature' or 'prototype'
            parameter_index: Index of the parameter
            
        Returns:
            ConceptGrounding if exists, None otherwise
        """
        key = f"{layer_name}.{parameter_type}.{parameter_index}"
        return self.groundings.get(key)
    
    def list_groundings(
        self, 
        layer_name: Optional[str] = None,
        concept_name: Optional[str] = None
    ) -> List[ConceptGrounding]:
        """
        List groundings, optionally filtered by layer or concept.
        
        Args:
            layer_name: Filter by layer name
            concept_name: Filter by concept name
            
        Returns:
            List of ConceptGrounding objects
        """
        groundings = list(self.groundings.values())
        
        if layer_name:
            groundings = [g for g in groundings if g.layer_name == layer_name]
        
        if concept_name:
            groundings = [g for g in groundings if g.concept_name == concept_name]
        
        return groundings
    
    def remove_grounding(
        self, 
        layer_name: str, 
        parameter_type: str, 
        parameter_index: int
    ) -> bool:
        """
        Remove a grounding.
        
        Args:
            layer_name: Name of the layer
            parameter_type: 'feature' or 'prototype'
            parameter_index: Index of the parameter
            
        Returns:
            True if grounding was removed, False if not found
        """
        key = f"{layer_name}.{parameter_type}.{parameter_index}"
        
        if key in self.groundings:
            grounding = self.groundings[key]
            
            # Remove from concept library
            concept = self.concept_library.get_concept(grounding.concept_name)
            if concept and grounding in concept["groundings"]:
                concept["groundings"].remove(grounding)
            
            # Remove from groundings
            del self.groundings[key]
            return True
        
        return False
    
    def explain_parameter(
        self, 
        layer_name: str, 
        parameter_type: str, 
        parameter_index: int
    ) -> str:
        """
        Generate human-readable explanation of a parameter.
        
        Args:
            layer_name: Name of the layer
            parameter_type: 'feature' or 'prototype'
            parameter_index: Index of the parameter
            
        Returns:
            Human-readable explanation string
        """
        grounding = self.get_grounding(layer_name, parameter_type, parameter_index)
        
        if grounding:
            explanation = f"{parameter_type.capitalize()} {parameter_index} in layer '{layer_name}' "
            explanation += f"represents the concept '{grounding.concept_name}': "
            explanation += f"{grounding.concept_description} "
            explanation += f"(confidence: {grounding.confidence:.2f}, "
            explanation += f"method: {grounding.grounding_method})"
            
            return explanation
        else:
            return f"{parameter_type.capitalize()} {parameter_index} in layer '{layer_name}' has no semantic grounding."
    
    def generate_model_explanation(self) -> str:
        """Generate a high-level explanation of the entire model."""
        lines = [
            f"Model Explanation for: {self.manager.model_name}",
            f"Total TNN Layers: {self.manager.num_layers}",
            f"Grounded Parameters: {len(self.groundings)}",
            f"Concepts in Library: {len(self.concept_library.concepts)}",
            "",
            "Layer-by-Layer Breakdown:"
        ]
        
        for layer_name in self.manager.layer_names:
            layer_groundings = self.list_groundings(layer_name=layer_name)
            
            lines.append(f"  {layer_name}:")
            
            if not layer_groundings:
                lines.append("    No semantic groundings")
            else:
                # Group by concept
                concept_groups = defaultdict(list)
                for grounding in layer_groundings:
                    concept_groups[grounding.concept_name].append(grounding)
                
                for concept_name, groundings in concept_groups.items():
                    concept_desc = self.concept_library.get_concept(concept_name)["description"]
                    lines.append(f"    Concept '{concept_name}': {concept_desc}")
                    
                    for grounding in groundings:
                        lines.append(
                            f"      {grounding.parameter_type} {grounding.parameter_index} "
                            f"(confidence: {grounding.confidence:.2f})"
                        )
        
        return "\n".join(lines)
    
    def validate_groundings(
        self, 
        validation_inputs: torch.Tensor,
        validation_concepts: List[str]
    ) -> Dict[str, float]:
        """
        Validate groundings using held-out validation data.
        
        Args:
            validation_inputs: Input data for validation
            validation_concepts: Expected concept labels for each input
            
        Returns:
            Dictionary with validation metrics
        """
        if len(validation_concepts) != len(validation_inputs):
            raise ValueError("Number of validation inputs must match number of concept labels")
        
        # Implementation would depend on specific validation approach
        # This is a placeholder for the validation framework
        
        results = {
            "overall_accuracy": 0.0,
            "concept_accuracies": {},
            "grounding_reliabilities": {},
            "num_validated_groundings": len(self.groundings)
        }
        
        # Placeholder validation logic
        # In a real implementation, this would:
        # 1. Run model on validation inputs
        # 2. Check which features/prototypes activate for each concept
        # 3. Compare with groundings to compute accuracy
        
        return results