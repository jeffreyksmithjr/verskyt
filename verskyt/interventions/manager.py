"""
Intervention Manager for Tversky Neural Networks.

Provides high-level APIs for inspecting and modifying TNN models,
enabling interpretability and counterfactual analysis.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import torch
import torch.nn as nn
from dataclasses import dataclass
from copy import deepcopy

from verskyt.layers.projection import TverskyProjectionLayer, TverskySimilarityLayer


@dataclass
class PrototypeInfo:
    """Information about a prototype in a TNN layer."""
    
    layer_name: str
    prototype_index: int
    vector: torch.Tensor
    layer_ref: Union[TverskyProjectionLayer, TverskySimilarityLayer]
    
    @property
    def shape(self) -> torch.Size:
        """Shape of the prototype vector."""
        return self.vector.shape
    
    @property
    def norm(self) -> float:
        """L2 norm of the prototype vector."""
        return torch.norm(self.vector).item()


@dataclass
class FeatureInfo:
    """Information about a feature in a TNN layer."""
    
    layer_name: str
    feature_index: int
    vector: torch.Tensor
    layer_ref: Union[TverskyProjectionLayer, TverskySimilarityLayer]
    
    @property
    def shape(self) -> torch.Size:
        """Shape of the feature vector."""
        return self.vector.shape
    
    @property
    def norm(self) -> float:
        """L2 norm of the feature vector."""
        return torch.norm(self.vector).item()


class InterventionManager:
    """
    Manager for interventions on Tversky Neural Networks.
    
    Provides unified API for:
    - Inspecting prototypes and features across all layers
    - Modifying prototypes with impact tracking
    - Analyzing model behavior under interventions
    - Grounding features to semantic concepts
    """
    
    def __init__(self, model: nn.Module, model_name: str = "TNN_Model"):
        """
        Initialize InterventionManager for a TNN model.
        
        Args:
            model: PyTorch model containing TverskyProjectionLayer or TverskySimilarityLayer
            model_name: Human-readable name for the model
        """
        self.model = model
        self.model_name = model_name
        self._tnn_layers = self._discover_tnn_layers()
        
        # Track original state for impact assessment
        self._original_state = self._capture_model_state()
        self._intervention_history: List[Dict[str, Any]] = []
    
    def _discover_tnn_layers(self) -> Dict[str, Union[TverskyProjectionLayer, TverskySimilarityLayer]]:
        """Discover all TNN layers in the model."""
        tnn_layers = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (TverskyProjectionLayer, TverskySimilarityLayer)):
                tnn_layers[name] = module
        
        return tnn_layers
    
    def _capture_model_state(self) -> Dict[str, torch.Tensor]:
        """Capture current state of all TNN layer parameters."""
        state = {}
        
        for layer_name, layer in self._tnn_layers.items():
            if hasattr(layer, 'prototypes'):
                state[f"{layer_name}.prototypes"] = layer.prototypes.data.clone()
            if hasattr(layer, 'feature_bank'):
                state[f"{layer_name}.feature_bank"] = layer.feature_bank.data.clone()
            if hasattr(layer, 'alpha'):
                state[f"{layer_name}.alpha"] = layer.alpha.data.clone()
            if hasattr(layer, 'beta'):
                state[f"{layer_name}.beta"] = layer.beta.data.clone()
        
        return state
    
    @property
    def num_layers(self) -> int:
        """Number of TNN layers in the model."""
        return len(self._tnn_layers)
    
    @property
    def layer_names(self) -> List[str]:
        """Names of all TNN layers."""
        return list(self._tnn_layers.keys())
    
    def get_layer_info(self, layer_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a TNN layer.
        
        Args:
            layer_name: Name of the layer to inspect
            
        Returns:
            Dictionary with layer configuration and parameter info
        """
        if layer_name not in self._tnn_layers:
            raise ValueError(f"Layer '{layer_name}' not found. Available: {self.layer_names}")
        
        layer = self._tnn_layers[layer_name]
        
        info = {
            "layer_name": layer_name,
            "layer_type": type(layer).__name__,
            "in_features": layer.in_features,
        }
        
        # Add layer-specific information
        if isinstance(layer, TverskyProjectionLayer):
            info.update({
                "num_prototypes": layer.num_prototypes,
                "num_features": layer.num_features,
                "has_bias": layer.bias is not None,
                "shared_features": getattr(layer, 'shared_features', False),
            })
        elif isinstance(layer, TverskySimilarityLayer):
            info.update({
                "num_features": layer.num_features,
                "use_contrast_form": layer.use_contrast_form,
            })
        
        # Add parameter information
        if hasattr(layer, 'alpha'):
            info["alpha"] = layer.alpha.item()
        if hasattr(layer, 'beta'):
            info["beta"] = layer.beta.item()
        if hasattr(layer, 'theta'):
            if isinstance(layer.theta, torch.Tensor):
                info["theta"] = layer.theta.item()
            else:
                info["theta"] = layer.theta
        
        # Add reduction methods
        info["intersection_reduction"] = str(layer.intersection_reduction)
        info["difference_reduction"] = str(layer.difference_reduction)
        
        return info
    
    def list_prototypes(self, layer_name: Optional[str] = None) -> List[PrototypeInfo]:
        """
        List all prototypes in the model or specific layer.
        
        Args:
            layer_name: If specified, only return prototypes from this layer
            
        Returns:
            List of PrototypeInfo objects
        """
        prototypes = []
        
        layers_to_check = [layer_name] if layer_name else self.layer_names
        
        for name in layers_to_check:
            if name not in self._tnn_layers:
                continue
            
            layer = self._tnn_layers[name]
            if hasattr(layer, 'prototypes'):
                for i in range(layer.prototypes.shape[0]):
                    prototypes.append(PrototypeInfo(
                        layer_name=name,
                        prototype_index=i,
                        vector=layer.get_prototype(i),
                        layer_ref=layer
                    ))
        
        return prototypes
    
    def list_features(self, layer_name: Optional[str] = None) -> List[FeatureInfo]:
        """
        List all features in the model or specific layer.
        
        Args:
            layer_name: If specified, only return features from this layer
            
        Returns:
            List of FeatureInfo objects
        """
        features = []
        
        layers_to_check = [layer_name] if layer_name else self.layer_names
        
        for name in layers_to_check:
            if name not in self._tnn_layers:
                continue
            
            layer = self._tnn_layers[name]
            if hasattr(layer, 'feature_bank'):
                for i in range(layer.feature_bank.shape[0]):
                    features.append(FeatureInfo(
                        layer_name=name,
                        feature_index=i,
                        vector=layer.get_feature(i),
                        layer_ref=layer
                    ))
        
        return features
    
    def get_prototype(self, layer_name: str, prototype_index: int) -> PrototypeInfo:
        """
        Get specific prototype information.
        
        Args:
            layer_name: Name of the layer
            prototype_index: Index of the prototype
            
        Returns:
            PrototypeInfo object
        """
        if layer_name not in self._tnn_layers:
            raise ValueError(f"Layer '{layer_name}' not found")
        
        layer = self._tnn_layers[layer_name]
        if not hasattr(layer, 'prototypes'):
            raise ValueError(f"Layer '{layer_name}' has no prototypes")
        
        if prototype_index >= layer.prototypes.shape[0]:
            raise ValueError(f"Prototype index {prototype_index} out of range for layer '{layer_name}'")
        
        return PrototypeInfo(
            layer_name=layer_name,
            prototype_index=prototype_index,
            vector=layer.get_prototype(prototype_index),
            layer_ref=layer
        )
    
    def get_feature(self, layer_name: str, feature_index: int) -> FeatureInfo:
        """
        Get specific feature information.
        
        Args:
            layer_name: Name of the layer
            feature_index: Index of the feature
            
        Returns:
            FeatureInfo object
        """
        if layer_name not in self._tnn_layers:
            raise ValueError(f"Layer '{layer_name}' not found")
        
        layer = self._tnn_layers[layer_name]
        if not hasattr(layer, 'feature_bank'):
            raise ValueError(f"Layer '{layer_name}' has no feature bank")
        
        if feature_index >= layer.feature_bank.shape[0]:
            raise ValueError(f"Feature index {feature_index} out of range for layer '{layer_name}'")
        
        return FeatureInfo(
            layer_name=layer_name,
            feature_index=feature_index,
            vector=layer.get_feature(feature_index),
            layer_ref=layer
        )
    
    def modify_prototype(
        self, 
        layer_name: str, 
        prototype_index: int, 
        new_vector: torch.Tensor,
        track_intervention: bool = True
    ) -> PrototypeInfo:
        """
        Modify a prototype vector.
        
        Args:
            layer_name: Name of the layer
            prototype_index: Index of the prototype to modify
            new_vector: New prototype vector
            track_intervention: Whether to track this intervention for impact assessment
            
        Returns:
            Updated PrototypeInfo object
        """
        if layer_name not in self._tnn_layers:
            raise ValueError(f"Layer '{layer_name}' not found")
        
        layer = self._tnn_layers[layer_name]
        if not hasattr(layer, 'prototypes'):
            raise ValueError(f"Layer '{layer_name}' has no prototypes")
        
        # Validate dimensions
        expected_shape = layer.prototypes[prototype_index].shape
        if new_vector.shape != expected_shape:
            raise ValueError(f"New vector shape {new_vector.shape} doesn't match expected {expected_shape}")
        
        # Store original for tracking
        if track_intervention:
            original_vector = layer.get_prototype(prototype_index)
            intervention_record = {
                "type": "prototype_modification",
                "layer_name": layer_name,
                "prototype_index": prototype_index,
                "original_vector": original_vector.clone(),
                "new_vector": new_vector.clone(),
                "timestamp": torch.tensor(len(self._intervention_history), dtype=torch.long)
            }
            self._intervention_history.append(intervention_record)
        
        # Apply modification
        layer.set_prototype(prototype_index, new_vector)
        
        return self.get_prototype(layer_name, prototype_index)
    
    def modify_feature(
        self, 
        layer_name: str, 
        feature_index: int, 
        new_vector: torch.Tensor,
        track_intervention: bool = True
    ) -> FeatureInfo:
        """
        Modify a feature vector.
        
        Args:
            layer_name: Name of the layer
            feature_index: Index of the feature to modify
            new_vector: New feature vector
            track_intervention: Whether to track this intervention for impact assessment
            
        Returns:
            Updated FeatureInfo object
        """
        if layer_name not in self._tnn_layers:
            raise ValueError(f"Layer '{layer_name}' not found")
        
        layer = self._tnn_layers[layer_name]
        if not hasattr(layer, 'feature_bank'):
            raise ValueError(f"Layer '{layer_name}' has no feature bank")
        
        # Validate dimensions
        expected_shape = layer.feature_bank[feature_index].shape
        if new_vector.shape != expected_shape:
            raise ValueError(f"New vector shape {new_vector.shape} doesn't match expected {expected_shape}")
        
        # Store original for tracking
        if track_intervention:
            original_vector = layer.get_feature(feature_index)
            intervention_record = {
                "type": "feature_modification",
                "layer_name": layer_name,
                "feature_index": feature_index,
                "original_vector": original_vector.clone(),
                "new_vector": new_vector.clone(),
                "timestamp": torch.tensor(len(self._intervention_history), dtype=torch.long)
            }
            self._intervention_history.append(intervention_record)
        
        # Apply modification
        layer.set_feature(feature_index, new_vector)
        
        return self.get_feature(layer_name, feature_index)
    
    def reset_to_original(self) -> None:
        """Reset all TNN layers to their original state."""
        for param_name, original_value in self._original_state.items():
            layer_name, param_type = param_name.rsplit('.', 1)
            layer = self._tnn_layers[layer_name]
            
            if param_type == 'prototypes' and hasattr(layer, 'prototypes'):
                layer.prototypes.data.copy_(original_value)
            elif param_type == 'feature_bank' and hasattr(layer, 'feature_bank'):
                layer.feature_bank.data.copy_(original_value)
            elif param_type == 'alpha' and hasattr(layer, 'alpha'):
                layer.alpha.data.copy_(original_value)
            elif param_type == 'beta' and hasattr(layer, 'beta'):
                layer.beta.data.copy_(original_value)
        
        # Clear intervention history
        self._intervention_history.clear()
    
    def get_intervention_history(self) -> List[Dict[str, Any]]:
        """Get history of all interventions performed."""
        return self._intervention_history.copy()
    
    def summary(self) -> str:
        """Get a summary of the model and available interventions."""
        lines = [
            f"Intervention Manager for: {self.model_name}",
            f"TNN Layers: {self.num_layers}",
            "",
            "Layer Details:"
        ]
        
        for layer_name in self.layer_names:
            info = self.get_layer_info(layer_name)
            lines.append(f"  {layer_name}: {info['layer_type']}")
            
            if 'num_prototypes' in info:
                lines.append(f"    Prototypes: {info['num_prototypes']}")
            if 'num_features' in info:
                lines.append(f"    Features: {info['num_features']}")
            
            lines.append(f"    α={info.get('alpha', 'N/A'):.3f}, β={info.get('beta', 'N/A'):.3f}")
        
        lines.extend([
            "",
            f"Interventions Performed: {len(self._intervention_history)}",
            f"Available Operations: inspect, modify, analyze, reset"
        ])
        
        return "\n".join(lines)