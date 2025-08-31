"""
Intervention Manager for Tversky Neural Networks.

Provides high-level APIs for inspecting and modifying TNN models,
enabling interpretability and counterfactual analysis.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from verskyt.layers.projection import TverskyProjectionLayer, TverskySimilarityLayer


@dataclass
class PrototypeInfo:
    """Information about a prototype in a TNN layer.

    Contains metadata and vector data for a single prototype, enabling
    inspection and modification of learned prototype representations.

    Attributes:
        layer_name (str): Name of the layer containing this prototype.
        prototype_index (int): Index of the prototype within the layer.
        vector (torch.Tensor): The prototype vector data.
        layer_ref (Union[TverskyProjectionLayer, TverskySimilarityLayer]):
            Reference to the layer object.
    """

    layer_name: str
    prototype_index: int
    vector: torch.Tensor
    layer_ref: Union[TverskyProjectionLayer, TverskySimilarityLayer]

    @property
    def shape(self) -> torch.Size:
        """Get the shape of the prototype vector.

        Returns:
            torch.Size: Shape of the prototype vector, typically [in_features].
        """
        return self.vector.shape

    @property
    def norm(self) -> float:
        """Get the L2 norm of the prototype vector.

        Returns:
            float: L2 norm of the prototype vector, useful for comparing
                prototype magnitudes and analyzing learned representations.
        """
        return torch.norm(self.vector).item()


@dataclass
class FeatureInfo:
    """Information about a feature in a TNN layer.

    Contains metadata and vector data for a single feature, enabling
    inspection and modification of learned feature representations.

    Attributes:
        layer_name (str): Name of the layer containing this feature.
        feature_index (int): Index of the feature within the layer's feature bank.
        vector (torch.Tensor): The feature vector data.
        layer_ref (Union[TverskyProjectionLayer, TverskySimilarityLayer]):
            Reference to the layer object.
    """

    layer_name: str
    feature_index: int
    vector: torch.Tensor
    layer_ref: Union[TverskyProjectionLayer, TverskySimilarityLayer]

    @property
    def shape(self) -> torch.Size:
        """Get the shape of the feature vector.

        Returns:
            torch.Size: Shape of the feature vector, typically [in_features].
        """
        return self.vector.shape

    @property
    def norm(self) -> float:
        """Get the L2 norm of the feature vector.

        Returns:
            float: L2 norm of the feature vector, useful for comparing
                feature magnitudes and analyzing learned representations.
        """
        return torch.norm(self.vector).item()


class InterventionManager:
    """Manager for interventions on Tversky Neural Networks.

    Provides a unified API for inspecting, modifying, and analyzing TNN models
    to enable interpretability research and counterfactual analysis. Supports
    tracking of interventions and restoration of original model states.

    This class serves as the central hub for TNN interpretability, offering:
    - Comprehensive prototype and feature discovery across all layers
    - Safe parameter modification with automatic state tracking
    - Integration with impact assessment and grounding frameworks
    - Batch operations for systematic intervention studies

    Note:
        The manager automatically discovers TNN layers (TverskyProjectionLayer
        and TverskySimilarityLayer) within the provided model and maintains
        original parameter states for restoration.
    """

    def __init__(self, model: nn.Module, model_name: str = "TNN_Model"):
        """Initialize InterventionManager for a TNN model.

        Automatically discovers all TNN layers within the model and captures
        the original parameter state for later restoration.

        Args:
            model (nn.Module): PyTorch model containing TverskyProjectionLayer
                or TverskySimilarityLayer instances.
            model_name (str, optional): Human-readable name for the model.
                Defaults to "TNN_Model".

        Note:
            The manager will only operate on TverskyProjectionLayer and
            TverskySimilarityLayer instances found within the model.
        """
        self.model = model
        self.model_name = model_name
        self._tnn_layers = self._discover_tnn_layers()

        # Track original state for impact assessment
        self._original_state = self._capture_model_state()
        self._intervention_history: List[Dict[str, Any]] = []

    def _discover_tnn_layers(
        self,
    ) -> Dict[str, Union[TverskyProjectionLayer, TverskySimilarityLayer]]:
        """Discover all TNN layers in the model.

        Recursively searches through all modules in the model to find
        TverskyProjectionLayer and TverskySimilarityLayer instances.

        Returns:
            Dict[str, Union[TverskyProjectionLayer, TverskySimilarityLayer]]:
                Dictionary mapping layer names to layer objects.
        """
        tnn_layers = {}

        for name, module in self.model.named_modules():
            if isinstance(module, (TverskyProjectionLayer, TverskySimilarityLayer)):
                tnn_layers[name] = module

        return tnn_layers

    def _capture_model_state(self) -> Dict[str, torch.Tensor]:
        """Capture current state of all TNN layer parameters.

        Creates deep copies of all prototype and feature bank tensors
        to enable restoration after interventions.

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping parameter names
                to cloned tensor data.
        """
        state = {}

        for layer_name, layer in self._tnn_layers.items():
            if hasattr(layer, "prototypes"):
                state[f"{layer_name}.prototypes"] = layer.prototypes.data.clone()
            if hasattr(layer, "feature_bank"):
                state[f"{layer_name}.feature_bank"] = layer.feature_bank.data.clone()
            if hasattr(layer, "alpha"):
                state[f"{layer_name}.alpha"] = layer.alpha.data.clone()
            if hasattr(layer, "beta"):
                state[f"{layer_name}.beta"] = layer.beta.data.clone()

        return state

    @property
    def num_layers(self) -> int:
        """Get the number of TNN layers in the model.

        Returns:
            int: Total count of TverskyProjectionLayer and TverskySimilarityLayer
                instances found in the model.
        """
        return len(self._tnn_layers)

    @property
    def layer_names(self) -> List[str]:
        """Get names of all TNN layers in the model.

        Returns:
            List[str]: List of layer names that can be used with other
                manager methods for layer-specific operations.
        """
        return list(self._tnn_layers.keys())

    def get_layer_info(self, layer_name: str) -> Dict[str, Any]:
        """Get comprehensive information about a TNN layer.

        Provides detailed metadata about layer configuration, parameter shapes,
        and capabilities for inspection and intervention planning.

        Args:
            layer_name (str): Name of the layer to inspect. Must be one of
                the names returned by the layer_names property.

        Returns:
            Dict[str, Any]: Dictionary containing layer metadata including:
                - layer_name: Name of the layer
                - layer_type: Class name of the layer
                - in_features: Input feature dimension
                - num_prototypes: Number of prototypes (if applicable)
                - num_features: Number of features (if applicable)
                - learnable_ab: Whether alpha/beta are learnable (if applicable)

        Raises:
            ValueError: If layer_name is not found in the model.
        """
        if layer_name not in self._tnn_layers:
            raise ValueError(
                f"Layer '{layer_name}' not found. Available: {self.layer_names}"
            )

        layer = self._tnn_layers[layer_name]

        info = {
            "layer_name": layer_name,
            "layer_type": type(layer).__name__,
            "in_features": layer.in_features,
        }

        # Add layer-specific information
        if isinstance(layer, TverskyProjectionLayer):
            info.update(
                {
                    "num_prototypes": layer.num_prototypes,
                    "num_features": layer.num_features,
                    "has_bias": layer.bias is not None,
                    "shared_features": getattr(layer, "shared_features", False),
                }
            )
        elif isinstance(layer, TverskySimilarityLayer):
            info.update(
                {
                    "num_features": layer.num_features,
                    "use_contrast_form": layer.use_contrast_form,
                }
            )

        # Add parameter information
        if hasattr(layer, "alpha"):
            info["alpha"] = layer.alpha.item()
        if hasattr(layer, "beta"):
            info["beta"] = layer.beta.item()
        if hasattr(layer, "theta"):
            if isinstance(layer.theta, torch.Tensor):
                info["theta"] = layer.theta.item()
            else:
                info["theta"] = layer.theta

        # Add reduction methods
        info["intersection_reduction"] = str(layer.intersection_reduction)
        info["difference_reduction"] = str(layer.difference_reduction)

        return info

    def list_prototypes(self, layer_name: Optional[str] = None) -> List[PrototypeInfo]:
        """List all prototypes in the model or specific layer.

        Discovers and returns metadata for all prototype vectors across
        TNN layers, enabling systematic inspection and analysis.

        Args:
            layer_name (Optional[str], optional): If specified, only return
                prototypes from this layer. If None, returns prototypes from
                all layers. Defaults to None.

        Returns:
            List[PrototypeInfo]: List of PrototypeInfo objects containing
                prototype vectors and metadata. Each object provides access
                to the prototype vector, layer reference, and computed properties.

        Note:
            Only layers with 'prototypes' attribute (typically TverskyProjectionLayer)
            will contribute to the returned list.
        """
        prototypes = []

        layers_to_check = [layer_name] if layer_name else self.layer_names

        for name in layers_to_check:
            if name not in self._tnn_layers:
                continue

            layer = self._tnn_layers[name]
            if hasattr(layer, "prototypes"):
                for i in range(layer.prototypes.shape[0]):
                    prototypes.append(
                        PrototypeInfo(
                            layer_name=name,
                            prototype_index=i,
                            vector=layer.get_prototype(i),
                            layer_ref=layer,
                        )
                    )

        return prototypes

    def list_features(self, layer_name: Optional[str] = None) -> List[FeatureInfo]:
        """List all features in the model or specific layer.

        Discovers and returns metadata for all feature vectors across
        TNN layers, enabling systematic inspection and analysis of the
        learned feature representations.

        Args:
            layer_name (Optional[str], optional): If specified, only return
                features from this layer. If None, returns features from
                all layers. Defaults to None.

        Returns:
            List[FeatureInfo]: List of FeatureInfo objects containing
                feature vectors and metadata. Each object provides access
                to the feature vector, layer reference, and computed properties.

        Note:
            Only layers with 'feature_bank' attribute will contribute to
            the returned list. This typically includes both TverskyProjectionLayer
            and TverskySimilarityLayer instances.
        """
        features = []

        layers_to_check = [layer_name] if layer_name else self.layer_names

        for name in layers_to_check:
            if name not in self._tnn_layers:
                continue

            layer = self._tnn_layers[name]
            if hasattr(layer, "feature_bank"):
                for i in range(layer.feature_bank.shape[0]):
                    features.append(
                        FeatureInfo(
                            layer_name=name,
                            feature_index=i,
                            vector=layer.get_feature(i),
                            layer_ref=layer,
                        )
                    )

        return features

    def get_prototype(self, layer_name: str, prototype_index: int) -> PrototypeInfo:
        """Get specific prototype information.

        Retrieves detailed information about a single prototype vector,
        including its current values and layer context.

        Args:
            layer_name (str): Name of the layer containing the prototype.
                Must be one of the names returned by layer_names.
            prototype_index (int): Index of the prototype within the layer.
                Must be in range [0, num_prototypes).

        Returns:
            PrototypeInfo: Object containing the prototype vector, metadata,
                and layer reference for further operations.

        Raises:
            ValueError: If layer_name is not found or layer has no prototypes.
            IndexError: If prototype_index is out of bounds.
        """
        if layer_name not in self._tnn_layers:
            raise ValueError(f"Layer '{layer_name}' not found")

        layer = self._tnn_layers[layer_name]
        if not hasattr(layer, "prototypes"):
            raise ValueError(f"Layer '{layer_name}' has no prototypes")

        if prototype_index >= layer.prototypes.shape[0]:
            raise ValueError(
                f"Prototype index {prototype_index} out of range for "
                f"layer '{layer_name}'"
            )

        return PrototypeInfo(
            layer_name=layer_name,
            prototype_index=prototype_index,
            vector=layer.get_prototype(prototype_index),
            layer_ref=layer,
        )

    def get_feature(self, layer_name: str, feature_index: int) -> FeatureInfo:
        """Get specific feature information.

        Retrieves detailed information about a single feature vector,
        including its current values and layer context.

        Args:
            layer_name (str): Name of the layer containing the feature.
                Must be one of the names returned by layer_names.
            feature_index (int): Index of the feature within the layer's
                feature bank. Must be in range [0, num_features).

        Returns:
            FeatureInfo: Object containing the feature vector, metadata,
                and layer reference for further operations.

        Raises:
            ValueError: If layer_name is not found or layer has no feature bank.
            IndexError: If feature_index is out of bounds.
        """
        if layer_name not in self._tnn_layers:
            raise ValueError(f"Layer '{layer_name}' not found")

        layer = self._tnn_layers[layer_name]
        if not hasattr(layer, "feature_bank"):
            raise ValueError(f"Layer '{layer_name}' has no feature bank")

        if feature_index >= layer.feature_bank.shape[0]:
            raise ValueError(
                f"Feature index {feature_index} out of range for "
                f"layer '{layer_name}'"
            )

        return FeatureInfo(
            layer_name=layer_name,
            feature_index=feature_index,
            vector=layer.get_feature(feature_index),
            layer_ref=layer,
        )

    def modify_prototype(
        self,
        layer_name: str,
        prototype_index: int,
        new_vector: torch.Tensor,
        track_intervention: bool = True,
    ) -> PrototypeInfo:
        """Modify a prototype vector in a TNN layer.

        Safely modifies a prototype vector with automatic validation and
        optional intervention tracking for impact assessment and restoration.

        Args:
            layer_name (str): Name of the layer containing the prototype.
                Must be one of the names returned by layer_names.
            prototype_index (int): Index of the prototype to modify.
                Must be in range [0, num_prototypes).
            new_vector (torch.Tensor): New prototype vector to set.
                Must match the shape of the existing prototype.
            track_intervention (bool, optional): Whether to record this
                intervention in the history for impact assessment.
                Defaults to True.

        Returns:
            PrototypeInfo: Updated PrototypeInfo object reflecting the
                new prototype vector state.

        Raises:
            ValueError: If layer_name is not found, layer has no prototypes,
                or new_vector shape doesn't match expected dimensions.
            IndexError: If prototype_index is out of bounds.

        Note:
            When track_intervention=True, the original vector is stored
            for potential restoration via reset_to_original().
        """
        if layer_name not in self._tnn_layers:
            raise ValueError(f"Layer '{layer_name}' not found")

        layer = self._tnn_layers[layer_name]
        if not hasattr(layer, "prototypes"):
            raise ValueError(f"Layer '{layer_name}' has no prototypes")

        # Validate dimensions
        expected_shape = layer.prototypes[prototype_index].shape
        if new_vector.shape != expected_shape:
            raise ValueError(
                f"New vector shape {new_vector.shape} doesn't match "
                f"expected {expected_shape}"
            )

        # Store original for tracking
        if track_intervention:
            original_vector = layer.get_prototype(prototype_index)
            intervention_record = {
                "type": "prototype_modification",
                "layer_name": layer_name,
                "prototype_index": prototype_index,
                "original_vector": original_vector.clone(),
                "new_vector": new_vector.clone(),
                "timestamp": torch.tensor(
                    len(self._intervention_history), dtype=torch.long
                ),
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
        track_intervention: bool = True,
    ) -> FeatureInfo:
        """Modify a feature vector in a TNN layer.

        Safely modifies a feature vector with automatic validation and
        optional intervention tracking for impact assessment and restoration.

        Args:
            layer_name (str): Name of the layer containing the feature.
                Must be one of the names returned by layer_names.
            feature_index (int): Index of the feature to modify within
                the layer's feature bank. Must be in range [0, num_features).
            new_vector (torch.Tensor): New feature vector to set.
                Must match the shape of the existing feature.
            track_intervention (bool, optional): Whether to record this
                intervention in the history for impact assessment.
                Defaults to True.

        Returns:
            FeatureInfo: Updated FeatureInfo object reflecting the
                new feature vector state.

        Raises:
            ValueError: If layer_name is not found, layer has no feature bank,
                or new_vector shape doesn't match expected dimensions.
            IndexError: If feature_index is out of bounds.

        Note:
            When track_intervention=True, the original vector is stored
            for potential restoration via reset_to_original().
        """
        if layer_name not in self._tnn_layers:
            raise ValueError(f"Layer '{layer_name}' not found")

        layer = self._tnn_layers[layer_name]
        if not hasattr(layer, "feature_bank"):
            raise ValueError(f"Layer '{layer_name}' has no feature bank")

        # Validate dimensions
        expected_shape = layer.feature_bank[feature_index].shape
        if new_vector.shape != expected_shape:
            raise ValueError(
                f"New vector shape {new_vector.shape} doesn't match "
                f"expected {expected_shape}"
            )

        # Store original for tracking
        if track_intervention:
            original_vector = layer.get_feature(feature_index)
            intervention_record = {
                "type": "feature_modification",
                "layer_name": layer_name,
                "feature_index": feature_index,
                "original_vector": original_vector.clone(),
                "new_vector": new_vector.clone(),
                "timestamp": torch.tensor(
                    len(self._intervention_history), dtype=torch.long
                ),
            }
            self._intervention_history.append(intervention_record)

        # Apply modification
        layer.set_feature(feature_index, new_vector)

        return self.get_feature(layer_name, feature_index)

    def reset_to_original(self) -> None:
        """Reset all TNN layers to their original state.

        Restores all prototype vectors, feature vectors, and learnable
        parameters (alpha, beta) to their values at manager initialization.
        Also clears the intervention history.

        Note:
            This operation cannot be undone. All modifications made through
            modify_prototype() and modify_feature() will be reverted to the
            original model state.
        """
        for param_name, original_value in self._original_state.items():
            layer_name, param_type = param_name.rsplit(".", 1)
            layer = self._tnn_layers[layer_name]

            if param_type == "prototypes" and hasattr(layer, "prototypes"):
                layer.prototypes.data.copy_(original_value)
            elif param_type == "feature_bank" and hasattr(layer, "feature_bank"):
                layer.feature_bank.data.copy_(original_value)
            elif param_type == "alpha" and hasattr(layer, "alpha"):
                layer.alpha.data.copy_(original_value)
            elif param_type == "beta" and hasattr(layer, "beta"):
                layer.beta.data.copy_(original_value)

        # Clear intervention history
        self._intervention_history.clear()

    def get_intervention_history(self) -> List[Dict[str, Any]]:
        """Get history of all interventions performed.

        Returns a copy of the intervention history containing detailed
        records of all modifications made through this manager.

        Returns:
            List[Dict[str, Any]]: List of intervention records, each containing:
                - type: Type of intervention ('prototype_modification'
                    or 'feature_modification')
                - layer_name: Name of the affected layer
                - index: Index of the modified parameter
                - original_vector: Original parameter vector (cloned)
                - new_vector: New parameter vector (cloned)
                - timestamp: Sequential intervention number
        """
        return self._intervention_history.copy()

    def summary(self) -> str:
        """Get a summary of the model and available interventions.

        Provides a comprehensive overview of the model structure,
        TNN layers, and intervention capabilities for inspection.

        Returns:
            str: Multi-line summary string containing:
                - Model name and layer count
                - Detailed information for each TNN layer
                - Number of prototypes and features per layer
                - Parameter values (alpha, beta, theta)
                - Total intervention count
        """
        lines = [
            f"Intervention Manager for: {self.model_name}",
            f"TNN Layers: {self.num_layers}",
            "",
            "Layer Details:",
        ]

        for layer_name in self.layer_names:
            info = self.get_layer_info(layer_name)
            lines.append(f"  {layer_name}: {info['layer_type']}")

            if "num_prototypes" in info:
                lines.append(f"    Prototypes: {info['num_prototypes']}")
            if "num_features" in info:
                lines.append(f"    Features: {info['num_features']}")

            lines.append(
                f"    α={info.get('alpha', 'N/A'):.3f}, β={info.get('beta', 'N/A'):.3f}"
            )

        lines.extend(
            [
                "",
                f"Interventions Performed: {len(self._intervention_history)}",
                "Available Operations: inspect, modify, analyze, reset",
            ]
        )

        return "\n".join(lines)
