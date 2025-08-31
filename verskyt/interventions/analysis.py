"""
Analysis tools for TNN interventions.

Provides counterfactual analysis and impact assessment capabilities
for understanding how interventions affect model behavior.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .manager import InterventionManager


@dataclass
class ImpactMetrics:
    """Metrics quantifying the impact of an intervention."""

    # Output-level metrics
    output_distance: float  # L2 distance between original and modified outputs
    output_correlation: float  # Correlation between original and modified outputs
    prediction_change_rate: float  # Fraction of samples with changed predictions
    confidence_change: float  # Average change in prediction confidence

    # Feature-level metrics (if applicable)
    feature_activation_change: Optional[torch.Tensor] = None
    similarity_score_change: Optional[torch.Tensor] = None

    # Statistical metrics
    effect_size: float = 0.0  # Cohen's d or similar effect size measure
    significance: Optional[float] = None  # p-value if statistical test performed


@dataclass
class CounterfactualResult:
    """Result of a counterfactual analysis."""

    original_input: torch.Tensor
    original_output: torch.Tensor
    original_prediction: int

    modified_input: torch.Tensor
    modified_output: torch.Tensor
    modified_prediction: int

    intervention_description: str
    success: bool  # Whether intervention achieved desired outcome

    # Metrics
    input_perturbation_norm: float
    output_change_norm: float
    confidence_change: float


class ImpactAssessment:
    """
    Assess the impact of interventions on model behavior.

    Provides methods to quantify how prototype or feature modifications
    affect model outputs across different inputs.
    """

    def __init__(self, intervention_manager: InterventionManager):
        """
        Initialize ImpactAssessment.

        Args:
            intervention_manager: InterventionManager instance to analyze
        """
        self.manager = intervention_manager
        self.model = intervention_manager.model

    def assess_prototype_impact(
        self,
        layer_name: str,
        prototype_index: int,
        new_vector: torch.Tensor,
        test_inputs: torch.Tensor,
        test_targets: Optional[torch.Tensor] = None,
    ) -> ImpactMetrics:
        """
        Assess impact of modifying a prototype.

        Args:
            layer_name: Name of layer containing the prototype
            prototype_index: Index of prototype to modify
            new_vector: New prototype vector to test
            test_inputs: Input data to test impact on
            test_targets: Optional target labels for computing accuracy changes

        Returns:
            ImpactMetrics quantifying the intervention's effects
        """
        # Get original outputs
        self.model.eval()
        with torch.no_grad():
            original_outputs = self.model(test_inputs)
            original_predictions = torch.argmax(original_outputs, dim=1)
            original_confidences = F.softmax(original_outputs, dim=1).max(dim=1)[0]

        # Store original prototype
        original_prototype = self.manager.get_prototype(layer_name, prototype_index)

        try:
            # Apply intervention
            self.manager.modify_prototype(
                layer_name, prototype_index, new_vector, track_intervention=False
            )

            # Get modified outputs
            with torch.no_grad():
                modified_outputs = self.model(test_inputs)
                modified_predictions = torch.argmax(modified_outputs, dim=1)
                modified_confidences = F.softmax(modified_outputs, dim=1).max(dim=1)[0]

            # Compute metrics
            output_distance = torch.norm(modified_outputs - original_outputs).item()

            # Handle correlation computation carefully
            orig_flat = original_outputs.flatten()
            mod_flat = modified_outputs.flatten()
            if torch.std(orig_flat) > 1e-8 and torch.std(mod_flat) > 1e-8:
                corrcoef = torch.corrcoef(torch.stack([orig_flat, mod_flat]))
                output_correlation = corrcoef[0, 1].item()
            else:
                output_correlation = 1.0 if torch.allclose(orig_flat, mod_flat) else 0.0

            prediction_change_rate = (
                (original_predictions != modified_predictions).float().mean().item()
            )
            confidence_change = (
                (modified_confidences - original_confidences).mean().item()
            )

            # Compute effect size (Cohen's d)
            pooled_std = torch.sqrt(
                (torch.var(original_outputs) + torch.var(modified_outputs)) / 2
            )
            if pooled_std > 1e-8:
                effect_size = (
                    torch.mean(modified_outputs - original_outputs).item()
                    / pooled_std.item()
                )
            else:
                effect_size = 0.0

            return ImpactMetrics(
                output_distance=output_distance,
                output_correlation=output_correlation,
                prediction_change_rate=prediction_change_rate,
                confidence_change=confidence_change,
                effect_size=effect_size,
            )

        finally:
            # Restore original prototype
            self.manager.modify_prototype(
                layer_name,
                prototype_index,
                original_prototype.vector,
                track_intervention=False,
            )

    def assess_feature_impact(
        self,
        layer_name: str,
        feature_index: int,
        new_vector: torch.Tensor,
        test_inputs: torch.Tensor,
        test_targets: Optional[torch.Tensor] = None,
    ) -> ImpactMetrics:
        """
        Assess impact of modifying a feature.

        Args:
            layer_name: Name of layer containing the feature
            feature_index: Index of feature to modify
            new_vector: New feature vector to test
            test_inputs: Input data to test impact on
            test_targets: Optional target labels for computing accuracy changes

        Returns:
            ImpactMetrics quantifying the intervention's effects
        """
        # Get original outputs
        self.model.eval()
        with torch.no_grad():
            original_outputs = self.model(test_inputs)
            original_predictions = torch.argmax(original_outputs, dim=1)
            original_confidences = F.softmax(original_outputs, dim=1).max(dim=1)[0]

        # Store original feature
        original_feature = self.manager.get_feature(layer_name, feature_index)

        try:
            # Apply intervention
            self.manager.modify_feature(
                layer_name, feature_index, new_vector, track_intervention=False
            )

            # Get modified outputs
            with torch.no_grad():
                modified_outputs = self.model(test_inputs)
                modified_predictions = torch.argmax(modified_outputs, dim=1)
                modified_confidences = F.softmax(modified_outputs, dim=1).max(dim=1)[0]

            # Compute metrics (same as prototype impact)
            output_distance = torch.norm(modified_outputs - original_outputs).item()

            orig_flat = original_outputs.flatten()
            mod_flat = modified_outputs.flatten()
            if torch.std(orig_flat) > 1e-8 and torch.std(mod_flat) > 1e-8:
                corrcoef = torch.corrcoef(torch.stack([orig_flat, mod_flat]))
                output_correlation = corrcoef[0, 1].item()
            else:
                output_correlation = 1.0 if torch.allclose(orig_flat, mod_flat) else 0.0

            prediction_change_rate = (
                (original_predictions != modified_predictions).float().mean().item()
            )
            confidence_change = (
                (modified_confidences - original_confidences).mean().item()
            )

            # Compute effect size
            pooled_std = torch.sqrt(
                (torch.var(original_outputs) + torch.var(modified_outputs)) / 2
            )
            if pooled_std > 1e-8:
                effect_size = (
                    torch.mean(modified_outputs - original_outputs).item()
                    / pooled_std.item()
                )
            else:
                effect_size = 0.0

            return ImpactMetrics(
                output_distance=output_distance,
                output_correlation=output_correlation,
                prediction_change_rate=prediction_change_rate,
                confidence_change=confidence_change,
                effect_size=effect_size,
            )

        finally:
            # Restore original feature
            self.manager.modify_feature(
                layer_name,
                feature_index,
                original_feature.vector,
                track_intervention=False,
            )

    def sensitivity_analysis(
        self,
        layer_name: str,
        parameter_type: str,  # 'prototype' or 'feature'
        parameter_index: int,
        test_inputs: torch.Tensor,
        perturbation_scales: List[float] = None,
    ) -> Dict[float, ImpactMetrics]:
        """
        Perform sensitivity analysis by applying different scales of perturbation.

        Args:
            layer_name: Name of layer to analyze
            parameter_type: 'prototype' or 'feature'
            parameter_index: Index of parameter to perturb
            test_inputs: Input data for evaluation
            perturbation_scales: List of perturbation scales to test

        Returns:
            Dictionary mapping perturbation scales to impact metrics
        """
        if perturbation_scales is None:
            perturbation_scales = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

        results = {}

        # Get original parameter
        if parameter_type == "prototype":
            original_param = self.manager.get_prototype(layer_name, parameter_index)
            assess_func = self.assess_prototype_impact
        elif parameter_type == "feature":
            original_param = self.manager.get_feature(layer_name, parameter_index)
            assess_func = self.assess_feature_impact
        else:
            raise ValueError(f"Unknown parameter_type: {parameter_type}")

        original_vector = original_param.vector

        for scale in perturbation_scales:
            # Create random perturbation
            perturbation = torch.randn_like(original_vector) * scale
            perturbed_vector = original_vector + perturbation

            # Assess impact
            impact = assess_func(
                layer_name, parameter_index, perturbed_vector, test_inputs
            )

            results[scale] = impact

        return results


class CounterfactualAnalyzer:
    """
    Perform counterfactual analysis on TNN models.

    Generates counterfactual examples by finding minimal interventions
    that change model predictions.
    """

    def __init__(self, intervention_manager: InterventionManager):
        """
        Initialize CounterfactualAnalyzer.

        Args:
            intervention_manager: InterventionManager instance to use
        """
        self.manager = intervention_manager
        self.model = intervention_manager.model

    def find_prototype_counterfactuals(
        self,
        input_sample: torch.Tensor,
        target_class: int,
        layer_name: str,
        max_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> List[CounterfactualResult]:
        """
        Find counterfactual examples by modifying prototypes.

        Args:
            input_sample: Input sample to generate counterfactuals for
            target_class: Desired output class
            layer_name: Layer to modify prototypes in
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for optimization

        Returns:
            List of successful counterfactual results
        """
        if layer_name not in self.manager.layer_names:
            raise ValueError(f"Layer '{layer_name}' not found")

        layer = self.manager._tnn_layers[layer_name]
        if not hasattr(layer, "prototypes"):
            raise ValueError(f"Layer '{layer_name}' has no prototypes")

        # Get original prediction
        self.model.eval()
        with torch.no_grad():
            original_output = self.model(input_sample.unsqueeze(0))
            original_prediction = torch.argmax(original_output, dim=1).item()

        if original_prediction == target_class:
            # Already the desired class
            return []

        counterfactuals = []

        # Try modifying each prototype
        for proto_idx in range(layer.prototypes.shape[0]):
            original_prototype = self.manager.get_prototype(layer_name, proto_idx)

            # Create a copy of the prototype for optimization
            modified_prototype = original_prototype.vector.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([modified_prototype], lr=learning_rate)

            for iteration in range(max_iterations):
                optimizer.zero_grad()

                # Temporarily set the prototype
                with torch.no_grad():
                    layer.prototypes[proto_idx] = modified_prototype

                # Forward pass
                output = self.model(input_sample.unsqueeze(0))

                # Loss: want to maximize probability of target class
                loss = -F.log_softmax(output, dim=1)[0, target_class]

                loss.backward()
                optimizer.step()

                # Check if we achieved the target
                with torch.no_grad():
                    prediction = torch.argmax(output, dim=1).item()
                    if prediction == target_class:
                        # Success! Create counterfactual result
                        result = CounterfactualResult(
                            original_input=input_sample.clone(),
                            original_output=original_output.clone(),
                            original_prediction=original_prediction,
                            modified_input=input_sample.clone(),  # Input didn't change
                            modified_output=output.clone(),
                            modified_prediction=prediction,
                            intervention_description=(
                                f"Modified prototype {proto_idx} in layer {layer_name}"
                            ),
                            success=True,
                            input_perturbation_norm=0.0,  # No input perturbation
                            output_change_norm=torch.norm(
                                output - original_output
                            ).item(),
                            confidence_change=F.softmax(output, dim=1).max().item()
                            - F.softmax(original_output, dim=1).max().item(),
                        )
                        counterfactuals.append(result)
                        break

            # Restore original prototype
            self.manager.modify_prototype(
                layer_name,
                proto_idx,
                original_prototype.vector,
                track_intervention=False,
            )

        return counterfactuals

    def find_feature_counterfactuals(
        self,
        input_sample: torch.Tensor,
        target_class: int,
        layer_name: str,
        max_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> List[CounterfactualResult]:
        """
        Find counterfactual examples by modifying features.

        Args:
            input_sample: Input sample to generate counterfactuals for
            target_class: Desired output class
            layer_name: Layer to modify features in
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for optimization

        Returns:
            List of successful counterfactual results
        """
        if layer_name not in self.manager.layer_names:
            raise ValueError(f"Layer '{layer_name}' not found")

        layer = self.manager._tnn_layers[layer_name]
        if not hasattr(layer, "feature_bank"):
            raise ValueError(f"Layer '{layer_name}' has no feature bank")

        # Get original prediction
        self.model.eval()
        with torch.no_grad():
            original_output = self.model(input_sample.unsqueeze(0))
            original_prediction = torch.argmax(original_output, dim=1).item()

        if original_prediction == target_class:
            return []

        counterfactuals = []

        # Try modifying each feature
        for feat_idx in range(layer.feature_bank.shape[0]):
            original_feature = self.manager.get_feature(layer_name, feat_idx)

            # Create a copy of the feature for optimization
            modified_feature = original_feature.vector.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([modified_feature], lr=learning_rate)

            for iteration in range(max_iterations):
                optimizer.zero_grad()

                # Temporarily set the feature
                with torch.no_grad():
                    layer.feature_bank[feat_idx] = modified_feature

                # Forward pass
                output = self.model(input_sample.unsqueeze(0))

                # Loss: want to maximize probability of target class
                loss = -F.log_softmax(output, dim=1)[0, target_class]

                loss.backward()
                optimizer.step()

                # Check if we achieved the target
                with torch.no_grad():
                    prediction = torch.argmax(output, dim=1).item()
                    if prediction == target_class:
                        # Success!
                        result = CounterfactualResult(
                            original_input=input_sample.clone(),
                            original_output=original_output.clone(),
                            original_prediction=original_prediction,
                            modified_input=input_sample.clone(),
                            modified_output=output.clone(),
                            modified_prediction=prediction,
                            intervention_description=(
                                f"Modified feature {feat_idx} in layer {layer_name}"
                            ),
                            success=True,
                            input_perturbation_norm=0.0,
                            output_change_norm=torch.norm(
                                output - original_output
                            ).item(),
                            confidence_change=F.softmax(output, dim=1).max().item()
                            - F.softmax(original_output, dim=1).max().item(),
                        )
                        counterfactuals.append(result)
                        break

            # Restore original feature
            self.manager.modify_feature(
                layer_name, feat_idx, original_feature.vector, track_intervention=False
            )

        return counterfactuals

    def analyze_decision_boundary(
        self, input_samples: torch.Tensor, layer_name: str, num_perturbations: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze how the decision boundary changes with interventions.

        Args:
            input_samples: Set of input samples near decision boundary
            layer_name: Layer to analyze
            num_perturbations: Number of random perturbations to test

        Returns:
            Dictionary with boundary analysis results
        """
        results = {
            "layer_name": layer_name,
            "num_samples": len(input_samples),
            "boundary_stability": {},
            "intervention_effects": [],
        }

        # Get original predictions
        self.model.eval()
        with torch.no_grad():
            original_outputs = self.model(input_samples)
            original_predictions = torch.argmax(original_outputs, dim=1)

        layer = self.manager._tnn_layers[layer_name]

        # Test prototype perturbations
        if hasattr(layer, "prototypes"):
            for proto_idx in range(
                min(layer.prototypes.shape[0], 3)
            ):  # Limit for efficiency
                original_proto = self.manager.get_prototype(layer_name, proto_idx)
                boundary_changes = 0

                for _ in range(num_perturbations):
                    # Random perturbation
                    perturbation = torch.randn_like(original_proto.vector) * 0.1
                    perturbed_proto = original_proto.vector + perturbation

                    # Apply intervention
                    self.manager.modify_prototype(
                        layer_name, proto_idx, perturbed_proto, track_intervention=False
                    )

                    # Check predictions
                    with torch.no_grad():
                        new_outputs = self.model(input_samples)
                        new_predictions = torch.argmax(new_outputs, dim=1)

                    # Count boundary crossings
                    boundary_changes += (
                        (original_predictions != new_predictions).sum().item()
                    )

                    # Restore original
                    self.manager.modify_prototype(
                        layer_name,
                        proto_idx,
                        original_proto.vector,
                        track_intervention=False,
                    )

                stability = 1.0 - (
                    boundary_changes / (num_perturbations * len(input_samples))
                )
                results["boundary_stability"][f"prototype_{proto_idx}"] = stability

        return results
