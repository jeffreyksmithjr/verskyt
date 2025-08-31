"""
Intervention API Demo for Tversky Neural Networks.

Demonstrates the complete intervention workflow:
1. Model inspection and parameter discovery
2. Prototype and feature modification
3. Impact assessment
4. Counterfactual analysis
5. Feature grounding to semantic concepts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import TNN components
from verskyt import TverskyProjectionLayer
from verskyt.interventions import (
    CounterfactualAnalyzer,
    FeatureGrounder,
    ImpactAssessment,
    InterventionManager,
)


class SimpleTNNClassifier(nn.Module):
    """
    Simple TNN classifier for demonstration.

    Classifies 2D points into two categories using a single TverskyProjectionLayer.
    """

    def __init__(self, num_features=8):
        super().__init__()
        self.tnn_layer = TverskyProjectionLayer(
            in_features=2,
            num_prototypes=2,  # 2 classes
            num_features=num_features,
            alpha=0.5,
            beta=0.5,
            learnable_ab=True,
            intersection_reduction="product",
            difference_reduction="substractmatch",
        )

    def forward(self, x):
        return self.tnn_layer(x)


def create_sample_data():
    """Create sample data for demonstration."""
    # Create XOR-like dataset
    torch.manual_seed(42)

    # Training data
    train_inputs = torch.tensor(
        [
            [0.0, 0.0],  # Class 0
            [0.0, 1.0],  # Class 1
            [1.0, 0.0],  # Class 1
            [1.0, 1.0],  # Class 0
            [0.1, 0.1],  # Class 0 (with noise)
            [0.1, 0.9],  # Class 1
            [0.9, 0.1],  # Class 1
            [0.9, 0.9],  # Class 0
        ]
    )

    train_targets = torch.tensor([0, 1, 1, 0, 0, 1, 1, 0])

    # Test data
    test_inputs = torch.tensor(
        [
            [0.2, 0.2],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.8, 0.8],
        ]
    )

    test_targets = torch.tensor([0, 1, 1, 0])

    return (train_inputs, train_targets), (test_inputs, test_targets)


def train_model(model, train_data, epochs=500):
    """Train the TNN model."""
    train_inputs, train_targets = train_data

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(train_inputs)
        loss = F.cross_entropy(outputs, train_targets)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                pred = torch.argmax(outputs, dim=1)
                acc = (pred == train_targets).float().mean().item()
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {acc:.2%}")

    print("Training completed!")
    return model


def demo_model_inspection(manager):
    """Demonstrate model inspection capabilities."""
    print("=== MODEL INSPECTION DEMO ===")

    # Print model summary
    print(manager.summary())
    print()

    # List all prototypes
    prototypes = manager.list_prototypes()
    print(f"Found {len(prototypes)} prototypes:")
    for proto in prototypes:
        print(
            f"  {proto.layer_name}.prototype[{proto.prototype_index}]: "
            f"shape={proto.shape}, norm={proto.norm:.3f}"
        )
    print()

    # List all features
    features = manager.list_features()
    print(f"Found {len(features)} features:")
    for feat in features[:4]:  # Show first 4 only
        print(
            f"  {feat.layer_name}.feature[{feat.feature_index}]: "
            f"shape={feat.shape}, norm={feat.norm:.3f}"
        )
    print(f"  ... and {len(features)-4} more features")
    print()


def demo_prototype_modification(manager, test_data):
    """Demonstrate prototype modification and impact assessment."""
    print("=== PROTOTYPE MODIFICATION DEMO ===")

    test_inputs, test_targets = test_data

    # Get original outputs for comparison
    model = manager.model
    model.eval()
    with torch.no_grad():
        original_outputs = model(test_inputs)
        original_predictions = torch.argmax(original_outputs, dim=1)

    print(f"Original predictions: {original_predictions.tolist()}")
    print(f"True targets:         {test_targets.tolist()}")

    # Get original prototype
    original_proto = manager.get_prototype("tnn_layer", 0)
    print(f"Original prototype 0: {original_proto.vector}")

    # Create a modified prototype (flip signs to see dramatic effect)
    modified_vector = -original_proto.vector

    # Apply modification
    manager.modify_prototype("tnn_layer", 0, modified_vector)
    print(f"Modified prototype 0: {modified_vector}")

    # Check new outputs
    with torch.no_grad():
        new_outputs = model(test_inputs)
        new_predictions = torch.argmax(new_outputs, dim=1)

    print(f"New predictions:      {new_predictions.tolist()}")

    # Show intervention history
    history = manager.get_intervention_history()
    print(f"Interventions performed: {len(history)}")

    # Reset to original
    manager.reset_to_original()
    print("Reset to original state")
    print()


def demo_impact_assessment(manager, test_data):
    """Demonstrate impact assessment capabilities."""
    print("=== IMPACT ASSESSMENT DEMO ===")

    test_inputs, test_targets = test_data

    # Create impact assessor
    assessor = ImpactAssessment(manager)

    # Get original prototype
    original_proto = manager.get_prototype("tnn_layer", 0)

    # Create various modifications to test
    modifications = {
        "small_noise": original_proto.vector
        + torch.randn_like(original_proto.vector) * 0.1,
        "medium_noise": original_proto.vector
        + torch.randn_like(original_proto.vector) * 0.5,
        "negated": -original_proto.vector,
        "zero": torch.zeros_like(original_proto.vector),
    }

    print("Impact assessment for different prototype modifications:")
    for mod_name, mod_vector in modifications.items():
        impact = assessor.assess_prototype_impact(
            "tnn_layer", 0, mod_vector, test_inputs
        )

        print(
            f"  {mod_name:12s}: "
            f"distance={impact.output_distance:.3f}, "
            f"correlation={impact.output_correlation:.3f}, "
            f"pred_change={impact.prediction_change_rate:.1%}, "
            f"effect_size={impact.effect_size:.3f}"
        )

    # Sensitivity analysis
    print("\nSensitivity analysis:")
    sensitivity = assessor.sensitivity_analysis(
        "tnn_layer", "prototype", 0, test_inputs[:2], [0.1, 0.5, 1.0]
    )

    for scale, impact in sensitivity.items():
        change_rate = impact.prediction_change_rate
        print(f"  Scale {scale:3.1f}: prediction change rate = {change_rate:.1%}")

    print()


def demo_counterfactual_analysis(manager, test_data):
    """Demonstrate counterfactual analysis."""
    print("=== COUNTERFACTUAL ANALYSIS DEMO ===")

    test_inputs, test_targets = test_data

    # Create counterfactual analyzer
    cf_analyzer = CounterfactualAnalyzer(manager)

    # Try to flip the prediction for the first test sample
    sample = test_inputs[0]
    model = manager.model

    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_pred = torch.argmax(original_output, dim=1).item()

    target_class = 1 - original_pred  # Flip the class

    print(f"Sample: {sample}")
    print(f"Original prediction: {original_pred}")
    print(f"Target class: {target_class}")

    # Find counterfactuals by modifying prototypes
    counterfactuals = cf_analyzer.find_prototype_counterfactuals(
        sample, target_class, "tnn_layer", max_iterations=50
    )

    print(f"Found {len(counterfactuals)} counterfactual explanations")

    for i, cf in enumerate(counterfactuals[:2]):  # Show first 2
        print(f"  Counterfactual {i+1}:")
        print(f"    Method: {cf.intervention_description}")
        print(f"    Success: {cf.success}")
        print(f"    Output change: {cf.output_change_norm:.3f}")
        print(f"    Confidence change: {cf.confidence_change:.3f}")

    print()


def demo_feature_grounding(manager, train_data):
    """Demonstrate feature grounding to semantic concepts."""
    print("=== FEATURE GROUNDING DEMO ===")

    train_inputs, train_targets = train_data

    # Create feature grounder
    grounder = FeatureGrounder(manager)

    # Define some semantic concepts for our XOR-like problem
    grounder.add_concept(
        "bottom_left",
        "Points in the bottom-left quadrant (low x, low y)",
        examples=["Represents the (0,0) region"],
    )

    grounder.add_concept(
        "top_right",
        "Points in the top-right quadrant (high x, high y)",
        examples=["Represents the (1,1) region"],
    )

    grounder.add_concept(
        "off_diagonal",
        "Points on the off-diagonal (bottom-right and top-left)",
        examples=["Represents the (0,1) and (1,0) regions"],
    )

    # Manually ground some features to concepts
    grounder.ground_feature_manually("tnn_layer", 0, "bottom_left", confidence=0.9)
    grounder.ground_feature_manually("tnn_layer", 1, "top_right", confidence=0.8)
    grounder.ground_prototype_manually("tnn_layer", 0, "bottom_left", confidence=0.85)
    grounder.ground_prototype_manually("tnn_layer", 1, "off_diagonal", confidence=0.7)

    # Show groundings
    groundings = grounder.list_groundings()
    print(f"Created {len(groundings)} semantic groundings:")

    for grounding in groundings:
        param_str = (
            f"{grounding.layer_name}.{grounding.parameter_type}"
            f"[{grounding.parameter_index}]"
        )
        concept_str = (
            f"-> '{grounding.concept_name}' "
            f"(confidence: {grounding.confidence:.2f})"
        )
        print(f"  {param_str} {concept_str}")

    print()

    # Generate explanations
    print("Parameter explanations:")
    for i in range(2):  # First 2 features
        explanation = grounder.explain_parameter("tnn_layer", "feature", i)
        print(f"  {explanation}")

    for i in range(2):  # Both prototypes
        explanation = grounder.explain_parameter("tnn_layer", "prototype", i)
        print(f"  {explanation}")

    print()

    # Generate full model explanation
    print("Full model explanation:")
    print(grounder.generate_model_explanation())
    print()


def demo_advanced_grounding(manager, train_data):
    """Demonstrate automatic feature grounding based on activations."""
    print("=== AUTOMATIC FEATURE GROUNDING DEMO ===")

    train_inputs, train_targets = train_data

    grounder = FeatureGrounder(manager)

    # Create concept-specific input sets
    concept_inputs = {
        "class_0": train_inputs[train_targets == 0],
        "class_1": train_inputs[train_targets == 1],
    }

    print("Attempting automatic grounding based on activation patterns...")

    # Ground features based on activation patterns
    auto_groundings = grounder.ground_features_by_activation(
        "tnn_layer", concept_inputs, confidence_threshold=0.6
    )

    print(f"Automatically grounded {len(auto_groundings)} features:")
    for grounding in auto_groundings:
        print(
            f"  Feature {grounding.parameter_index} -> '{grounding.concept_name}' "
            f"(confidence: {grounding.confidence:.3f})"
        )

    print()


def main():
    """Run the complete intervention demo."""
    print("Tversky Neural Network Intervention API Demo")
    print("=" * 50)

    # Create and train model
    print("Creating and training TNN model...")
    model = SimpleTNNClassifier(num_features=8)

    # Create sample data
    train_data, test_data = create_sample_data()

    # Train the model
    model = train_model(model, train_data)

    # Create intervention manager
    manager = InterventionManager(model, "XOR_TNN_Demo")

    # Run all demonstrations
    demo_model_inspection(manager)
    demo_prototype_modification(manager, test_data)
    demo_impact_assessment(manager, test_data)
    demo_counterfactual_analysis(manager, test_data)
    demo_feature_grounding(manager, train_data)
    demo_advanced_grounding(manager, train_data)

    print("Demo completed! The intervention API provides:")
    print("✓ Model inspection and parameter discovery")
    print("✓ Prototype and feature modification with tracking")
    print("✓ Impact assessment and sensitivity analysis")
    print("✓ Counterfactual explanation generation")
    print("✓ Semantic grounding of parameters to concepts")
    print("✓ Automatic grounding based on activation patterns")


if __name__ == "__main__":
    main()
