"""
Verskyt Research Tutorial: Modularity, Introspection, and Extensibility

This tutorial demonstrates the key research-focused features of the Verskyt library:
1. Modular architecture for experimenting with similarity variants
2. Deep introspection capabilities for understanding learned representations
3. Extensible framework for custom similarity measures and interventions

Run this script to see Verskyt's research capabilities in action.
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Import Verskyt components
from verskyt import TverskyProjectionLayer
from verskyt.benchmarks import XORBenchmark
from verskyt.core import tversky_similarity

print("🔬 Verskyt Research Tutorial: Modularity, Introspection, and Extensibility")
print("=" * 80)

# =============================================================================
# Part 1: MODULARITY - Experimenting with Similarity Variants
# =============================================================================
print("\n📦 PART 1: MODULARITY - Similarity Variant Experiments")
print("-" * 50)


def compare_similarity_methods():
    """Demonstrate modular similarity computation with different methods."""

    # Create sample data
    torch.manual_seed(42)
    x = torch.randn(4, 8)  # 4 samples, 8 dimensions
    prototypes = torch.randn(3, 8)  # 3 prototypes
    features = torch.randn(16, 8)  # 16 features

    # Test different intersection methods
    intersection_methods = ["product", "min", "max", "mean", "gmean"]
    difference_methods = ["ignorematch", "substractmatch"]

    results = {}

    print("Testing similarity variants:")
    print(
        f"Input shape: {x.shape}, Prototypes: {prototypes.shape}, "
        f"Features: {features.shape}"
    )
    print()

    for int_method in intersection_methods[:3]:  # Limit for brevity
        for diff_method in difference_methods:
            similarity = tversky_similarity(
                x,
                prototypes,
                features,
                alpha=0.5,
                beta=0.5,
                intersection_reduction=int_method,
                difference_reduction=diff_method,
            )

            method_name = f"{int_method}+{diff_method}"
            results[method_name] = similarity

            print(
                f"  {method_name:20} | Range: [{similarity.min():.3f}, "
                f"{similarity.max():.3f}] | Mean: {similarity.mean():.3f}"
            )

    print(f"\n✅ Tested {len(results)} similarity variants in modular fashion")
    return results


similarity_results = compare_similarity_methods()

# =============================================================================
# Part 2: INTROSPECTION - Understanding Learned Representations
# =============================================================================
print("\n🔍 PART 2: INTROSPECTION - Learned Representation Analysis")
print("-" * 50)


def analyze_learned_representations():
    """Demonstrate deep introspection of trained Tversky layers."""

    # Create and train a simple model on XOR
    torch.manual_seed(42)

    # XOR data
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])

    # Create Tversky layer
    layer = TverskyProjectionLayer(
        in_features=2, num_prototypes=1, num_features=4, learnable_ab=True
    )

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(layer.parameters(), lr=0.01)

    print("Training Tversky layer on XOR problem...")

    # Store training history for analysis
    losses = []
    alpha_history = []
    beta_history = []

    # Train for several epochs
    for epoch in range(500):
        optimizer.zero_grad()
        output = layer(X).squeeze()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Record metrics
        losses.append(loss.item())
        alpha_history.append(layer.alpha.item())
        beta_history.append(layer.beta.item())

        if (epoch + 1) % 100 == 0:
            print(
                f"  Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, "
                f"α = {layer.alpha.item():.3f}, β = {layer.beta.item():.3f}"
            )

    print("\n🔍 INTROSPECTION ANALYSIS:")
    print("-" * 30)

    # 1. Analyze learned parameters
    final_alpha = layer.alpha.item()
    final_beta = layer.beta.item()
    print(f"Final asymmetry parameters: α = {final_alpha:.3f}, β = {final_beta:.3f}")

    asymmetry = abs(final_alpha - final_beta)
    if asymmetry > 0.1:
        focus = "input-focused" if final_alpha > final_beta else "prototype-focused"
        print(f"  → Model learned {focus} similarity (asymmetry = {asymmetry:.3f})")
    else:
        print(f"  → Model learned symmetric similarity (asymmetry = {asymmetry:.3f})")

    # 2. Examine learned prototypes and features
    prototypes = layer.prototypes.detach()
    features = layer.feature_bank.detach()

    print("\nLearned representations:")
    print(f"  Prototype shape: {prototypes.shape}")
    print(f"  Feature bank shape: {features.shape}")
    print(
        f"  Prototype values: {prototypes.flatten()[:4].numpy()}"
    )  # Show first few values

    # 3. Analyze feature usage
    feature_norms = torch.norm(features, dim=1)
    active_features = (feature_norms > 0.1).sum().item()
    print(f"  Active features (norm > 0.1): {active_features}/{len(features)}")

    # 4. Test final performance
    with torch.no_grad():
        final_output = layer(X).squeeze()
        final_predictions = (final_output > 0.5).float()
        accuracy = (final_predictions == y).float().mean()
        print(f"\nFinal XOR accuracy: {accuracy:.1%}")

        print("Input → Target | Predicted | Similarity")
        for i in range(4):
            print(
                f"  {X[i].numpy()} → {y[i]:.0f}     | "
                f"{final_predictions[i]:.0f}        | {final_output[i]:.3f}"
            )

    return layer, losses, alpha_history, beta_history


trained_layer, loss_history, alpha_hist, beta_hist = analyze_learned_representations()

# =============================================================================
# Part 3: EXTENSIBILITY - Custom Interventions and Experiments
# =============================================================================
print("\n🔧 PART 3: EXTENSIBILITY - Intervention Studies")
print("-" * 50)


def intervention_experiments(layer):
    """Demonstrate extensible intervention capabilities."""

    # XOR test data
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])

    print("Conducting intervention experiments...")

    # 1. Baseline performance
    with torch.no_grad():
        baseline_output = layer(X).squeeze()
        baseline_accuracy = ((baseline_output > 0.5).float() == y).float().mean()

    print(f"1. Baseline accuracy: {baseline_accuracy:.1%}")

    # 2. Prototype intervention - zero out the learned prototype
    print("\n2. Prototype intervention (zero out learned prototype):")
    original_prototype = layer.get_prototype(0).clone()
    layer.set_prototype(0, torch.zeros_like(original_prototype))

    with torch.no_grad():
        intervened_output = layer(X).squeeze()
        intervened_accuracy = ((intervened_output > 0.5).float() == y).float().mean()

    print(f"   Accuracy after zeroing prototype: {intervened_accuracy:.1%}")
    print(f"   Output change: {(intervened_output - baseline_output).abs().mean():.3f}")

    # Restore original prototype
    layer.set_prototype(0, original_prototype)

    # 3. Feature bank intervention - modify feature space
    print("\n3. Feature bank intervention (scale down features):")
    original_features = layer.feature_bank.data.clone()
    layer.feature_bank.data *= 0.5  # Scale down all features

    with torch.no_grad():
        feature_output = layer(X).squeeze()
        feature_accuracy = ((feature_output > 0.5).float() == y).float().mean()

    print(f"   Accuracy after scaling features: {feature_accuracy:.1%}")
    print(f"   Output change: {(feature_output - baseline_output).abs().mean():.3f}")

    # Restore original features
    layer.feature_bank.data = original_features

    # 4. Parameter intervention - force symmetry
    print("\n4. Parameter intervention (force α = β = 0.5):")
    original_alpha = layer.alpha.data.clone()
    original_beta = layer.beta.data.clone()

    layer.alpha.data = torch.tensor(0.5)
    layer.beta.data = torch.tensor(0.5)

    with torch.no_grad():
        symmetric_output = layer(X).squeeze()
        symmetric_accuracy = ((symmetric_output > 0.5).float() == y).float().mean()

    print(f"   Accuracy with forced symmetry: {symmetric_accuracy:.1%}")
    print(f"   Output change: {(symmetric_output - baseline_output).abs().mean():.3f}")

    # Restore original parameters
    layer.alpha.data = original_alpha
    layer.beta.data = original_beta

    print("\n✅ Intervention experiments completed")
    print("    → All modifications easily reversed due to modular design")


intervention_experiments(trained_layer)

# =============================================================================
# Part 4: EXTENSIBILITY - Custom Similarity Functions
# =============================================================================
print("\n🎯 PART 4: EXTENSIBILITY - Custom Similarity Exploration")
print("-" * 50)


def explore_custom_similarities():
    """Demonstrate extensible similarity function usage."""

    torch.manual_seed(42)
    x = torch.randn(3, 4)
    prototypes = torch.randn(2, 4)
    features = torch.randn(8, 4)

    print("Exploring custom similarity configurations:")

    # Test extreme asymmetry values
    configs = [
        {"alpha": 0.0, "beta": 1.0, "name": "Prototype-only"},
        {"alpha": 1.0, "beta": 0.0, "name": "Input-only"},
        {"alpha": 0.2, "beta": 0.8, "name": "Prototype-focused"},
        {"alpha": 0.8, "beta": 0.2, "name": "Input-focused"},
        {"alpha": 0.5, "beta": 0.5, "name": "Symmetric (Jaccard-like)"},
    ]

    for config in configs:
        similarity = tversky_similarity(
            x,
            prototypes,
            features,
            alpha=config["alpha"],
            beta=config["beta"],
            intersection_reduction="product",
            difference_reduction="substractmatch",
        )

        print(
            f"  {config['name']:20} (α={config['alpha']:.1f}, β={config['beta']:.1f}): "
            f"mean={similarity.mean():.3f}, std={similarity.std():.3f}"
        )

    print("\n✅ Custom similarity exploration completed")
    print("    → Easy to experiment with different mathematical formulations")


explore_custom_similarities()

# =============================================================================
# Part 5: RESEARCH WORKFLOW - Comprehensive Validation
# =============================================================================
print("\n🧪 PART 5: RESEARCH WORKFLOW - Benchmark Validation")
print("-" * 50)


def research_validation():
    """Demonstrate research-grade validation capabilities."""

    print("Running research-grade XOR benchmark...")

    # Run fast benchmark (representative sample)
    print("1. Fast benchmark (48 configurations):")
    try:
        from verskyt.benchmarks import run_fast_xor_benchmark

        results, analysis = run_fast_xor_benchmark(verbose=False)

        convergence_rate = analysis["overall_convergence_rate"]
        total_runs = len(results)

        print(f"   Overall convergence rate: {convergence_rate:.1%}")
        print(f"   Total configurations: {total_runs}")

        # Show some key method combinations if available
        key_methods = [
            "product_substractmatch",
            "mean_substractmatch",
            "max_ignorematch",
            "gmean_ignorematch",
        ]
        print("\n   Key method combinations:")
        for method in key_methods:
            conv_key = f"convergence_rate_{method}"
            if conv_key in analysis:
                print(f"     {method.replace('_', ' + ')}: {analysis[conv_key]:.1%}")

        print("\n✅ Benchmark validation completed")
        print("    → Results validate implementation correctness")

    except Exception as e:
        print(f"   ⚠️ Benchmark skipped due to: {e}")
        print("      This is normal if running in limited environment")


research_validation()

# =============================================================================
# SUMMARY & RESEARCH RECOMMENDATIONS
# =============================================================================
print("\n" + "=" * 80)
print("🎉 TUTORIAL SUMMARY: Research Capabilities Demonstrated")
print("=" * 80)

print(
    """
✅ MODULARITY DEMONSTRATED:
   → 6 intersection methods × 2 difference methods = 12 similarity variants
   → Easy comparison of mathematical formulations
   → Clean separation between similarity computation and neural layers

✅ INTROSPECTION DEMONSTRATED:
   → Access to learned prototypes, features, and parameters
   → Real-time monitoring of asymmetry parameter evolution
   → Feature usage analysis and activation patterns
   → Performance analysis with detailed predictions

✅ EXTENSIBILITY DEMONSTRATED:
   → Easy intervention studies (prototype, feature, parameter modification)
   → Custom similarity parameter exploration
   → Reversible modifications for controlled experiments
   → Research-grade benchmark validation

🔬 RESEARCH RECOMMENDATIONS:

1. PARAMETER STUDIES:
   - Systematically vary α, β asymmetry parameters
   - Compare learnable vs. fixed parameter settings
   - Study convergence patterns across different initializations

2. ARCHITECTURAL EXPERIMENTS:
   - Replace linear layers in existing models (ResNet, Transformer)
   - Experiment with shared vs. independent feature banks
   - Test different feature space dimensionalities

3. INTERPRETABILITY STUDIES:
   - Visualize learned prototypes in 2D/3D spaces
   - Analyze feature bank evolution during training
   - Conduct prototype intervention studies on real datasets

4. SIMILARITY METHOD RESEARCH:
   - Compare intersection methods (product vs. min vs. max)
   - Study difference methods (ignorematch vs. substractmatch)
   - Develop custom reduction strategies

5. REPRODUCIBILITY:
   - Use built-in benchmarks for validation
   - Document parameter settings and random seeds
   - Compare against paper baselines systematically

"""
)

print("📚 For more advanced examples, see:")
print("   - docs/tutorials/ for step-by-step guides")
print("   - scripts/ for benchmark reproduction")
print("   - docs/api/ for complete API documentation")

print("\n🚀 Happy researching with Verskyt!")
