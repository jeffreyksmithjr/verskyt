# Tversky Neural Networks (TNNs) Implementation Requirements

## Paper Reference
Based on: "Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity" (Doumbouya et al., 2025)

## 1. Core Mathematical Formulations

### 1.1 Tversky Similarity Function

The fundamental equation for Tversky similarity between objects a and b:

```
S(a, b) = θ * f(A ∩ B) - α * f(A - B) - β * f(B - A)
```

Where:
- `A ∩ B`: Common features between objects a and b
- `A - B`: Features present in a but not in b
- `B - A`: Features present in b but not in a
- `θ, α, β`: Learnable scalar parameters (Tversky's contrast model weights)
- `f()`: Feature measurement function

### 1.2 Dual Object Representation

Objects have dual representation:
1. **Vector form**: `x ∈ ℝᵈ`
2. **Set form**: `X = {fₖ ∈ Ω | x · fₖ > 0}` (features with positive dot product)

Where `Ω` is the learnable finite universe of feature vectors `fₖ ∈ ℝᵈ`.

### 1.3 Feature Salience

Salience of features in object A:
```
f(A) = Σₖ₌₁|Ω| (a · fₖ) * 𝟙[a · fₖ > 0]
```

### 1.4 Feature Set Intersections

Common features measure with aggregation function Ψ:
```
f(A ∩ B) = Σₖ₌₁|Ω| Ψ(a · fₖ, b · fₖ) * 𝟙[a · fₖ > 0 ∧ b · fₖ > 0]
```

**Intersection Reduction Methods** (Ψ function options):
- `min`: min(a·fₖ, b·fₖ)
- `max`: max(a·fₖ, b·fₖ)
- `product`: (a·fₖ) * (b·fₖ)
- `mean`: (a·fₖ + b·fₖ) / 2
- `gmean`: √((a·fₖ) * (b·fₖ))
- `softmin`: soft minimum function

**Best performing**: `product` (based on XOR experiments)

### 1.5 Feature Set Differences

Two formulations for distinctive features:

**Ignore Match** (recommended):
```
f(A - B) = Σₖ₌₁|Ω| (a · fₖ) * 𝟙[a · fₖ > 0 ∧ b · fₖ ≤ 0]
```

**Subtract Match**:
```
f(A - B) = Σₖ₌₁|Ω| (a · fₖ - b · fₖ) * 𝟙[a · fₖ > 0 ∧ b · fₖ > 0 ∧ a · fₖ > b · fₖ]
```

**Best performing**: `substractmatch` for XOR, but `ignorematch` is simpler and recommended generally.

## 2. Neural Network Modules

### 2.1 Tversky Similarity Layer

**Purpose**: Compute similarity between two objects
**Input**: Two vectors a, b ∈ ℝᵈ
**Output**: Scalar similarity value
**Parameters**:
- Feature bank Ω (shape: [num_features, d])
- Scalar parameters α, β, θ

**Function**:
```python
def tversky_similarity(a, b, features, alpha, beta, theta):
    return theta * intersection_measure(a, b, features) - alpha * difference_measure(a, b, features) - beta * difference_measure(b, a, features)
```

### 2.2 Tversky Projection Layer

**Purpose**: Replace linear/fully-connected layers
**Input**: Vector a ∈ ℝᵈ
**Output**: Vector ∈ ℝᵖ (similarities to p prototypes)
**Parameters**:
- Feature bank Ω (shape: [num_features, d])
- Prototype bank Π (shape: [num_prototypes, d])
- Scalar parameters α, β, θ

**Function**:
```python
def tversky_projection(a, prototypes, features, alpha, beta, theta):
    similarities = []
    for prototype in prototypes:
        sim = tversky_similarity(a, prototype, features, alpha, beta, theta)
        similarities.append(sim)
    return torch.stack(similarities)
```

## 3. Implementation Requirements

### 3.1 Shape Conventions

- **Objects/inputs**: [batch_size, in_features]
- **Prototypes**: [num_prototypes, in_features]
- **Features**: [num_features, in_features]
- **Similarity outputs**: Must be in range [0, 1] (normalized)

### 3.2 Differentiability Requirements

All operations must be differentiable for gradient-based training:
- Use continuous approximations for discrete set operations
- Indicator functions 𝟙[·] should be implemented as smooth approximations (e.g., sigmoid with high temperature)
- All aggregation functions (Ψ) must be differentiable

### 3.3 Parameter Initialization

**Based on XOR experiments, best practices**:
- **Prototypes and Features**: Uniform distribution initialization (highest convergence probability)
- **α, β, θ parameters**: Small positive values, with θ > α, β typically
- **Avoid**: Normal or orthogonal initialization (lower convergence rates)
- **Normalization**: Do NOT normalize prototype/object vectors (decreases convergence)

### 3.4 Feature Bank Sizing

- **Minimum**: Can work with as few as 1 feature (proven on XOR)
- **Optimal**: 16 features showed best convergence in XOR experiments
- **Scaling**: Feature count doesn't need to scale with output dimensionality
- **Overparameterization**: Very large feature banks may lead to feature clustering

## 4. Architecture Integration Patterns

### 4.1 Drop-in Replacement for Linear Layers

Tversky Projection layers can directly replace:
- Final classification layers in CNNs (ResNet-50 experiments)
- Language modeling heads in transformers (GPT-2 experiments)
- Intermediate projection layers in attention blocks

### 4.2 Parameter Sharing Strategies

**Feature Bank Sharing**: Multiple layers can share the same feature bank Ω when semantically compatible:
- Language model layers processing tokens can share token feature banks
- Attention layers can share attention feature banks

**Prototype Sharing**:
- Language modeling heads can use token embeddings as prototypes (weight tying)
- Similar to existing weight tying in transformer language models

**Benefits**: Dramatic parameter reduction (up to 34.8% shown in GPT-2 experiments)

### 4.3 Multi-layer Stacking

- Can stack multiple Tversky projection layers
- Each layer can have independent or shared feature/prototype banks
- Allows for hierarchical feature learning

## 5. Training Considerations

### 5.1 Hyperparameter Sensitivity

**Critical hyperparameters** (from XOR analysis):
- Intersection reduction method: `product` recommended
- Difference reduction method: `substractmatch` or `ignorematch`
- Initialization distribution: `uniform` for both features and prototypes
- Normalization: Avoid normalizing vectors during similarity computation

### 5.2 Convergence Issues

**Known failure modes**:
- Some random initializations may not converge
- Geometric mean (gmean) aggregation causes numerical instability
- Normalization reduces convergence probability

**Mitigation strategies**:
- Multiple random restarts
- Careful hyperparameter selection
- Monitor gradient flow during training

### 5.3 Training Efficiency

**Computational considerations**:
- Feature banks add parameters but enable parameter sharing
- Forward pass requires computing similarities with all prototypes
- Can be more parameter-efficient than linear layers when using sharing

## 6. Performance Expectations

### 6.1 Capabilities

**Non-linearity**: Single Tversky projection layer can model XOR function (impossible for single linear layer)

**Performance improvements demonstrated**:
- ResNet-50 + Tversky projection: Up to 24.7% accuracy improvement on NABirds
- GPT-2 + Tversky projections: 7.5% perplexity reduction with 34.8% fewer parameters

### 6.2 Interpretability Benefits

**Prototype interpretability**: Learned prototypes are more recognizable than linear layer weights
**Feature explanations**: Can explain similarity in terms of common and distinctive features
**Salience analysis**: Can compute and analyze feature salience using Equation from 1.3

## 7. Evaluation Criteria

### 7.1 Functional Tests

**XOR Test**: Single Tversky projection should learn XOR function with various feature counts
**Gradient Flow**: All operations must have well-defined gradients
**Shape Consistency**: Output shapes must match expected dimensions

### 7.2 Performance Benchmarks

**Drop-in replacement**: Should match or exceed linear layer performance
**Parameter efficiency**: Should achieve similar performance with fewer parameters when using sharing
**Convergence**: Should converge reliably with proper hyperparameters

### 7.3 Interpretability Measures

**Prototype recognizability**: Visual inspection of learned prototypes should show interpretable patterns
**Feature salience**: Salience rankings should align with intuitive feature importance
**Asymmetry**: Should demonstrate asymmetric similarity (S(a,b) ≠ S(b,a))

## 8. Implementation Architecture

### 8.1 Core Components Required

```python
# Core similarity computation
class TverskySimilarity(nn.Module):
    def __init__(self, num_features, feature_dim):
        self.features = nn.Parameter(torch.randn(num_features, feature_dim))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.theta = nn.Parameter(torch.tensor(1.0))

    def forward(self, a, b):
        # Implement similarity computation
        pass

# Projection layer
class TverskyProjectionLayer(nn.Module):
    def __init__(self, in_features, num_prototypes, num_features):
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, in_features))
        self.similarity = TverskySimilarity(num_features, in_features)

    def forward(self, x):
        # Compute similarities to all prototypes
        pass
```

### 8.2 Utility Functions

**Required helper functions**:
- Feature membership computation
- Intersection/difference measures with various reduction methods
- Salience computation
- Initialization utilities
- Visualization helpers for data-domain parameter visualization

## 9. Validation and Testing

### 9.1 Unit Tests

- Test each aggregation function (min, max, product, mean, etc.)
- Test gradient computation for all operations
- Test shape handling for batched inputs
- Test parameter sharing mechanisms

### 9.2 Integration Tests

- XOR learning with single layer
- Drop-in replacement in existing architectures
- Parameter reduction with sharing
- Convergence with various initializations

### 9.3 Performance Tests

- Compare against linear baselines on standard benchmarks
- Measure parameter efficiency
- Verify interpretability claims through visualization

## 10. Known Limitations and Considerations

### 10.1 Computational Overhead

- Additional parameters from feature banks
- More complex forward pass computation
- Data-domain parameter visualization increases parameter count

### 10.2 Training Sensitivity

- More sensitive to initialization than linear layers
- Some hyperparameter combinations may not converge
- Requires careful tuning of intersection/difference reduction methods

### 10.3 Implementation Complexity

- More complex than standard linear layers
- Requires careful implementation of differentiable set operations
- Parameter sharing requires architectural consideration

## 11. Future Extensions

### 11.1 Attention Mechanisms

Paper mentions Tversky attention layers that don't require query/key mechanism due to built-in asymmetry.

### 11.2 Multi-head Variants

Extension to multi-head attention-style architectures with multiple feature/prototype banks.

### 11.3 Architectural Innovations

- Full Tversky networks (all layers use Tversky similarity)
- Hybrid architectures mixing linear and Tversky layers
- Domain-specific feature bank designs

---

This document provides complete requirements for implementing Tversky Neural Networks based on the original paper. All mathematical formulations, architectural patterns, and empirical findings should be implemented as specified to achieve the reported performance improvements and interpretability benefits.
