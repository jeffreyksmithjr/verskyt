# Verskyt Implementation Plan
## A Testable and Verifiably Accurate Implementation of Tversky Neural Networks

### Core Design Principles
1. **Modularity**: Clean separation between mathematical operations, layers, and tools
2. **Introspection**: Full access to internal states, features, and prototypes
3. **Extensibility**: Easy integration with existing PyTorch architectures
4. **Verifiability**: Every component tested against paper specifications

### Phase 1: Core Mathematical Foundation (Week 1)

#### 1.1 Core Similarity Module (`verskyt/core/similarity.py`)
**Critical Implementation Details from Paper:**
- Dual representation: Objects as vectors AND as sets
- Feature membership: `x·fₖ > 0` determines feature presence
- Intersection measures: product, min, max, mean, gmean, softmin
- Difference measures: ignorematch, substractmatch
- Salience calculation: Equation 2 from paper
- Asymmetry parameters: α, β, θ

**Testability Requirements:**
- Unit tests for each reduction method
- Numerical stability tests with edge cases (zero vectors, negative values)
- Gradient flow verification
- Comparison with paper's mathematical formulations

#### 1.2 Differentiability Validation
- Ensure all operations are differentiable
- Test gradient computation through similarity function
- Verify backpropagation correctness

### Phase 2: Neural Network Layers (Week 1-2)

#### 2.1 TverskySimilarityLayer
- Implements Equation 6 from paper
- Learnable parameters: Ω (feature bank), α, β, θ
- Output: scalar similarity between two objects

#### 2.2 TverskyProjectionLayer
- Implements Equation 7 from paper
- Replaces traditional linear layers
- Parameters: prototypes (Π), feature bank (Ω), α, β
- Non-linear decision boundaries without activation functions

**Validation Against Paper:**
- XOR test: Single layer must solve XOR (Section 3.1)
- Parameter initialization strategies from Appendix D
- Feature count sensitivity analysis

### Phase 3: Verification Suite (Week 2)

#### 3.1 XOR Benchmark
**Paper Specifications (Figure 1):**
- 2D input space
- 2 prototypes, 2 features minimum
- Specific parameter values from paper:
  - p₀ = {}, p₁ = {f₀, f₁}
  - Must achieve 100% accuracy

**Test Matrix (from Appendix D):**
- 6 intersection reduction methods
- 2 difference reduction methods
- Multiple feature counts: {1, 2, 4, 8, 16, 32}
- 3 initialization distributions
- Convergence probability tracking

#### 3.2 MNIST Benchmark
**Paper Specifications (Table 1, Section 3.4):**
- ResNet-50 backbone integration
- Frozen backbone: 24.7% improvement expected
- Full fine-tuning: marginal improvements
- Feature bank size: 224 for NABirds, variable for MNIST

### Phase 4: Research Tools (Week 3)

#### 4.1 Intervention Manager (`verskyt/interventions/manager.py`)
**Capabilities from Paper:**
- Prototype inspection and modification
- Feature grounding to concepts
- Counterfactual analysis
- Salience computation (Equation 2)

#### 4.2 Visualization Tools
**Data-Domain Visualization (Section 2.5):**
- Project parameters back to input space
- Visualize learned features as interpretable patterns
- Compare with linear layer weights

### Phase 5: Advanced Models (Week 3-4)

#### 5.1 TverskyResNet
- Drop-in replacement for ResNet final layer
- Support for frozen backbone adaptation
- Feature sharing capabilities

#### 5.2 Parameter Efficiency
- Feature bank sharing across layers
- Prototype tying (similar to weight tying)
- Expected: 34.8% parameter reduction (Table 2)

### Testing Strategy

#### Unit Tests
```python
# Each component tested individually
- test_tversky_similarity_product_reduction()
- test_tversky_similarity_substractmatch()
- test_gradient_flow()
- test_feature_membership()
- test_salience_computation()
```

#### Integration Tests
```python
# End-to-end verification
- test_xor_single_layer()
- test_mnist_accuracy()
- test_resnet_integration()
- test_intervention_manager()
```

#### Regression Tests
```python
# Ensure paper results reproducible
- test_xor_convergence_probability()
- test_mnist_baseline_comparison()
- test_asymmetry_properties()
```

### Validation Metrics

1. **XOR Test**:
   - Must solve with single layer
   - Convergence rate > 40% with optimal settings

2. **MNIST Accuracy**:
   - Frozen backbone: >60% accuracy
   - Full training: >98% accuracy

3. **Asymmetry Verification**:
   - α > β in trained models
   - Salience correlates with "goodness of form"

4. **Interpretability**:
   - Features visually recognizable
   - Prototypes combine class features

### Implementation Order

1. **Core Mathematics** (Most Critical)
   - Tversky similarity function
   - All reduction methods
   - Gradient verification

2. **Basic Layers**
   - TverskyProjectionLayer
   - TverskySimilarityLayer
   - Parameter initialization

3. **Validation Suite**
   - XOR test implementation
   - Convergence analysis
   - MNIST benchmark

4. **Research Tools**
   - Intervention API
   - Visualization methods
   - Salience computation

5. **Advanced Features**
   - Model architectures
   - Feature sharing
   - Performance optimizations

### Success Criteria

✓ XOR solved with single layer (matches Figure 1)
✓ Gradient flow verified mathematically
✓ MNIST results within 5% of paper
✓ Asymmetry properties confirmed (α > β)
✓ Features interpretable via visualization
✓ Intervention API functional
✓ All paper equations implemented correctly

### Risk Mitigation

1. **Numerical Instability**:
   - Add θ parameter for stability (default 1e-7)
   - Clamp α, β to non-negative values
   - ReLU for membership computation

2. **Convergence Issues**:
   - Multiple initialization strategies
   - Hyperparameter search for optimal settings
   - Early stopping criteria

3. **Performance**:
   - Efficient einsum operations
   - Batch processing optimizations
   - GPU acceleration support

This plan ensures a rigorous, testable implementation that faithfully reproduces the paper's results while providing a research-grade tool for the community.
