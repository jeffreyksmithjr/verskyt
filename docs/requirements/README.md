# Requirements Documentation

This section contains comprehensive specifications and analysis for implementing Tversky Neural Networks.

## Documents

### [TNN Specification](tnn-specification.md)
**The definitive implementation guide** - Complete mathematical formulations, architectural patterns, hyperparameter guidance, and performance benchmarks extracted from the original paper.

**Key sections:**
- Core mathematical formulations (Tversky similarity, feature intersections/differences)
- Neural network module specifications (Similarity Layer, Projection Layer)
- Implementation requirements (shapes, differentiability, initialization)
- Performance expectations and validation criteria

**Use this document for:**
- Understanding the complete mathematical framework
- Implementation specifications and constraints
- Hyperparameter recommendations based on paper experiments
- Performance benchmarks and validation criteria

## Quick Reference

### Core Equations
- **Tversky Similarity**: `S(a,b) = θ*f(A∩B) - α*f(A-B) - β*f(B-A)`
- **Feature Salience**: `f(A) = Σₖ (a·fₖ) * 𝟙[a·fₖ > 0]`
- **Intersection**: `f(A∩B) = Σₖ Ψ(a·fₖ, b·fₖ) * 𝟙[a·fₖ>0 ∧ b·fₖ>0]`

### Key Implementation Requirements
- **Dual representation**: Objects as both vectors and sets
- **Best hyperparameters**: `product` intersection, `substractmatch` difference
- **Initialization**: Uniform distribution for prototypes and features
- **Shape conventions**: [batch_size, features] for inputs

### Performance Benchmarks
- **XOR**: Single layer must achieve 100% accuracy
- **ResNet-50**: Up to 24.7% improvement on NABirds with frozen backbone
- **GPT-2**: 7.5% perplexity reduction with 34.8% parameter reduction

## Paper Reference
Based on: "Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity" (Doumbouya et al., 2025)