# Research Documentation

Research tools, experimental features, and analysis capabilities for Tversky Neural Networks.

## Research Capabilities

### [Experiments](experiments.md) *(Coming Soon)*
**Reproducible experimental setups** - Tools and scripts for reproducing paper results

**Available experiments:**
- **XOR Benchmark**: Single layer XOR learning with convergence analysis
- **MNIST Classification**: Handwritten digit recognition with interpretability
- **ResNet-50 Integration**: Image classification improvements
- **GPT-2 Language Modeling**: Parameter efficiency and perplexity improvements

**Experiment framework:**
- Standardized experimental protocols
- Automated hyperparameter sweeps  
- Statistical significance testing
- Result reproduction verification

### [Interpretability](interpretability.md) *(Coming Soon)*
**Visualization and analysis tools** - Understanding learned representations

**Visualization capabilities:**
- **Data-domain parameter visualization**: Project parameters back to input space
- **Prototype analysis**: Understand learned concept prototypes
- **Feature interpretation**: Visualize learned feature patterns
- **Salience computation**: Measure and visualize feature salience

**Analysis tools:**
- Asymmetry measurement and visualization
- Common/distinctive feature decomposition
- Prototype hierarchy analysis
- Similarity explanation generation

## Research Tools

### Intervention Manager
```python
from verskyt.research.interventions import InterventionManager

manager = InterventionManager(model)

# Manipulate prototypes
manager.set_prototype(class_id=5, prototype=custom_prototype)

# Analyze feature contributions
contributions = manager.analyze_features(input_sample)

# Counterfactual analysis
counterfactual = manager.generate_counterfactual(
    input_sample, target_class=3
)
```

### Visualization Suite
```python
from verskyt.research.visualization import (
    visualize_prototypes,
    plot_salience_distribution,
    show_similarity_decomposition
)

# Visualize learned prototypes in input space
visualize_prototypes(model.prototypes, input_shape=(28, 28))

# Show feature salience distribution
plot_salience_distribution(model, validation_data)

# Decompose similarity into components
show_similarity_decomposition(
    object_a, object_b, model.features,
    alpha=model.alpha, beta=model.beta, theta=model.theta
)
```

### Experimental Framework
```python
from verskyt.research.experiments import XORExperiment, MNISTExperiment

# Run XOR convergence analysis
xor_exp = XORExperiment()
results = xor_exp.run_convergence_study(
    intersection_methods=['product', 'mean', 'min'],
    difference_methods=['ignorematch', 'substractmatch'],
    num_trials=100
)

# Reproduce MNIST results
mnist_exp = MNISTExperiment()
accuracy = mnist_exp.run_baseline_comparison(
    model_type='TverskyResNet50',
    frozen_backbone=True
)
```

## Experimental Protocols

### XOR Benchmark Protocol
**Objective**: Verify non-linear decision boundary capability
- **Setup**: 2D input space, binary classification
- **Architecture**: Single TverskyProjectionLayer
- **Metrics**: Convergence probability, accuracy, parameter sensitivity
- **Expected result**: >40% convergence rate with optimal hyperparameters

### MNIST Interpretability Protocol
**Objective**: Demonstrate interpretable feature learning
- **Setup**: 28x28 handwritten digits, 10 classes
- **Architecture**: CNN backbone + TverskyProjectionLayer
- **Metrics**: Accuracy, prototype interpretability, feature recognizability
- **Expected result**: >98% accuracy with interpretable prototypes

### ResNet-50 Integration Protocol
**Objective**: Show performance improvements in standard architectures
- **Setup**: ImageNet pre-trained ResNet-50, various datasets
- **Architecture**: ResNet-50 + TverskyProjectionLayer head
- **Metrics**: Classification accuracy, parameter efficiency
- **Expected result**: Up to 24.7% improvement on NABirds (frozen backbone)

### GPT-2 Language Modeling Protocol
**Objective**: Demonstrate parameter efficiency and performance
- **Setup**: GPT-2 small, Penn Treebank dataset
- **Architecture**: GPT-2 with Tversky projection layers
- **Metrics**: Perplexity, parameter count, training time
- **Expected result**: 7.5% perplexity reduction, 34.8% parameter reduction

## Analysis Methods

### Asymmetry Analysis
```python
def analyze_asymmetry(model, dataset):
    """Measure and visualize similarity asymmetry."""
    asymmetry_scores = []
    for a, b in dataset.pairs:
        sim_ab = model.similarity(a, b)
        sim_ba = model.similarity(b, a)
        asymmetry = abs(sim_ab - sim_ba) / max(sim_ab, sim_ba)
        asymmetry_scores.append(asymmetry)
    return asymmetry_scores
```

### Salience Ranking
```python
def compute_salience_ranking(model, input_sample):
    """Rank features by salience for given input."""
    feature_activations = model.compute_feature_activations(input_sample)
    salience_scores = feature_activations * (feature_activations > 0)
    return torch.argsort(salience_scores, descending=True)
```

### Prototype Hierarchy
```python
def build_prototype_hierarchy(model):
    """Build hierarchical clustering of learned prototypes."""
    similarities = model.compute_prototype_similarities()
    hierarchy = hierarchical_clustering(similarities)
    return hierarchy
```

## Reproducibility Guidelines

### Experimental Setup
- **Random seeds**: Always set and document random seeds
- **Hyperparameters**: Record all hyperparameter settings
- **Environment**: Document software versions and hardware specs
- **Data**: Use standardized datasets and splits

### Statistical Analysis
- **Multiple runs**: Report mean ± std across multiple runs
- **Significance testing**: Use appropriate statistical tests
- **Effect sizes**: Report practical significance, not just statistical
- **Confidence intervals**: Provide uncertainty estimates

### Result Reporting
- **Baseline comparisons**: Always compare against appropriate baselines
- **Ablation studies**: Isolate the effect of individual components
- **Failure cases**: Document when and why methods fail
- **Computational costs**: Report training time and resource usage

## Research Applications

### Cognitive Science
- **Human similarity modeling**: Compare TNN similarities with human judgments
- **Prototype theory validation**: Test psychological theories of categorization
- **Feature salience studies**: Investigate attention and feature weighting

### Machine Learning
- **Interpretable AI**: Develop more interpretable neural architectures
- **Few-shot learning**: Leverage similarity-based learning
- **Transfer learning**: Study how similarity representations transfer

### Computer Vision
- **Object recognition**: Understand how humans recognize objects
- **Medical imaging**: Interpretable diagnostic systems  
- **Artistic style**: Model artistic similarity and style transfer

### Natural Language Processing
- **Semantic similarity**: Model human-like semantic judgments
- **Language models**: More interpretable language understanding
- **Question answering**: Similarity-based reasoning systems

## Contributing Research

We welcome research contributions! Areas of particular interest:

- **New similarity measures**: Extensions to Tversky similarity
- **Architectural innovations**: Novel ways to use Tversky layers
- **Application studies**: New domains and use cases
- **Theoretical analysis**: Mathematical properties and guarantees
- **Empirical studies**: Large-scale evaluation and comparison

See our [contribution guidelines](../../README.md) for submission process.