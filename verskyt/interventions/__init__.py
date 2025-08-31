"""
Intervention tools for Tversky Neural Networks.

Provides APIs for inspecting, modifying, and analyzing TNN models:
- Prototype inspection and modification
- Feature grounding to semantic concepts
- Counterfactual analysis
- Impact assessment of model changes
"""

from .analysis import CounterfactualAnalyzer, ImpactAssessment
from .grounding import ConceptGrounding, ConceptLibrary, FeatureGrounder
from .manager import InterventionManager

__all__ = [
    "InterventionManager",
    "CounterfactualAnalyzer",
    "ImpactAssessment",
    "FeatureGrounder",
    "ConceptGrounding",
    "ConceptLibrary",
]
