"""
Tests for intervention manager and analysis tools.

Tests all components of the intervention system:
- InterventionManager: prototype/feature inspection and modification
- ImpactAssessment: quantifying intervention effects
- CounterfactualAnalyzer: counterfactual generation
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import Mock

from verskyt.layers.projection import TverskyProjectionLayer
from verskyt.interventions.manager import InterventionManager, PrototypeInfo, FeatureInfo
from verskyt.interventions.analysis import ImpactAssessment, CounterfactualAnalyzer, ImpactMetrics


class SimpleTNNModel(nn.Module):
    """Simple TNN model for testing interventions."""
    
    def __init__(self, in_features=2, num_classes=2, num_features=4):
        super().__init__()
        self.tnn_layer = TverskyProjectionLayer(
            in_features=in_features,
            num_prototypes=num_classes,
            num_features=num_features,
            alpha=0.5,
            beta=0.5,
            learnable_ab=True,
            prototype_init="xavier_uniform",
            feature_init="xavier_uniform"
        )
    
    def forward(self, x):
        return self.tnn_layer(x)


@pytest.fixture
def simple_model():
    """Create a simple TNN model for testing."""
    torch.manual_seed(42)
    return SimpleTNNModel()


@pytest.fixture
def intervention_manager(simple_model):
    """Create an InterventionManager for testing."""
    return InterventionManager(simple_model, "test_model")


@pytest.fixture
def test_data():
    """Create test data."""
    torch.manual_seed(42)
    inputs = torch.randn(10, 2)
    targets = torch.randint(0, 2, (10,))
    return inputs, targets


class TestInterventionManager:
    """Test InterventionManager functionality."""
    
    def test_initialization(self, intervention_manager):
        """Test InterventionManager initialization."""
        assert intervention_manager.model_name == "test_model"
        assert intervention_manager.num_layers == 1
        assert "tnn_layer" in intervention_manager.layer_names
        assert len(intervention_manager.get_intervention_history()) == 0
    
    def test_layer_discovery(self, intervention_manager):
        """Test TNN layer discovery."""
        layer_names = intervention_manager.layer_names
        assert len(layer_names) == 1
        assert "tnn_layer" in layer_names
        
        layer_info = intervention_manager.get_layer_info("tnn_layer")
        assert layer_info["layer_type"] == "TverskyProjectionLayer"
        assert layer_info["in_features"] == 2
        assert layer_info["num_prototypes"] == 2
        assert layer_info["num_features"] == 4
    
    def test_prototype_listing(self, intervention_manager):
        """Test prototype listing functionality."""
        prototypes = intervention_manager.list_prototypes()
        assert len(prototypes) == 2  # 2 prototypes for 2 classes
        
        for proto in prototypes:
            assert isinstance(proto, PrototypeInfo)
            assert proto.layer_name == "tnn_layer"
            assert proto.shape == torch.Size([2])  # in_features=2
            assert proto.norm > 0
    
    def test_feature_listing(self, intervention_manager):
        """Test feature listing functionality."""
        features = intervention_manager.list_features()
        assert len(features) == 4  # 4 features
        
        for feat in features:
            assert isinstance(feat, FeatureInfo)
            assert feat.layer_name == "tnn_layer"
            assert feat.shape == torch.Size([2])  # in_features=2
            assert feat.norm > 0
    
    def test_get_specific_prototype(self, intervention_manager):
        """Test getting specific prototype."""
        proto = intervention_manager.get_prototype("tnn_layer", 0)
        assert isinstance(proto, PrototypeInfo)
        assert proto.prototype_index == 0
        assert proto.layer_name == "tnn_layer"
        
        # Test error handling
        with pytest.raises(ValueError, match="Layer 'nonexistent' not found"):
            intervention_manager.get_prototype("nonexistent", 0)
        
        with pytest.raises(ValueError, match="out of range"):
            intervention_manager.get_prototype("tnn_layer", 10)
    
    def test_get_specific_feature(self, intervention_manager):
        """Test getting specific feature."""
        feat = intervention_manager.get_feature("tnn_layer", 0)
        assert isinstance(feat, FeatureInfo)
        assert feat.feature_index == 0
        assert feat.layer_name == "tnn_layer"
        
        # Test error handling
        with pytest.raises(ValueError, match="Layer 'nonexistent' not found"):
            intervention_manager.get_feature("nonexistent", 0)
        
        with pytest.raises(ValueError, match="out of range"):
            intervention_manager.get_feature("tnn_layer", 10)
    
    def test_prototype_modification(self, intervention_manager):
        """Test prototype modification functionality."""
        # Get original prototype
        original_proto = intervention_manager.get_prototype("tnn_layer", 0)
        original_vector = original_proto.vector.clone()
        
        # Create new vector
        new_vector = torch.randn_like(original_vector)
        
        # Modify prototype
        modified_proto = intervention_manager.modify_prototype(
            "tnn_layer", 0, new_vector
        )
        
        # Verify modification
        assert torch.allclose(modified_proto.vector, new_vector)
        assert not torch.allclose(modified_proto.vector, original_vector)
        
        # Check intervention history
        history = intervention_manager.get_intervention_history()
        assert len(history) == 1
        assert history[0]["type"] == "prototype_modification"
        assert history[0]["layer_name"] == "tnn_layer"
        assert history[0]["prototype_index"] == 0
    
    def test_feature_modification(self, intervention_manager):
        """Test feature modification functionality."""
        # Get original feature
        original_feat = intervention_manager.get_feature("tnn_layer", 0)
        original_vector = original_feat.vector.clone()
        
        # Create new vector
        new_vector = torch.randn_like(original_vector)
        
        # Modify feature
        modified_feat = intervention_manager.modify_feature(
            "tnn_layer", 0, new_vector
        )
        
        # Verify modification
        assert torch.allclose(modified_feat.vector, new_vector)
        assert not torch.allclose(modified_feat.vector, original_vector)
        
        # Check intervention history
        history = intervention_manager.get_intervention_history()
        assert len(history) == 1
        assert history[0]["type"] == "feature_modification"
    
    def test_reset_to_original(self, intervention_manager):
        """Test resetting to original state."""
        # Get original state
        original_proto = intervention_manager.get_prototype("tnn_layer", 0)
        original_vector = original_proto.vector.clone()
        
        # Modify prototype
        new_vector = torch.randn_like(original_vector)
        intervention_manager.modify_prototype("tnn_layer", 0, new_vector)
        
        # Verify modification took effect
        modified_proto = intervention_manager.get_prototype("tnn_layer", 0)
        assert not torch.allclose(modified_proto.vector, original_vector)
        
        # Reset to original
        intervention_manager.reset_to_original()
        
        # Verify reset
        reset_proto = intervention_manager.get_prototype("tnn_layer", 0)
        assert torch.allclose(reset_proto.vector, original_vector)
        assert len(intervention_manager.get_intervention_history()) == 0
    
    def test_summary(self, intervention_manager):
        """Test summary generation."""
        summary = intervention_manager.summary()
        assert "test_model" in summary
        assert "TNN Layers: 1" in summary
        assert "tnn_layer: TverskyProjectionLayer" in summary
        assert "Prototypes: 2" in summary
        assert "Features: 4" in summary


class TestImpactAssessment:
    """Test ImpactAssessment functionality."""
    
    @pytest.fixture
    def impact_assessor(self, intervention_manager):
        """Create ImpactAssessment instance."""
        return ImpactAssessment(intervention_manager)
    
    def test_initialization(self, impact_assessor, intervention_manager):
        """Test ImpactAssessment initialization."""
        assert impact_assessor.manager is intervention_manager
        assert impact_assessor.model is intervention_manager.model
    
    def test_prototype_impact_assessment(self, impact_assessor, test_data):
        """Test prototype impact assessment."""
        inputs, targets = test_data
        
        # Get original prototype
        original_proto = impact_assessor.manager.get_prototype("tnn_layer", 0)
        
        # Create modified prototype
        new_vector = torch.randn_like(original_proto.vector)
        
        # Assess impact
        impact = impact_assessor.assess_prototype_impact(
            "tnn_layer", 0, new_vector, inputs
        )
        
        # Verify impact metrics
        assert isinstance(impact, ImpactMetrics)
        assert impact.output_distance >= 0
        assert -1 <= impact.output_correlation <= 1
        assert 0 <= impact.prediction_change_rate <= 1
        assert isinstance(impact.confidence_change, float)
        assert isinstance(impact.effect_size, float)
    
    def test_feature_impact_assessment(self, impact_assessor, test_data):
        """Test feature impact assessment."""
        inputs, targets = test_data
        
        # Get original feature
        original_feat = impact_assessor.manager.get_feature("tnn_layer", 0)
        
        # Create modified feature
        new_vector = torch.randn_like(original_feat.vector)
        
        # Assess impact
        impact = impact_assessor.assess_feature_impact(
            "tnn_layer", 0, new_vector, inputs
        )
        
        # Verify impact metrics
        assert isinstance(impact, ImpactMetrics)
        assert impact.output_distance >= 0
        assert -1 <= impact.output_correlation <= 1
        assert 0 <= impact.prediction_change_rate <= 1
    
    def test_sensitivity_analysis(self, impact_assessor, test_data):
        """Test sensitivity analysis."""
        inputs, targets = test_data
        
        # Test prototype sensitivity
        results = impact_assessor.sensitivity_analysis(
            "tnn_layer", "prototype", 0, inputs, [0.1, 0.5]
        )
        
        assert len(results) == 2
        assert 0.1 in results
        assert 0.5 in results
        
        for scale, impact in results.items():
            assert isinstance(impact, ImpactMetrics)
        
        # Test feature sensitivity
        results = impact_assessor.sensitivity_analysis(
            "tnn_layer", "feature", 0, inputs, [0.1, 0.5]
        )
        
        assert len(results) == 2
    
    def test_parameter_restoration(self, impact_assessor, test_data):
        """Test that impact assessment doesn't permanently modify parameters."""
        inputs, targets = test_data
        
        # Get original prototype
        original_proto = impact_assessor.manager.get_prototype("tnn_layer", 0)
        original_vector = original_proto.vector.clone()
        
        # Create modified prototype
        new_vector = torch.randn_like(original_vector)
        
        # Assess impact (should restore original)
        impact_assessor.assess_prototype_impact(
            "tnn_layer", 0, new_vector, inputs
        )
        
        # Verify original is restored
        current_proto = impact_assessor.manager.get_prototype("tnn_layer", 0)
        assert torch.allclose(current_proto.vector, original_vector)


class TestCounterfactualAnalyzer:
    """Test CounterfactualAnalyzer functionality."""
    
    @pytest.fixture
    def counterfactual_analyzer(self, intervention_manager):
        """Create CounterfactualAnalyzer instance."""
        return CounterfactualAnalyzer(intervention_manager)
    
    def test_initialization(self, counterfactual_analyzer, intervention_manager):
        """Test CounterfactualAnalyzer initialization."""
        assert counterfactual_analyzer.manager is intervention_manager
        assert counterfactual_analyzer.model is intervention_manager.model
    
    def test_prototype_counterfactuals(self, counterfactual_analyzer):
        """Test prototype-based counterfactual generation."""
        # Create a simple input that should be easy to flip
        input_sample = torch.tensor([1.0, 0.0])
        
        # Try to find counterfactuals
        counterfactuals = counterfactual_analyzer.find_prototype_counterfactuals(
            input_sample, target_class=1, layer_name="tnn_layer", max_iterations=10
        )
        
        # Should be a list (may be empty if no counterfactuals found)
        assert isinstance(counterfactuals, list)
        
        # If counterfactuals found, validate structure
        for cf in counterfactuals:
            assert hasattr(cf, 'original_input')
            assert hasattr(cf, 'original_output')
            assert hasattr(cf, 'modified_prediction')
            assert hasattr(cf, 'success')
            assert cf.success is True
    
    def test_feature_counterfactuals(self, counterfactual_analyzer):
        """Test feature-based counterfactual generation."""
        input_sample = torch.tensor([1.0, 0.0])
        
        counterfactuals = counterfactual_analyzer.find_feature_counterfactuals(
            input_sample, target_class=1, layer_name="tnn_layer", max_iterations=10
        )
        
        assert isinstance(counterfactuals, list)
        
        for cf in counterfactuals:
            assert hasattr(cf, 'intervention_description')
            assert "feature" in cf.intervention_description.lower()
    
    def test_decision_boundary_analysis(self, counterfactual_analyzer):
        """Test decision boundary analysis."""
        # Create samples near decision boundary
        inputs = torch.tensor([
            [0.1, 0.1],
            [-0.1, 0.1], 
            [0.1, -0.1],
            [-0.1, -0.1]
        ])
        
        results = counterfactual_analyzer.analyze_decision_boundary(
            inputs, "tnn_layer", num_perturbations=3
        )
        
        assert results["layer_name"] == "tnn_layer"
        assert results["num_samples"] == 4
        assert "boundary_stability" in results
        assert isinstance(results["boundary_stability"], dict)
    
    def test_parameter_restoration_after_counterfactuals(self, counterfactual_analyzer):
        """Test that counterfactual analysis doesn't permanently modify parameters."""
        input_sample = torch.tensor([1.0, 0.0])
        
        # Get original prototype
        original_proto = counterfactual_analyzer.manager.get_prototype("tnn_layer", 0)
        original_vector = original_proto.vector.clone()
        
        # Run counterfactual analysis
        counterfactual_analyzer.find_prototype_counterfactuals(
            input_sample, target_class=1, layer_name="tnn_layer", max_iterations=5
        )
        
        # Verify original is restored
        current_proto = counterfactual_analyzer.manager.get_prototype("tnn_layer", 0)
        assert torch.allclose(current_proto.vector, original_vector)


class TestIntegration:
    """Integration tests for intervention system."""
    
    def test_full_intervention_workflow(self, intervention_manager, test_data):
        """Test complete intervention workflow."""
        inputs, targets = test_data
        
        # 1. Inspect model
        summary = intervention_manager.summary()
        assert "test_model" in summary
        
        # 2. List prototypes and features
        prototypes = intervention_manager.list_prototypes()
        features = intervention_manager.list_features()
        assert len(prototypes) == 2
        assert len(features) == 4
        
        # 3. Modify a prototype
        original_proto = prototypes[0]
        new_vector = torch.randn_like(original_proto.vector)
        modified_proto = intervention_manager.modify_prototype(
            "tnn_layer", 0, new_vector
        )
        
        # 4. Assess impact
        impact_assessor = ImpactAssessment(intervention_manager)
        impact = impact_assessor.assess_prototype_impact(
            "tnn_layer", 1, torch.randn_like(original_proto.vector), inputs[:5]
        )
        assert isinstance(impact, ImpactMetrics)
        
        # 5. Generate counterfactuals
        cf_analyzer = CounterfactualAnalyzer(intervention_manager)
        counterfactuals = cf_analyzer.find_prototype_counterfactuals(
            inputs[0], target_class=1, layer_name="tnn_layer", max_iterations=5
        )
        assert isinstance(counterfactuals, list)
        
        # 6. Reset to original
        intervention_manager.reset_to_original()
        reset_proto = intervention_manager.get_prototype("tnn_layer", 0)
        # Note: may not be exactly equal due to floating point precision
        assert torch.norm(reset_proto.vector - original_proto.vector) < 1e-5
    
    def test_error_handling(self, intervention_manager):
        """Test error handling across intervention system."""
        # Invalid layer names
        with pytest.raises(ValueError, match="not found"):
            intervention_manager.get_layer_info("invalid_layer")
        
        with pytest.raises(ValueError, match="not found"):
            intervention_manager.get_prototype("invalid_layer", 0)
        
        with pytest.raises(ValueError, match="not found"):
            intervention_manager.get_feature("invalid_layer", 0)
        
        # Invalid indices
        with pytest.raises(ValueError, match="out of range"):
            intervention_manager.get_prototype("tnn_layer", 999)
        
        with pytest.raises(ValueError, match="out of range"):
            intervention_manager.get_feature("tnn_layer", 999)
        
        # Invalid vector shapes for modification
        with pytest.raises(ValueError, match="shape"):
            wrong_shape_vector = torch.randn(5)  # Should be shape (2,)
            intervention_manager.modify_prototype("tnn_layer", 0, wrong_shape_vector)
    
    def test_intervention_tracking(self, intervention_manager):
        """Test intervention history tracking."""
        # Start with empty history
        assert len(intervention_manager.get_intervention_history()) == 0
        
        # Make several interventions
        new_proto = torch.randn(2)
        intervention_manager.modify_prototype("tnn_layer", 0, new_proto)
        
        new_feat = torch.randn(2)
        intervention_manager.modify_feature("tnn_layer", 0, new_feat)
        
        # Check history
        history = intervention_manager.get_intervention_history()
        assert len(history) == 2
        
        proto_intervention = history[0]
        assert proto_intervention["type"] == "prototype_modification"
        assert proto_intervention["layer_name"] == "tnn_layer"
        assert proto_intervention["prototype_index"] == 0
        
        feat_intervention = history[1]
        assert feat_intervention["type"] == "feature_modification"
        assert feat_intervention["layer_name"] == "tnn_layer"
        assert feat_intervention["feature_index"] == 0
        
        # Reset should clear history
        intervention_manager.reset_to_original()
        assert len(intervention_manager.get_intervention_history()) == 0