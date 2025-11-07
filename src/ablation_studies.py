"""
Ablation Studies for Mushroom Classification System

This script performs ablation studies to evaluate the contribution of different
components of the hybrid CNN-MDP system.

Components tested:
1. CNN only (baseline)
2. CNN + Rule-based (no MDP)
3. CNN + MDP (no conflict detection)
4. CNN + MDP + Conflict Detection (full system)
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import os
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

from model import FineTuneResNet18
from mdp_system import (
    MDPState, Action, SafetyStatus, TransitionModel, RewardModel, MDPPolicy,
    create_initial_state, make_final_decision
)


class AblationEvaluator:
    """Evaluator for ablation studies"""
    
    def __init__(self, model_path: str, kb_path: str, device: str = "cpu"):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained CNN model
            kb_path: Path to knowledge base JSON
            device: Device for model inference
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.kb = self._load_kb(kb_path)
        self.transition_model = TransitionModel()
        self.reward_model = RewardModel()
        self.mdp_policy = MDPPolicy(self.transition_model, self.reward_model)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path: str):
        """Load CNN model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        classes = checkpoint['classes']
        num_classes = len(classes)
        model = FineTuneResNet18(num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model, classes
    
    def _load_kb(self, kb_path: str) -> Dict:
        """Load knowledge base"""
        with open(kb_path, 'r') as f:
            return json.load(f)
    
    def predict_cnn(self, image: Image.Image) -> Tuple[str, float, Dict]:
        """
        CNN-only prediction (baseline)
        
        Returns:
            (predicted_class, confidence, kb_info)
        """
        model, classes = self.model
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            confidence = confidence.item()
            pred_class = classes[pred_idx.item()]
        
        kb_info = self.kb.get(pred_class, {})
        return pred_class, confidence, kb_info
    
    def predict_cnn_rule_based(self, image: Image.Image, 
                                answers: Dict[str, str]) -> Tuple[str, SafetyStatus, float]:
        """
        CNN + Rule-based (no MDP)
        
        Uses fixed rule-based logic without adaptive questioning
        """
        pred_class, confidence, kb_info = self.predict_cnn(image)
        
        # Fixed rule-based logic (no MDP)
        if pred_class == "Agaricus":
            volva = answers.get("ask_volva", "")
            spore = answers.get("ask_spore_print", "")
            
            if volva == "Yes" or spore == "White":
                return pred_class, SafetyStatus.DANGER, 0.1
            elif spore == "Dark Brown / Chocolate" and volva == "No":
                return pred_class, SafetyStatus.SAFE, confidence
            else:
                return pred_class, SafetyStatus.UNCERTAIN, confidence * 0.7
        
        elif pred_class == "Amanita":
            return pred_class, SafetyStatus.DANGER, 0.9
        
        elif kb_info.get('edibility') in ["Deadly Poisonous", "Varies (Some Deadly)"]:
            return pred_class, SafetyStatus.DANGER, confidence
        
        else:
            if confidence > 0.8:
                return pred_class, SafetyStatus.SAFE, confidence
            else:
                return pred_class, SafetyStatus.UNCERTAIN, confidence
    
    def predict_cnn_mdp_no_conflict(self, image: Image.Image,
                                     answers: Dict[str, str]) -> Tuple[str, SafetyStatus, float]:
        """
        CNN + MDP (no conflict detection)
        
        Uses MDP but without conflict detection mechanism
        """
        pred_class, confidence, kb_info = self.predict_cnn(image)
        initial_state = create_initial_state(pred_class, confidence, kb_info)
        
        # Simulate MDP transitions without conflict detection
        current_state = initial_state
        for action_value, answer in answers.items():
            action_enum = None
            for action in Action:
                if action.value == action_value:
                    action_enum = action
                    break
            
            if action_enum and action_enum != Action.MAKE_DECISION:
                current_state, _ = self.transition_model.get_transition(
                    current_state, action_enum, answer
                )
        
        # Make decision without conflict detection
        final_state = self._make_decision_no_conflict(current_state)
        return final_state.predicted_class, final_state.safety_status, final_state.decision_confidence
    
    def _make_decision_no_conflict(self, state: MDPState) -> MDPState:
        """Make decision without conflict detection (simplified version)"""
        state.is_terminal = True
        
        if state.predicted_class == "Agaricus":
            volva = state.answers.get("ask_volva", "")
            spore = state.answers.get("ask_spore_print", "")
            
            if volva == "Yes" or spore == "White":
                state.safety_status = SafetyStatus.DANGER
                state.decision_confidence = 0.1
            elif spore == "Dark Brown / Chocolate" and volva == "No":
                state.safety_status = SafetyStatus.SAFE
                state.decision_confidence = state.confidence
            else:
                state.safety_status = SafetyStatus.UNCERTAIN
                state.decision_confidence = state.confidence * 0.7
        else:
            if state.confidence > 0.8:
                state.safety_status = SafetyStatus.SAFE
            else:
                state.safety_status = SafetyStatus.UNCERTAIN
            state.decision_confidence = state.confidence
        
        return state
    
    def predict_full_system(self, image: Image.Image,
                            answers: Dict[str, str]) -> Tuple[str, SafetyStatus, float]:
        """
        Full system: CNN + MDP + Conflict Detection
        """
        pred_class, confidence, kb_info = self.predict_cnn(image)
        initial_state = create_initial_state(pred_class, confidence, kb_info)
        
        # Simulate MDP transitions
        current_state = initial_state
        for action_value, answer in answers.items():
            action_enum = None
            for action in Action:
                if action.value == action_value:
                    action_enum = action
                    break
            
            if action_enum and action_enum != Action.MAKE_DECISION:
                current_state, _ = self.transition_model.get_transition(
                    current_state, action_enum, answer
                )
        
        # Make decision with conflict detection
        final_state = make_final_decision(current_state)
        return final_state.predicted_class, final_state.safety_status, final_state.decision_confidence
    
    def evaluate_ablation(self, test_cases: List[Dict]) -> Dict:
        """
        Run ablation study on test cases
        
        Args:
            test_cases: List of test cases, each with:
                - 'image': PIL Image
                - 'answers': Dict of feature answers
                - 'ground_truth': True class (optional)
                - 'expected_safety': Expected safety status (optional)
        
        Returns:
            Dictionary with results for each ablation variant
        """
        results = {
            'cnn_only': {'correct': 0, 'safe_correct': 0, 'danger_correct': 0, 'total': 0},
            'cnn_rule': {'correct': 0, 'safe_correct': 0, 'danger_correct': 0, 'total': 0},
            'cnn_mdp_no_conflict': {'correct': 0, 'safe_correct': 0, 'danger_correct': 0, 'total': 0},
            'full_system': {'correct': 0, 'safe_correct': 0, 'danger_correct': 0, 'total': 0}
        }
        
        for case in test_cases:
            image = case['image']
            answers = case.get('answers', {})
            ground_truth = case.get('ground_truth')
            expected_safety = case.get('expected_safety')
            
            # CNN only
            pred_class, confidence, _ = self.predict_cnn(image)
            if ground_truth:
                results['cnn_only']['correct'] += (pred_class == ground_truth)
            results['cnn_only']['total'] += 1
            
            # CNN + Rule-based
            pred_class_r, safety_r, conf_r = self.predict_cnn_rule_based(image, answers)
            if expected_safety:
                results['cnn_rule']['safe_correct'] += (safety_r == expected_safety)
            if ground_truth:
                results['cnn_rule']['correct'] += (pred_class_r == ground_truth)
            results['cnn_rule']['total'] += 1
            
            # CNN + MDP (no conflict)
            pred_class_m, safety_m, conf_m = self.predict_cnn_mdp_no_conflict(image, answers)
            if expected_safety:
                results['cnn_mdp_no_conflict']['safe_correct'] += (safety_m == expected_safety)
            if ground_truth:
                results['cnn_mdp_no_conflict']['correct'] += (pred_class_m == ground_truth)
            results['cnn_mdp_no_conflict']['total'] += 1
            
            # Full system
            pred_class_f, safety_f, conf_f = self.predict_full_system(image, answers)
            if expected_safety:
                results['full_system']['safe_correct'] += (safety_f == expected_safety)
            if ground_truth:
                results['full_system']['correct'] += (pred_class_f == ground_truth)
            results['full_system']['total'] += 1
        
        # Calculate accuracies
        for variant in results:
            if results[variant]['total'] > 0:
                results[variant]['accuracy'] = results[variant]['correct'] / results[variant]['total']
                results[variant]['safety_accuracy'] = results[variant]['safe_correct'] / results[variant]['total']
        
        return results


def create_test_cases() -> List[Dict]:
    """
    Create test cases for ablation study
    
    Returns:
        List of test cases with different scenarios
    """
    test_cases = []
    
    # Test case 1: Agaricus with correct features (should be SAFE)
    test_cases.append({
        'name': 'Agaricus_Correct',
        'image': None,  # Would load actual image
        'answers': {
            'ask_volva': 'No',
            'ask_spore_print': 'Dark Brown / Chocolate',
            'ask_habitat': 'On ground/soil',
            'ask_gill_color': 'Pink',
            'ask_bruising': 'No, no color change',
            'ask_odor': 'Anise/licorice'
        },
        'ground_truth': 'Agaricus',
        'expected_safety': SafetyStatus.SAFE
    })
    
    # Test case 2: Agaricus with conflicting features (should be UNCERTAIN)
    test_cases.append({
        'name': 'Agaricus_Conflicting',
        'image': None,
        'answers': {
            'ask_volva': 'No',
            'ask_spore_print': 'Dark Brown / Chocolate',
            'ask_habitat': 'On wood',  # Conflict
            'ask_gill_color': 'Yellow',  # Conflict
            'ask_bruising': 'Yes, it changes color',  # Conflict
            'ask_odor': 'Foul/rotten'  # Conflict
        },
        'ground_truth': 'Agaricus',
        'expected_safety': SafetyStatus.UNCERTAIN
    })
    
    # Test case 3: Amanita (should be DANGER)
    test_cases.append({
        'name': 'Amanita_Danger',
        'image': None,
        'answers': {
            'ask_volva': 'Yes',
            'ask_spore_print': 'White',
            'ask_habitat': 'On ground/soil',
            'ask_gill_color': 'White',
            'ask_bruising': 'No, no color change',
            'ask_odor': 'No distinct odor'
        },
        'ground_truth': 'Amanita',
        'expected_safety': SafetyStatus.DANGER
    })
    
    # Test case 4: Agaricus misidentified as Amanita (should detect danger)
    test_cases.append({
        'name': 'Agaricus_With_Volva',
        'image': None,
        'answers': {
            'ask_volva': 'Yes',  # Critical danger signal
            'ask_spore_print': 'Dark Brown / Chocolate',
            'ask_habitat': 'On ground/soil',
            'ask_gill_color': 'Pink',
            'ask_bruising': 'No, no color change',
            'ask_odor': 'Anise/licorice'
        },
        'ground_truth': 'Agaricus',
        'expected_safety': SafetyStatus.DANGER  # Should override to danger
    })
    
    return test_cases


if __name__ == "__main__":
    # Example usage
    evaluator = AblationEvaluator(
        model_path="mushroom_model.pt",
        kb_path="knowledge_base.json",
        device="cpu"
    )
    
    test_cases = create_test_cases()
    results = evaluator.evaluate_ablation(test_cases)
    
    print("Ablation Study Results:")
    print("=" * 60)
    for variant, metrics in results.items():
        print(f"\n{variant}:")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
        print(f"  Safety Accuracy: {metrics.get('safety_accuracy', 0):.3f}")
        print(f"  Correct: {metrics.get('correct', 0)}/{metrics.get('total', 0)}")

