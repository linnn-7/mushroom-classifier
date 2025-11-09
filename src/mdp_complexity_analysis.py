"""
MDP Complexity Analysis

This script analyzes the computational complexity of the MDP system,
including state space size, transition model size, and policy complexity.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from mdp_system import Action, MDPState


class ComplexityAnalyzer:
    """Analyze computational complexity of MDP system"""
    
    def __init__(self):
        self.num_classes = 9
        self.num_features = 6
        self.num_actions = 7
        
        # Answer options per feature
        self.answer_options = {
            'ask_volva': 3,  # Yes, No, I don't know
            'ask_spore_print': 5,  # White, Dark Brown, Pink, Black, I don't know
            'ask_habitat': 3,  # On wood, On ground/soil, I don't know
            'ask_gill_color': 6,  # White, Pink, Brown, Black, Yellow, I don't know
            'ask_bruising': 3,  # Yes, No, I don't know
            'ask_odor': 5  # Anise, Almond, Foul, No odor, I don't know
        }
    
    def calculate_theoretical_state_space(self) -> Dict:
        """
        Calculate theoretical state space size
        
        Returns:
            Dictionary with state space calculations
        """
        # State components:
        # - predicted_class: 9 classes
        # - confidence: discretized into 10 levels (0.0-0.1, ..., 0.9-1.0)
        # - features_observed: 2^6 = 64 combinations (each feature observed or not)
        # - answers: product of answer options for observed features
        # - uncertainty: discretized into 10 levels
        
        num_classes = 9
        num_confidence_levels = 10
        num_feature_combinations = 2 ** self.num_features  # 64
        num_uncertainty_levels = 10
        
        # Calculate answer combinations
        # For each feature combination, calculate possible answer combinations
        max_answer_combinations = 1
        for feature, num_options in self.answer_options.items():
            max_answer_combinations *= num_options
        
        # Theoretical maximum
        theoretical_max = (num_classes * num_confidence_levels * 
                          num_feature_combinations * max_answer_combinations * 
                          num_uncertainty_levels)
        
        # More realistic: average answer combinations
        avg_answer_options = np.mean(list(self.answer_options.values()))
        avg_answer_combinations = avg_answer_options ** self.num_features
        
        realistic_estimate = (num_classes * num_confidence_levels * 
                            num_feature_combinations * avg_answer_combinations * 
                            num_uncertainty_levels)
        
        return {
            'theoretical_max': theoretical_max,
            'realistic_estimate': realistic_estimate,
            'components': {
                'num_classes': num_classes,
                'num_confidence_levels': num_confidence_levels,
                'num_feature_combinations': num_feature_combinations,
                'max_answer_combinations': max_answer_combinations,
                'avg_answer_combinations': avg_answer_combinations,
                'num_uncertainty_levels': num_uncertainty_levels
            }
        }
    
    def calculate_effective_state_space(self) -> Dict:
        """
        Calculate effective (reachable) state space
        
        In practice, we only explore states reachable from initial CNN predictions
        """
        # For each initial prediction, we explore at most:
        # - All feature combinations: 2^6 = 64
        # - All answer combinations for observed features
        
        num_initial_predictions = self.num_classes
        num_confidence_levels = 10  # Discretized
        
        # Effective states per initial prediction
        # We typically ask 4-6 questions, so we explore 2^4 to 2^6 feature combinations
        avg_features_observed = 4.5
        num_feature_combinations_effective = 2 ** int(avg_features_observed)  # ~32
        
        # Average answer combinations for observed features
        avg_answer_options = np.mean(list(self.answer_options.values()))
        avg_answer_combinations_effective = avg_answer_options ** avg_features_observed
        
        effective_states = (num_initial_predictions * num_confidence_levels * 
                          num_feature_combinations_effective * 
                          avg_answer_combinations_effective)
        
        return {
            'effective_states': effective_states,
            'reduction_factor': self.calculate_theoretical_state_space()['theoretical_max'] / effective_states,
            'components': {
                'num_initial_predictions': num_initial_predictions,
                'num_confidence_levels': num_confidence_levels,
                'num_feature_combinations_effective': num_feature_combinations_effective,
                'avg_answer_combinations_effective': avg_answer_combinations_effective
            }
        }
    
    def analyze_transition_model_complexity(self) -> Dict:
        """
        Analyze transition model storage and computation complexity
        """
        # Transition model stores: (state, action, answer) -> (next_state, probability)
        
        effective_states = self.calculate_effective_state_space()['effective_states']
        num_actions = self.num_actions
        avg_answer_options = np.mean(list(self.answer_options.values()))
        
        # Storage complexity: O(|S| * |A| * |Answers|)
        storage_size = effective_states * num_actions * avg_answer_options
        
        # Computation complexity per transition: O(1) lookup (dictionary)
        # But we need to compute uncertainty updates: O(1) per feature
        
        return {
            'storage_size': storage_size,
            'storage_complexity': f'O(|S| * |A| * |Answers|) = O({effective_states:.0e} * {num_actions} * {avg_answer_options:.1f})',
            'computation_per_transition': 'O(1)',
            'uncertainty_update': 'O(|F|) = O(6)'
        }
    
    def analyze_policy_complexity(self) -> Dict:
        """
        Analyze policy selection complexity
        """
        # Heuristic policy: O(|A|) per state
        heuristic_complexity = self.num_actions
        
        # Value iteration would be: O(|S|^2 * |A|) per iteration
        effective_states = self.calculate_effective_state_space()['effective_states']
        value_iteration_complexity = effective_states ** 2 * self.num_actions
        
        return {
            'heuristic_policy': {
                'complexity': f'O(|A|) = O({heuristic_complexity})',
                'time_per_state': 'O(1)',
                'total_time': f'O(T) where T ≤ {self.num_features}'
            },
            'value_iteration': {
                'complexity': f'O(|S|^2 * |A|) = O({effective_states:.0e}^2 * {self.num_actions})',
                'time_per_iteration': f'O({value_iteration_complexity:.0e})',
                'iterations_needed': 'O(log(1/ε)) for convergence'
            },
            'speedup_factor': value_iteration_complexity / heuristic_complexity
        }
    
    def analyze_worst_case_time(self) -> Dict:
        """
        Analyze worst-case time complexity for full identification
        """
        # Worst case: ask all 6 questions
        max_questions = self.num_features
        
        # Per question:
        # - Policy selection: O(|A|) = O(7)
        # - Transition computation: O(1)
        # - State update: O(1)
        
        # Final decision:
        # - Conflict detection: O(|F|) = O(6)
        # - Confidence adjustment: O(1)
        
        per_question_time = 7 + 1 + 1  # Policy + transition + update
        decision_time = 6 + 1  # Conflict detection + confidence adjustment
        
        worst_case_time = max_questions * per_question_time + decision_time
        
        return {
            'worst_case_questions': max_questions,
            'per_question_time': per_question_time,
            'decision_time': decision_time,
            'total_time': worst_case_time,
            'complexity': f'O(T) where T ≤ {max_questions}',
            'practical_time': f'< {worst_case_time * 0.001}ms (assuming O(1) operations)'
        }
    
    def generate_complexity_report(self) -> str:
        """
        Generate comprehensive complexity analysis report
        """
        state_space = self.calculate_theoretical_state_space()
        effective_space = self.calculate_effective_state_space()
        transition = self.analyze_transition_model_complexity()
        policy = self.analyze_policy_complexity()
        worst_case = self.analyze_worst_case_time()
        
        report = f"""
MDP Complexity Analysis Report
{'=' * 60}

1. STATE SPACE ANALYSIS
   Theoretical Maximum: {state_space['theoretical_max']:.2e} states
   Realistic Estimate: {state_space['realistic_estimate']:.2e} states
   Effective (Reachable): {effective_space['effective_states']:.2e} states
   Reduction Factor: {effective_space['reduction_factor']:.2e}x

2. TRANSITION MODEL COMPLEXITY
   Storage Size: {transition['storage_size']:.2e} entries
   Storage Complexity: {transition['storage_complexity']}
   Computation per Transition: {transition['computation_per_transition']}
   Uncertainty Update: {transition['uncertainty_update']}

3. POLICY COMPLEXITY
   Heuristic Policy: {policy['heuristic_policy']['complexity']}
   Time per State: {policy['heuristic_policy']['time_per_state']}
   Total Time: {policy['heuristic_policy']['total_time']}
   
   Value Iteration: {policy['value_iteration']['complexity']}
   Time per Iteration: {policy['value_iteration']['time_per_iteration']}
   Speedup Factor: {policy['speedup_factor']:.2e}x

4. WORST-CASE TIME COMPLEXITY
   Worst Case Questions: {worst_case['worst_case_questions']}
   Per Question Time: {worst_case['per_question_time']} operations
   Decision Time: {worst_case['decision_time']} operations
   Total Time: {worst_case['total_time']} operations
   Complexity: {worst_case['complexity']}
   Practical Time: {worst_case['practical_time']}

5. KEY INSIGHTS
   - State space is exponential in theory but manageable in practice
   - Heuristic policy provides O(1) per state vs O(|S|^2) for value iteration
   - Sparse representation reduces storage from {state_space['theoretical_max']:.2e} to ~{effective_space['effective_states']:.2e} states
   - Worst-case time is linear in number of questions (typically ≤ 6)
"""
        return report


if __name__ == "__main__":
    analyzer = ComplexityAnalyzer()
    report = analyzer.generate_complexity_report()
    print(report)

