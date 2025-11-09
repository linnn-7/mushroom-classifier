"""
Conflict Detection Analysis

This script analyzes the effectiveness of conflict detection in preventing
false positives and adjusting confidence appropriately.
"""

import numpy as np
from typing import Dict, List, Tuple
from mdp_system import MDPState, SafetyStatus, create_initial_state, make_final_decision


class ConflictAnalysis:
    """Analyze conflict detection effectiveness"""
    
    def __init__(self):
        self.test_scenarios = []
    
    def create_conflict_scenarios(self) -> List[Dict]:
        """
        Create test scenarios with varying levels of conflict
        
        Returns:
            List of scenarios with conflict counts and expected outcomes
        """
        scenarios = []
        
        # Scenario 1: No conflicts (should maintain/high confidence)
        scenarios.append({
            'name': 'No_Conflicts',
            'predicted_class': 'Agaricus',
            'initial_confidence': 0.85,
            'answers': {
                'ask_volva': 'No',
                'ask_spore_print': 'Dark Brown / Chocolate',
                'ask_habitat': 'On ground/soil',
                'ask_gill_color': 'Pink',
                'ask_bruising': 'No, no color change',
                'ask_odor': 'Anise/licorice'
            },
            'expected_conflicts': 0,
            'expected_critical_conflicts': 0,
            'expected_confidence_range': (0.85, 0.95),
            'expected_safety': SafetyStatus.SAFE
        })
        
        # Scenario 2: 1 critical conflict (should reduce confidence moderately)
        scenarios.append({
            'name': 'One_Critical_Conflict',
            'predicted_class': 'Agaricus',
            'initial_confidence': 0.85,
            'answers': {
                'ask_volva': 'No',
                'ask_spore_print': 'Dark Brown / Chocolate',
                'ask_habitat': 'On wood',  # Critical conflict
                'ask_gill_color': 'Pink',
                'ask_bruising': 'No, no color change',
                'ask_odor': 'Anise/licorice'
            },
            'expected_conflicts': 1,
            'expected_critical_conflicts': 1,
            'expected_confidence_range': (0.3, 0.5),
            'expected_safety': SafetyStatus.UNCERTAIN
        })
        
        # Scenario 3: 2 critical conflicts (should reduce confidence significantly)
        scenarios.append({
            'name': 'Two_Critical_Conflicts',
            'predicted_class': 'Agaricus',
            'initial_confidence': 0.85,
            'answers': {
                'ask_volva': 'No',
                'ask_spore_print': 'Dark Brown / Chocolate',
                'ask_habitat': 'On wood',  # Critical conflict
                'ask_gill_color': 'Yellow',  # Critical conflict
                'ask_bruising': 'No, no color change',
                'ask_odor': 'Anise/licorice'
            },
            'expected_conflicts': 2,
            'expected_critical_conflicts': 2,
            'expected_confidence_range': (0.2, 0.35),
            'expected_safety': SafetyStatus.UNCERTAIN
        })
        
        # Scenario 4: 3 critical conflicts (should reduce confidence drastically)
        scenarios.append({
            'name': 'Three_Critical_Conflicts',
            'predicted_class': 'Agaricus',
            'initial_confidence': 0.85,
            'answers': {
                'ask_volva': 'No',
                'ask_spore_print': 'Dark Brown / Chocolate',
                'ask_habitat': 'On wood',  # Critical conflict
                'ask_gill_color': 'Yellow',  # Critical conflict
                'ask_bruising': 'Yes, it changes color',  # Minor conflict
                'ask_odor': 'Foul/rotten'  # Critical conflict
            },
            'expected_conflicts': 3.5,
            'expected_critical_conflicts': 3,
            'expected_confidence_range': (0.2, 0.35),
            'expected_safety': SafetyStatus.UNCERTAIN
        })
        
        # Scenario 5: Critical danger signal (should override to DANGER)
        scenarios.append({
            'name': 'Critical_Danger_Signal',
            'predicted_class': 'Agaricus',
            'initial_confidence': 0.90,
            'answers': {
                'ask_volva': 'Yes',  # Critical danger
                'ask_spore_print': 'Dark Brown / Chocolate',
                'ask_habitat': 'On ground/soil',
                'ask_gill_color': 'Pink',
                'ask_bruising': 'No, no color change',
                'ask_odor': 'Anise/licorice'
            },
            'expected_conflicts': 0,
            'expected_critical_conflicts': 0,
            'expected_confidence_range': (0.1, 0.1),
            'expected_safety': SafetyStatus.DANGER
        })
        
        return scenarios
    
    def analyze_conflict_detection(self, scenarios: List[Dict]) -> Dict:
        """
        Analyze conflict detection effectiveness
        
        Returns:
            Dictionary with analysis results
        """
        results = {
            'scenarios': [],
            'conflict_accuracy': 0,
            'confidence_adjustment_accuracy': 0,
            'safety_status_accuracy': 0
        }
        
        for scenario in scenarios:
            # Create initial state
            kb_info = {}  # Would load from knowledge base
            initial_state = create_initial_state(
                scenario['predicted_class'],
                scenario['initial_confidence'],
                kb_info
            )
            
            # Simulate transitions (simplified - would use actual transition model)
            current_state = initial_state
            for action_value, answer in scenario['answers'].items():
                # Update state with answer (simplified)
                current_state.answers[action_value] = answer
                current_state.features_observed.add(action_value)
            
            # Make final decision
            final_state = make_final_decision(current_state)
            
            # Analyze results
            scenario_result = {
                'name': scenario['name'],
                'initial_confidence': scenario['initial_confidence'],
                'final_confidence': final_state.decision_confidence,
                'confidence_reduction': scenario['initial_confidence'] - final_state.decision_confidence,
                'expected_range': scenario['expected_confidence_range'],
                'in_range': (scenario['expected_confidence_range'][0] <= final_state.decision_confidence <= 
                           scenario['expected_confidence_range'][1]),
                'safety_status': final_state.safety_status,
                'expected_safety': scenario['expected_safety'],
                'safety_correct': (final_state.safety_status == scenario['expected_safety'])
            }
            
            results['scenarios'].append(scenario_result)
            
            if scenario_result['safety_correct']:
                results['safety_status_accuracy'] += 1
            if scenario_result['in_range']:
                results['confidence_adjustment_accuracy'] += 1
        
        results['safety_status_accuracy'] /= len(scenarios)
        results['confidence_adjustment_accuracy'] /= len(scenarios)
        
        return results
    
    def generate_conflict_statistics(self, scenarios: List[Dict]) -> Dict:
        """
        Generate statistics on conflict detection
        
        Returns:
            Dictionary with conflict statistics
        """
        stats = {
            'total_scenarios': len(scenarios),
            'conflict_distribution': defaultdict(int),
            'confidence_reductions': [],
            'safety_status_distribution': defaultdict(int)
        }
        
        for scenario in scenarios:
            # Count conflicts
            conflicts = 0
            critical_conflicts = 0
            
            if scenario['predicted_class'] == 'Agaricus':
                if scenario['answers'].get('ask_habitat') == 'On wood':
                    conflicts += 1
                    critical_conflicts += 1
                if scenario['answers'].get('ask_gill_color') in ['Yellow', 'Black']:
                    conflicts += 1
                    critical_conflicts += 1
                if scenario['answers'].get('ask_odor') == 'Foul/rotten':
                    conflicts += 1
                    critical_conflicts += 1
                if scenario['answers'].get('ask_bruising') == 'Yes, it changes color':
                    conflicts += 0.5
            
            stats['conflict_distribution'][critical_conflicts] += 1
            stats['safety_status_distribution'][scenario['expected_safety']] += 1
        
        return stats


if __name__ == "__main__":
    analyzer = ConflictAnalysis()
    scenarios = analyzer.create_conflict_scenarios()
    results = analyzer.analyze_conflict_detection(scenarios)
    stats = analyzer.generate_conflict_statistics(scenarios)
    
    print("Conflict Detection Analysis")
    print("=" * 60)
    print(f"\nSafety Status Accuracy: {results['safety_status_accuracy']:.3f}")
    print(f"Confidence Adjustment Accuracy: {results['confidence_adjustment_accuracy']:.3f}")
    
    print("\nScenario Results:")
    for scenario_result in results['scenarios']:
        print(f"\n{scenario_result['name']}:")
        print(f"  Initial Confidence: {scenario_result['initial_confidence']:.3f}")
        print(f"  Final Confidence: {scenario_result['final_confidence']:.3f}")
        print(f"  Confidence Reduction: {scenario_result['confidence_reduction']:.3f}")
        print(f"  Safety Status: {scenario_result['safety_status']}")
        print(f"  Safety Correct: {scenario_result['safety_correct']}")

