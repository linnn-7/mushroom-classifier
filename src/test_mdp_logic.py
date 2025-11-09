"""
Test MDP Logic Without Images

This script tests the MDP logic directly without requiring actual images,
to verify the numbers reported in the paper.
"""

from mdp_system import (
    MDPState, Action, SafetyStatus, TransitionModel, RewardModel, MDPPolicy,
    create_initial_state, make_final_decision
)


def test_conflict_detection():
    """Test conflict detection scenarios"""
    print("Testing Conflict Detection Scenarios")
    print("=" * 60)
    
    # Test Case 1: Agaricus with conflicting evidence
    kb_info = {}
    state1 = create_initial_state("Agaricus", 0.85, kb_info)
    state1.answers = {
        "ask_volva": "No",
        "ask_spore_print": "Dark Brown / Chocolate",
        "ask_habitat": "On wood",  # Conflict
        "ask_gill_color": "Yellow",  # Conflict
        "ask_bruising": "Yes, it changes color",  # Conflict
        "ask_odor": "Foul/rotten"  # Conflict
    }
    state1.features_observed = set(state1.answers.keys())
    final1 = make_final_decision(state1)
    print(f"\nTest 1: Agaricus with 3 critical conflicts")
    print(f"  Initial confidence: {state1.confidence:.2f}")
    print(f"  Final confidence: {final1.decision_confidence:.2f}")
    print(f"  Safety status: {final1.safety_status}")
    print(f"  Confidence reduction: {state1.confidence - final1.decision_confidence:.2f}")
    
    # Test Case 2: Agaricus with supporting evidence
    state2 = create_initial_state("Agaricus", 0.78, kb_info)
    state2.answers = {
        "ask_volva": "No",
        "ask_spore_print": "Dark Brown / Chocolate",
        "ask_habitat": "On ground/soil",
        "ask_gill_color": "Pink",
        "ask_bruising": "No, no color change",
        "ask_odor": "Anise/licorice"
    }
    state2.features_observed = set(state2.answers.keys())
    final2 = make_final_decision(state2)
    print(f"\nTest 2: Agaricus with supporting evidence")
    print(f"  Initial confidence: {state2.confidence:.2f}")
    print(f"  Final confidence: {final2.decision_confidence:.2f}")
    print(f"  Safety status: {final2.safety_status}")
    print(f"  Confidence change: {final2.decision_confidence - state2.confidence:.2f}")
    
    # Test Case 3: Agaricus with 1 critical conflict
    state3 = create_initial_state("Agaricus", 0.80, kb_info)
    state3.answers = {
        "ask_volva": "No",
        "ask_spore_print": "Dark Brown / Chocolate",
        "ask_habitat": "On wood",  # Critical conflict
        "ask_gill_color": "Pink",
        "ask_bruising": "No, no color change",
        "ask_odor": "Anise/licorice"
    }
    state3.features_observed = set(state3.answers.keys())
    final3 = make_final_decision(state3)
    print(f"\nTest 3: Agaricus with 1 critical conflict (wood habitat)")
    print(f"  Initial confidence: {state3.confidence:.2f}")
    print(f"  Final confidence: {final3.decision_confidence:.2f}")
    print(f"  Safety status: {final3.safety_status}")
    print(f"  Confidence reduction: {state3.confidence - final3.decision_confidence:.2f}")
    
    # Test Case 4: Critical danger signal (volva = Yes)
    state4 = create_initial_state("Agaricus", 0.90, kb_info)
    state4.answers = {
        "ask_volva": "Yes",  # Critical danger
        "ask_spore_print": "Dark Brown / Chocolate",
        "ask_habitat": "On ground/soil",
        "ask_gill_color": "Pink",
        "ask_bruising": "No, no color change",
        "ask_odor": "Anise/licorice"
    }
    state4.features_observed = set(state4.answers.keys())
    final4 = make_final_decision(state4)
    print(f"\nTest 4: Critical danger signal (volva = Yes)")
    print(f"  Initial confidence: {state4.confidence:.2f}")
    print(f"  Final confidence: {final4.decision_confidence:.2f}")
    print(f"  Safety status: {final4.safety_status}")
    
    # Test Case 5: Amanita (should be DANGER)
    state5 = create_initial_state("Amanita", 0.75, kb_info)
    state5.answers = {
        "ask_volva": "Yes",
        "ask_spore_print": "White",
        "ask_habitat": "On ground/soil",
        "ask_gill_color": "White",
        "ask_bruising": "No, no color change",
        "ask_odor": "No distinct odor"
    }
    state5.features_observed = set(state5.answers.keys())
    final5 = make_final_decision(state5)
    print(f"\nTest 5: Amanita (should be DANGER)")
    print(f"  Initial confidence: {state5.confidence:.2f}")
    print(f"  Final confidence: {final5.decision_confidence:.2f}")
    print(f"  Safety status: {final5.safety_status}")
    
    return {
        'test1': (state1.confidence, final1.decision_confidence, final1.safety_status),
        'test2': (state2.confidence, final2.decision_confidence, final2.safety_status),
        'test3': (state3.confidence, final3.decision_confidence, final3.safety_status),
        'test4': (state4.confidence, final4.decision_confidence, final4.safety_status),
        'test5': (state5.confidence, final5.decision_confidence, final5.safety_status),
    }


def test_policy_termination():
    """Test policy termination conditions"""
    print("\n\nTesting Policy Termination Conditions")
    print("=" * 60)
    
    transition_model = TransitionModel()
    reward_model = RewardModel()
    policy = MDPPolicy(transition_model, reward_model)
    
    # Test termination condition: at least 4 features and uncertainty < 0.15
    kb_info = {}
    state = create_initial_state("Agaricus", 0.90, kb_info)  # High confidence = low uncertainty
    state.answers = {
        "ask_volva": "No",
        "ask_spore_print": "Dark Brown / Chocolate",
        "ask_habitat": "On ground/soil",
        "ask_gill_color": "Pink",
    }
    state.features_observed = set(state.answers.keys())
    state.uncertainty = 0.10  # Low uncertainty
    
    action = policy.select_action(state)
    print(f"\nState with 4 features observed, uncertainty = {state.uncertainty:.2f}")
    print(f"  Selected action: {action}")
    print(f"  Should terminate: {action == Action.MAKE_DECISION}")
    
    # Test with only 3 features
    state2 = create_initial_state("Agaricus", 0.85, kb_info)
    state2.answers = {
        "ask_volva": "No",
        "ask_spore_print": "Dark Brown / Chocolate",
        "ask_habitat": "On ground/soil",
    }
    state2.features_observed = set(state2.answers.keys())
    state2.uncertainty = 0.10
    
    action2 = policy.select_action(state2)
    print(f"\nState with 3 features observed, uncertainty = {state2.uncertainty:.2f}")
    print(f"  Selected action: {action2}")
    print(f"  Should continue: {action2 != Action.MAKE_DECISION}")


if __name__ == "__main__":
    results = test_conflict_detection()
    test_policy_termination()
    
    print("\n\nSummary of Conflict Detection Results:")
    print("=" * 60)
    for test_name, (init_conf, final_conf, safety) in results.items():
        print(f"{test_name}: {init_conf:.2f} -> {final_conf:.2f} ({safety})")

