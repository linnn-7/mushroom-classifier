"""
MDP (Markov Decision Process) System for Mushroom Identification

This module implements an MDP-based decision system for sequential
mushroom identification, integrating CNN classification with adaptive
question selection.
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Tuple
from enum import Enum
from collections import defaultdict
import numpy as np


class SafetyStatus(Enum):
    """Safety status for mushroom identification"""
    SAFE = "SAFE"
    DANGER = "DANGER"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class MDPState:
    """MDP state representation for mushroom identification"""
    # From CNN
    predicted_class: str  # Output from mushroom_model.pt
    confidence: float  # CNN confidence [0, 1]
    kb_info: Dict  # Knowledge base information
    
    # From user interactions
    features_observed: Set[str] = field(default_factory=set)  # Set of features checked
    answers: Dict[str, str] = field(default_factory=dict)  # User answers to questions
    uncertainty: float = 0.0  # Updated uncertainty [0, 1]
    
    # Decision state
    is_terminal: bool = False
    safety_status: Optional[SafetyStatus] = None
    decision_confidence: Optional[float] = None
    
    def __hash__(self):
        """Make state hashable for MDP algorithms"""
        return hash((
            self.predicted_class,
            round(self.confidence, 2),
            tuple(sorted(self.features_observed)),
            tuple(sorted(self.answers.items())),
            round(self.uncertainty, 2)
        ))
    
    def __eq__(self, other):
        """Equality comparison for states"""
        if not isinstance(other, MDPState):
            return False
        return (self.predicted_class == other.predicted_class and
                abs(self.confidence - other.confidence) < 0.01 and
                self.features_observed == other.features_observed and
                self.answers == other.answers and
                abs(self.uncertainty - other.uncertainty) < 0.01)


class Action(Enum):
    """MDP actions for mushroom identification"""
    ASK_SPORE_PRINT = "ask_spore_print"
    ASK_VOLVA = "ask_volva"
    ASK_GILL_COLOR = "ask_gill_color"
    ASK_HABITAT = "ask_habitat"
    ASK_BRUISING = "ask_bruising"
    ASK_ODOR = "ask_odor"
    REQUEST_ADDITIONAL_IMAGE = "request_additional_image"
    MAKE_DECISION = "make_decision"  # Terminal action
    
    def get_question_text(self):
        """Get human-readable question for action"""
        questions = {
            Action.ASK_SPORE_PRINT: "What color is the spore print? (Place the cap gill-side down on paper for 6-12 hours)",
            Action.ASK_VOLVA: "Is there a volva (cup-like base) at the stem base, possibly under the soil?",
            Action.ASK_GILL_COLOR: "What color are the gills?",
            Action.ASK_HABITAT: "Where is the mushroom growing?",
            Action.ASK_BRUISING: "Does the mushroom change color when bruised?",
            Action.ASK_ODOR: "What does the mushroom smell like?",
        }
        return questions.get(self, "")


class TransitionModel:
    """Models state transition probabilities P(s_{t+1} | s_t, a_t)"""
    
    def __init__(self):
        # Learned from domain knowledge or historical data
        self.transition_probs = self._initialize_transitions()
    
    def _initialize_transitions(self) -> Dict:
        """
        Initialize transition probabilities based on mycological knowledge
        
        Returns:
            Dict mapping (predicted_class, action, answer) -> transition_info
        """
        transitions = {}
        
        # Example: Agaricus + Ask Volva + "No" → High confidence Agaricus
        transitions[("Agaricus", Action.ASK_VOLVA, "No")] = {
            "confidence_increase": 0.15,
            "uncertainty_decrease": 0.2,
            "probability": 0.8
        }
        
        # Example: Agaricus + Ask Volva + "Yes" → Danger (Amanita)
        transitions[("Agaricus", Action.ASK_VOLVA, "Yes")] = {
            "safety_status": SafetyStatus.DANGER,
            "probability": 0.9
        }
        
        # Example: Agaricus + Ask Spore Print + "Dark Brown" → Confirm Agaricus
        transitions[("Agaricus", Action.ASK_SPORE_PRINT, "Dark Brown / Chocolate")] = {
            "confidence_increase": 0.1,
            "uncertainty_decrease": 0.15,
            "probability": 0.7
        }
        
        # Example: Agaricus + Ask Spore Print + "White" → Danger (Amanita)
        transitions[("Agaricus", Action.ASK_SPORE_PRINT, "White")] = {
            "safety_status": SafetyStatus.DANGER,
            "probability": 0.85
        }
        
        # Example: Agaricus + Ask Spore Print + "I don't know" → Slight uncertainty decrease
        transitions[("Agaricus", Action.ASK_SPORE_PRINT, "I don't know")] = {
            "uncertainty_decrease": 0.05,
            "probability": 0.5
        }
        
        # Example: Agaricus + Ask Volva + "I don't know" → Slight uncertainty decrease
        transitions[("Agaricus", Action.ASK_VOLVA, "I don't know")] = {
            "uncertainty_decrease": 0.05,
            "probability": 0.5
        }
        
        return transitions
    
    def get_transition(self, state: MDPState, action: Action, answer: str) -> Tuple[MDPState, float]:
        """
        Get next state and transition probability
        
        Args:
            state: Current state
            action: Action taken
            answer: User's answer
        
        Returns:
            Tuple of (next_state, probability)
        """
        key = (state.predicted_class, action, answer)
        transition_info = self.transition_probs.get(key, {})
        
        # Create next state
        next_state = MDPState(
            predicted_class=state.predicted_class,
            confidence=min(1.0, state.confidence + transition_info.get("confidence_increase", 0)),
            kb_info=state.kb_info,
            features_observed=state.features_observed | {action.value},
            answers={**state.answers, action.value: answer},
            uncertainty=max(0.0, state.uncertainty - transition_info.get("uncertainty_decrease", 0))
        )
        
        # Update safety status if transition indicates danger
        if "safety_status" in transition_info:
            next_state.safety_status = transition_info["safety_status"]
            next_state.is_terminal = True
        
        probability = transition_info.get("probability", 0.5)  # Default uncertainty
        
        return next_state, probability


class RewardModel:
    """Defines reward function R(s, a) for MDP"""
    
    # Reward constants
    REWARD_CORRECT_ID = 100
    REWARD_SAFETY = 50
    REWARD_HIGH_CONFIDENCE = 10
    COST_QUESTION = -1
    COST_UNNECESSARY_QUESTION = -2
    PENALTY_FALSE_NEGATIVE_DEADLY = -1000
    PENALTY_FALSE_POSITIVE_EDIBLE = -100
    PENALTY_USER_HARM = -10000
    
    def compute_reward(self, state: MDPState, action: Action, next_state: MDPState, 
                      ground_truth: Optional[str] = None) -> float:
        """
        Compute reward for taking action in state
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            ground_truth: True class (if known, for training)
        
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # Cost of asking questions
        if action != Action.MAKE_DECISION:
            reward += self.COST_QUESTION
            
            # Extra cost if question doesn't reduce uncertainty
            if next_state.uncertainty >= state.uncertainty:
                reward += self.COST_UNNECESSARY_QUESTION
        
        # Terminal state rewards
        if next_state.is_terminal:
            # High confidence decision
            if next_state.decision_confidence and next_state.decision_confidence > 0.8:
                reward += self.REWARD_HIGH_CONFIDENCE
            
            # Safety preservation
            if next_state.safety_status == SafetyStatus.SAFE:
                reward += self.REWARD_SAFETY
            elif next_state.safety_status == SafetyStatus.DANGER:
                # Preventing harm is highly rewarded
                reward += self.REWARD_SAFETY * 2
            
            # Correct identification (if ground truth available)
            if ground_truth:
                if next_state.predicted_class == ground_truth:
                    reward += self.REWARD_CORRECT_ID
                else:
                    # Penalty for wrong identification
                    if ground_truth == "Amanita" and next_state.safety_status != SafetyStatus.DANGER:
                        reward += self.PENALTY_FALSE_NEGATIVE_DEADLY
                    else:
                        reward += self.PENALTY_FALSE_POSITIVE_EDIBLE
        
        return reward


class MDPPolicy:
    """MDP policy for action selection"""
    
    def __init__(self, transition_model: TransitionModel, reward_model: RewardModel,
                 gamma: float = 0.9, epsilon: float = 0.1):
        """
        Initialize policy
        
        Args:
            transition_model: Transition probability model
            reward_model: Reward function
            gamma: Discount factor
            epsilon: Exploration rate (for epsilon-greedy)
        """
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Value function V(s)
        self.value_function = defaultdict(float)
        
        # Policy π(a | s) - default to asking most critical questions first
        self.policy = defaultdict(lambda: Action.ASK_VOLVA)  # Default: ask about volva
    
    def select_action(self, state: MDPState, available_actions: Optional[list] = None) -> Action:
        """
        Select action using heuristic policy (simplified for now)
        
        For full implementation, would use value iteration. For now, use
        heuristic-based policy that prioritizes safety-critical questions.
        
        Args:
            state: Current state
            available_actions: List of available actions (if None, use all)
        
        Returns:
            Action to take
        """
        if state.is_terminal:
            return Action.MAKE_DECISION
        
        # Determine available actions
        if available_actions is None:
            available_actions = [a for a in Action if a != Action.MAKE_DECISION]
        
        # Remove already asked questions
        available_actions = [a for a in available_actions if a.value not in state.features_observed]
        
        if not available_actions:
            return Action.MAKE_DECISION
        
        # Heuristic policy: prioritize safety-critical questions
        # For Agaricus, volva and spore print are most critical
        if state.predicted_class == "Agaricus":
            # Priority 1: Volva (most critical for safety)
            if Action.ASK_VOLVA in available_actions:
                return Action.ASK_VOLVA
            # Priority 2: Spore print (critical for distinguishing from Amanita)
            if Action.ASK_SPORE_PRINT in available_actions:
                return Action.ASK_SPORE_PRINT
            # Priority 3: Habitat (important for identification)
            if Action.ASK_HABITAT in available_actions:
                return Action.ASK_HABITAT
            # Priority 4: Other features
            if Action.ASK_GILL_COLOR in available_actions:
                return Action.ASK_GILL_COLOR
            if Action.ASK_BRUISING in available_actions:
                return Action.ASK_BRUISING
            if Action.ASK_ODOR in available_actions:
                return Action.ASK_ODOR
        
        # For other classes, prioritize habitat and other informative questions
        else:
            # Priority 1: Habitat (very informative for most mushrooms)
            if Action.ASK_HABITAT in available_actions:
                return Action.ASK_HABITAT
            # Priority 2: Gill color (important for many genera)
            if Action.ASK_GILL_COLOR in available_actions:
                return Action.ASK_GILL_COLOR
            # Priority 3: Spore print (important for identification)
            if Action.ASK_SPORE_PRINT in available_actions:
                return Action.ASK_SPORE_PRINT
            # Priority 4: Other features
            if Action.ASK_VOLVA in available_actions:
                return Action.ASK_VOLVA
            if Action.ASK_BRUISING in available_actions:
                return Action.ASK_BRUISING
            if Action.ASK_ODOR in available_actions:
                return Action.ASK_ODOR
        
        # Only make decision if we have asked at least 4 questions AND uncertainty is very low
        # This ensures we ask about habitat and other features
        if len(state.features_observed) >= 4 and state.uncertainty < 0.15:
            return Action.MAKE_DECISION
        
        # If we've asked all available questions, make decision
        if len(available_actions) == 0:
            return Action.MAKE_DECISION
        
        # Otherwise, ask any remaining available question
        return available_actions[0] if available_actions else Action.MAKE_DECISION


def create_initial_state(pred_class: str, confidence: float, kb_info: Dict) -> MDPState:
    """
    Create initial MDP state from CNN classification
    
    Args:
        pred_class: Predicted class from CNN
        confidence: CNN confidence score
        kb_info: Knowledge base information
    
    Returns:
        MDPState: Initial state s_0
    """
    s_0 = MDPState(
        predicted_class=pred_class,
        confidence=confidence,
        kb_info=kb_info,
        features_observed=set(),
        answers={},
        uncertainty=1.0 - confidence  # High uncertainty if low confidence
    )
    
    return s_0


def make_final_decision(state: MDPState) -> MDPState:
    """
    Make final decision based on current state, considering conflicting evidence
    
    Args:
        state: Current MDP state
    
    Returns:
        Updated state with final decision
    """
    state.is_terminal = True
    
    # Start with base confidence
    decision_confidence = state.confidence
    conflicts = 0
    critical_conflicts = 0
    
    # Decision logic based on state
    if state.predicted_class == "Agaricus":
        # Check for critical danger signals
        volva_answer = state.answers.get("ask_volva", "")
        spore_answer = state.answers.get("ask_spore_print", "")
        gill_answer = state.answers.get("ask_gill_color", "")
        habitat_answer = state.answers.get("ask_habitat", "")
        odor_answer = state.answers.get("ask_odor", "")
        
        # Critical danger signals (override everything)
        if "Yes" in volva_answer or volva_answer == "Yes":
            state.safety_status = SafetyStatus.DANGER
            state.decision_confidence = 0.1  # Very low confidence
            return state
        if spore_answer == "White":
            state.safety_status = SafetyStatus.DANGER
            state.decision_confidence = 0.1  # Very low confidence
            return state
        if gill_answer == "White":
            state.safety_status = SafetyStatus.DANGER
            state.decision_confidence = 0.1  # Very low confidence
            return state
        
        # Count conflicts (unusual features for Agaricus)
        if habitat_answer == "On wood":
            conflicts += 1
            critical_conflicts += 1  # Habitat is a strong indicator
        
        if gill_answer not in ["I don't know", ""] and gill_answer not in ["Pink", "Brown", "Dark Brown / Chocolate"]:
            conflicts += 1
            if gill_answer in ["Yellow", "Black"]:
                critical_conflicts += 1  # Yellow/black gills are very unusual for Agaricus
        
        if odor_answer == "Foul/rotten":
            conflicts += 1
            critical_conflicts += 1  # Foul odor is very unusual for Agaricus
        
        if state.answers.get("ask_bruising", "") == "Yes, it changes color":
            conflicts += 0.5  # Bruising is less critical but still unusual
        
        # Adjust confidence based on conflicts
        if critical_conflicts >= 2:
            # Multiple critical conflicts - very uncertain
            decision_confidence = max(0.2, state.confidence - 0.5)
            state.safety_status = SafetyStatus.UNCERTAIN
        elif critical_conflicts >= 1 or conflicts >= 2:
            # At least one critical conflict or multiple conflicts - reduce confidence significantly
            decision_confidence = max(0.3, state.confidence - 0.4)
            state.safety_status = SafetyStatus.UNCERTAIN
        elif conflicts >= 1:
            # Some conflicts - reduce confidence moderately
            decision_confidence = max(0.4, state.confidence - 0.2)
            state.safety_status = SafetyStatus.UNCERTAIN
        elif (state.confidence > 0.7 and 
              spore_answer == "Dark Brown / Chocolate" and
              ("No" in volva_answer or volva_answer == "No") and
              habitat_answer in ["On ground/soil", "I don't know", ""] and
              gill_answer in ["Pink", "Brown", "Dark Brown / Chocolate", "I don't know", ""]):
            # Strong supporting evidence with no conflicts
            state.safety_status = SafetyStatus.SAFE
            decision_confidence = min(0.95, state.confidence + 0.1)  # Boost confidence slightly
        else:
            # Default to uncertain if we don't have strong evidence
            state.safety_status = SafetyStatus.UNCERTAIN
            decision_confidence = max(0.3, state.confidence - 0.2)
    
    elif state.predicted_class == "Amanita":
        # Always dangerous
        state.safety_status = SafetyStatus.DANGER
        decision_confidence = 0.9  # High confidence in danger
    elif state.kb_info and state.kb_info.get('edibility') in ["Deadly Poisonous", "Varies (Some Deadly)"]:
        # High risk classes
        state.safety_status = SafetyStatus.DANGER
        decision_confidence = state.confidence
    else:
        # For other classes, check for conflicts and use confidence threshold
        # Check for habitat conflicts (if applicable)
        habitat_answer = state.answers.get("ask_habitat", "")
        if habitat_answer != "I don't know" and habitat_answer != "":
            # For Boletus/Suillus, wood habitat is unusual
            if state.predicted_class in ["Boletus", "Suillus"] and habitat_answer == "On wood":
                conflicts += 1
        
        # Adjust confidence based on conflicts
        if conflicts >= 1:
            decision_confidence = max(0.3, state.confidence - 0.3)
            state.safety_status = SafetyStatus.UNCERTAIN
        elif state.confidence > 0.8:
            state.safety_status = SafetyStatus.SAFE
            decision_confidence = state.confidence
        else:
            state.safety_status = SafetyStatus.UNCERTAIN
            decision_confidence = state.confidence
    
    state.decision_confidence = decision_confidence
    
    return state

