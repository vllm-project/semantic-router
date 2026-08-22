# complete code
"""
Evaluation of query-difficulty and model-budget ability estimates using RADAR-inspired formulation.
"""
import numpy as np
from sklearn.metrics import accuracy_score

def evaluate_difficulty(model_budget, difficulty, y_true, y_pred):
    # Evaluate query-difficulty estimates
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def evaluate_model_budget(model_budget, difficulty, y_true, y_pred):
    # Evaluate model-budget ability estimates
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy