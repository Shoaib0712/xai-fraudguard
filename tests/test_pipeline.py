import pytest
import pandas as pd
import numpy as np
from predict import calculate_risk_score, evaluate_rule_engine, compute_hybrid_risk, classify_risk, determine_decision

def test_risk_score_conversion():
    assert calculate_risk_score(0.25) == 25.0
    assert calculate_risk_score(1.0) == 100.0
    assert calculate_risk_score(0.0) == 0.0

def test_rule_engine_triggers():
    rules = evaluate_rule_engine(1500.0, 80.0, 0.6)
    triggered_rules = [r for r in rules if r['Status'] == 'TRIGGERED']
    assert len(triggered_rules) == 3

def test_hybrid_risk_bounds():
    score = compute_hybrid_risk(0.9, 90.0, 100.0)
    assert 0.0 <= score <= 100.0

def test_three_way_decisions():
    assert classify_risk(15.0) == "LOW"
    assert determine_decision("LOW") == "APPROVE"
    
    assert classify_risk(50.0) == "MEDIUM"
    assert determine_decision("MEDIUM") == "REVIEW"
    
    assert classify_risk(95.0) == "CRITICAL"
    assert determine_decision("CRITICAL") == "BLOCK"