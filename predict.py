import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
import shap
import datetime
import networkx as nx

# ==========================================
# 🤖 MULTI-AGENT SYSTEM (MAS) DEFINITIONS
# ==========================================

class BaseFraudAgent:
    def __init__(self, name, role):
        self.name = name
        self.role = role

    def log_action(self, action_details):
        return f"[{self.name} | {self.role}] {action_details}"


class IngestionAnomalyAgent(BaseFraudAgent):
    """Agent 1: Handles data normalization and unsupervised behavioral anomaly detection."""
    def __init__(self):
        super().__init__(name="Agent-Alpha", role="Ingestion & Anomaly Specialist")

    def detect_anomalies(self, X_data):
        iso = IsolationForest(contamination=0.01, random_state=42)
        iso.fit(X_data)
        raw_scores = iso.decision_function(X_data)
        normalized = 100 * (1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8))
        return np.round(normalized, 2)


class RiskEvaluationAgent(BaseFraudAgent):
    """Agent 2: Evaluates compliance rules, combines probabilities, and computes hybrid risk scores."""
    def __init__(self):
        super().__init__(name="Agent-Beta", role="Rule & Risk Decision Engine")

    def evaluate_rules(self, row_amount, anomaly_score, ml_prob):
        rules = []
        if row_amount > 1000.0:
            rules.append({
                "Rule": "High Amount Threshold",
                "Condition": f"Amount (${row_amount:,.2f}) > $1,000.00",
                "Risk Contribution": 25.0,
                "Status": "TRIGGERED"
            })
        else:
            rules.append({
                "Rule": "High Amount Threshold",
                "Condition": f"Amount (${row_amount:,.2f}) <= $1,000.00",
                "Risk Contribution": 0.0,
                "Status": "PASSED"
            })
            
        if anomaly_score > 75.0:
            rules.append({
                "Rule": "Behavioral Outlier Isolation",
                "Condition": f"Anomaly Score ({anomaly_score:.1f}) > 75.0",
                "Risk Contribution": 30.0,
                "Status": "TRIGGERED"
            })
        else:
            rules.append({
                "Rule": "Behavioral Outlier Isolation",
                "Condition": f"Anomaly Score ({anomaly_score:.1f}) <= 75.0",
                "Risk Contribution": 0.0,
                "Status": "PASSED"
            })
            
        if ml_prob > 0.5:
            rules.append({
                "Rule": "Supervised Fraud Classifier Alert",
                "Condition": f"Fraud Probability ({ml_prob*100:.1f}%) > 50%",
                "Risk Contribution": 45.0,
                "Status": "TRIGGERED"
            })
        else:
            rules.append({
                "Rule": "Supervised Fraud Classifier Alert",
                "Condition": f"Fraud Probability ({ml_prob*100:.1f}%) <= 50%",
                "Risk Contribution": 0.0,
                "Status": "PASSED"
            })
        return rules

    def compute_hybrid_score(self, ml_prob, anomaly_score, rule_risk_sum):
        ml_component = ml_prob * 100 * 0.45
        anomaly_component = anomaly_score * 0.25
        rule_component = min(rule_risk_sum, 100.0) * 0.30
        final_score = round(ml_component + anomaly_component + rule_component, 2)
        return min(max(final_score, 0.0), 100.0)

    def classify_risk_tier(self, risk_score):
        if risk_score < 30.0:
            return "LOW"
        elif risk_score < 70.0:
            return "MEDIUM"
        elif risk_score < 90.0:
            return "HIGH"
        else:
            return "CRITICAL"

    def determine_final_decision(self, risk_level):
        if risk_level in ["LOW"]:
            return "APPROVE"
        elif risk_level in ["MEDIUM", "HIGH"]:
            return "REVIEW"
        else:
            return "BLOCK"


class ExplainabilityAgent(BaseFraudAgent):
    """Agent 3: Generates SHAP explainability attributions, counterfactuals, and natural language narratives."""
    def __init__(self):
        super().__init__(name="Agent-Gamma", role="XAI & Natural Language Narrator")

    def explain_shap(self, model, background_df, sample_row_df):
        explainer = shap.Explainer(model.predict_proba, background_df)
        return explainer(sample_row_df)

    def generate_counterfactual(self, model, scaler, sample_row):
        cf = sample_row.copy()
        if 'Amount' in cf.columns:
            original_amount_scaled = cf['Amount'].values[0]
            cf['Amount'] = original_amount_scaled * 0.2
            new_prob = model.predict_proba(cf)[0, 1]
            new_risk = round(float(new_prob) * 100, 2)
            return cf, new_risk
        return cf, round(float(model.predict_proba(sample_row)[0, 1]) * 100, 2)

    def create_narrative(self, risk_level, decision, amount, top_feature):
        return (
            f"**[Agent-Gamma Narrative]** Transaction evaluated as **{risk_level} risk** resulting in an immediate "
            f"**{decision}** action. The transaction amount of **${amount:,.2f}** combined with "
            f"behavioral anomaly indicators (notably contributing via `{top_feature}`) deviated substantially "
            f"from historical baseline distributions."
        )


class SyndicateIntelligenceAgent(BaseFraudAgent):
    """Agent 4: Maps network-level connections and card-testing syndicates using graph theory."""
    def __init__(self):
        super().__init__(name="Agent-Delta", role="Graph Syndicate Investigator")

    def build_fraud_ring_network(self, df_subset):
        G = nx.Graph()
        high_risk = df_subset[df_subset['Hybrid_Risk_Score'] > 40].head(30)
        if len(high_risk) < 5:
            high_risk = df_subset.head(30)
        
        for idx, row in high_risk.iterrows():
            node_id = f"Txn #{int(row['index'] if 'index' in row else idx)}"
            G.add_node(node_id, amount=row['Amount'], risk=row['Hybrid_Risk_Score'])
            
        nodes = list(G.nodes(data=True))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                amt_diff = abs(nodes[i][1]['amount'] - nodes[j][1]['amount'])
                if amt_diff < 500.0:
                    G.add_edge(nodes[i][0], nodes[j][0], weight=amt_diff)
        return G


# ==========================================
# 🔗 AGENT ORCHESTRATOR & BACKWARD-COMPATIBLE WRAPPERS
# ==========================================

# Instantiate Agent Ecosystem
anomaly_agent = IngestionAnomalyAgent()
risk_agent = RiskEvaluationAgent()
xai_agent = ExplainabilityAgent()
syndicate_agent = SyndicateIntelligenceAgent()

def calculate_risk_score(probability):
    return round(float(probability) * 100, 2)

def evaluate_rule_engine(row_amount, anomaly_score, ml_prob):
    return risk_agent.evaluate_rules(row_amount, anomaly_score, ml_prob)

def compute_hybrid_risk(ml_prob, anomaly_score, rule_risk_sum):
    return risk_agent.compute_hybrid_score(ml_prob, anomaly_score, rule_risk_sum)

def classify_risk(risk_score):
    return risk_agent.classify_risk_tier(risk_score)

def determine_decision(risk_level):
    return risk_agent.determine_final_decision(risk_level)

def run_anomaly_detection_scores(X_data):
    return anomaly_agent.detect_anomalies(X_data)

def generate_shap_explanation(model, background_df, sample_row_df):
    return xai_agent.explain_shap(model, background_df, sample_row_df)

def generate_counterfactual(model, scaler, sample_row):
    return xai_agent.generate_counterfactual(model, scaler, sample_row)

def generate_fraud_narrative(risk_level, decision, amount, top_feature):
    return xai_agent.create_narrative(risk_level, decision, amount, top_feature)

def record_analyst_action(transaction_id, model_prob, hybrid_risk, analyst_decision, notes=""):
    audit_record = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Transaction_ID": transaction_id,
        "Model_Probability": round(float(model_prob) * 100, 2),
        "Hybrid_Risk_Score": hybrid_risk,
        "Analyst_Decision": analyst_decision,
        "Notes": notes,
        "Orchestrating_Agents": "Agent-Alpha, Agent-Beta, Agent-Gamma"
    }
    return audit_record

def generate_fraud_ring_network(df_subset):
    return syndicate_agent.build_fraud_ring_network(df_subset)