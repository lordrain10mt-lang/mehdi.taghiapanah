import numpy as np
from typing import List, Dict, Tuple

class RegulatoryInput:
    def __init__(self, controls: List[Tuple[float, float, float]]):
        self.controls = controls

class RiskInput:
    def __init__(self, 
                 incident_counts: List[int],
                 severity_scores: List[float],
                 operational_deviations: List[float]):
        self.incident_counts = incident_counts
        self.severity_scores = severity_scores
        self.operational_deviations = operational_deviations

class MultiCloudInput:
    def __init__(self,
                 latency_ms: List[float],
                 bandwidth_mbps: List[float],
                 availability: List[float],
                 provider_count: int):
        self.latency_ms = latency_ms
        self.bandwidth_mbps = bandwidth_mbps
        self.availability = availability
        self.provider_count = provider_count

class FeedbackInput:
    def __init__(self,
                 raw_feedbacks: List[float],
                 auditor_rating: float,
                 trust_score_history: List[float]):
        self.raw_feedbacks = raw_feedbacks
        self.auditor_rating = auditor_rating
        self.trust_score_history = trust_score_history

class RegulatoryLayer:
    def __init__(self, weight: float = 0.35):
        self.weight = weight

    def normalize_control(self, ccv: float, acm: float, csf: float) -> float:
        ccv_n = np.clip(ccv / 100.0, 0.0, 1.0)
        acm_n = np.clip(acm / 100.0, 0.0, 1.0)
        csf_n = np.clip(csf, 0.0, 1.0)
        return (ccv_n * 0.4 + acm_n * 0.3 + csf_n * 0.3)

    def calculate_score(self, regulatory_input: RegulatoryInput) -> float:
        if not regulatory_input.controls:
            return 0.0
        normalized_scores = [self.normalize_control(ccv, acm, csf) 
                             for ccv, acm, csf in regulatory_input.controls]
        return np.mean(normalized_scores)

class RiskLayer:
    def __init__(self, weight: float = 0.30):
        self.weight = weight

    def calculate_score(self, risk_input: RiskInput) -> float:
        incident_norm = np.mean([min(c / 10.0, 1.0) for c in risk_input.incident_counts])
        severity_norm = np.mean([s / 10.0 for s in risk_input.severity_scores])
        deviation_norm = np.mean(risk_input.operational_deviations)
        risk_score = (incident_norm * 0.4 + severity_norm * 0.4 + deviation_norm * 0.2)
        return 1.0 - risk_score

class MultiCloudLayer:
    def __init__(self, weight: float = 0.20):
        self.weight = weight

    def calculate_score(self, mc_input: MultiCloudInput) -> float:
        latency_scores = [max(0, 1.0 - lat/100.0) for lat in mc_input.latency_ms]
        latency_norm = np.mean(latency_scores)
        bandwidth_scores = [min(bw / 10.0, 1.0) for bw in mc_input.bandwidth_mbps]
        bandwidth_norm = np.mean(bandwidth_scores)
        availability_norm = np.mean(mc_input.availability)
        provider_penalty = min(1.0, mc_input.provider_count / 5.0)
        provider_factor = 1.0 - (provider_penalty * 0.2)
        context_score = (latency_norm * 0.3 + bandwidth_norm * 0.3 + 
                         availability_norm * 0.4) * provider_factor
        return context_score

class FeedbackLayer:
    def __init__(self, weight: float = 0.15):
        self.weight = weight

    def remove_outliers(self, values: List[float]) -> List[float]:
        if len(values) < 3:
            return values
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return [v for v in values if lower <= v <= upper]

    def calculate_score(self, feedback_input: FeedbackInput) -> float:
        raw_norm = [fb / 5.0 for fb in feedback_input.raw_feedbacks]
        cleaned = self.remove_outliers(raw_norm)
        if not cleaned:
            feedback_avg = 0.0
        else:
            feedback_avg = np.mean(cleaned)
        auditor_score = feedback_input.auditor_rating
        if len(feedback_input.trust_score_history) >= 2:
            history_std = np.std(feedback_input.trust_score_history)
            stability = 1.0 - min(history_std, 0.5)
        else:
            stability = 1.0
        validated_score = (feedback_avg * 0.5 + auditor_score * 0.3 + stability * 0.2)
        return validated_score

class TrustworthinessModel:
    def __init__(self,
                 reg_weight=0.35,
                 risk_weight=0.30,
                 context_weight=0.20,
                 feedback_weight=0.15):
        total = reg_weight + risk_weight + context_weight + feedback_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1 (current: {total})")
        self.reg_layer = RegulatoryLayer(weight=reg_weight)
        self.risk_layer = RiskLayer(weight=risk_weight)
        self.mc_layer = MultiCloudLayer(weight=context_weight)
        self.fb_layer = FeedbackLayer(weight=feedback_weight)
        self.weights = {
            'regulatory': reg_weight,
            'risk': risk_weight,
            'context': context_weight,
            'feedback': feedback_weight
        }

    def compute_layer_scores(self,
                             reg_input: RegulatoryInput,
                             risk_input: RiskInput,
                             mc_input: MultiCloudInput,
                             fb_input: FeedbackInput) -> Dict[str, float]:
        scores = {}
        scores['regulatory'] = self.reg_layer.calculate_score(reg_input)
        scores['risk'] = self.risk_layer.calculate_score(risk_input)
        scores['context'] = self.mc_layer.calculate_score(mc_input)
        scores['feedback'] = self.fb_layer.calculate_score(fb_input)
        return scores

    def compute_trustworthiness(self,
                                reg_input: RegulatoryInput,
                                risk_input: RiskInput,
                                mc_input: MultiCloudInput,
                                fb_input: FeedbackInput) -> Dict[str, float]:
        layer_scores = self.compute_layer_scores(reg_input, risk_input, mc_input, fb_input)
        tw_score = (layer_scores['regulatory'] * self.weights['regulatory'] +
                    layer_scores['risk'] * self.weights['risk'] +
                    layer_scores['context'] * self.weights['context'] +
                    layer_scores['feedback'] * self.weights['feedback'])
        tw_score = round(tw_score, 4)
        for key in layer_scores:
            layer_scores[key] = round(layer_scores[key], 4)
        return {
            'layer_scores': layer_scores,
            'trustworthiness_score': tw_score,
            'weights': self.weights
        }

def run_example():
    print("=" * 60)
    print("Trustworthiness Model - Complete Example")
    print("=" * 60)

    reg_input = RegulatoryInput(controls=[
        (85, 90, 0.8),
        (70, 60, 0.6),
        (95, 100, 1.0),
        (50, 80, 0.4)
    ])
    
    risk_input = RiskInput(
        incident_counts=[2, 1, 0, 3],
        severity_scores=[3.5, 7.0, 0.0, 5.0],
        operational_deviations=[0.1, 0.3, 0.05, 0.2]
    )
    
    mc_input = MultiCloudInput(
        latency_ms=[45, 120, 80, 60],
        bandwidth_mbps=[15, 8, 12, 20],
        availability=[0.99, 0.95, 0.98, 0.99],
        provider_count=3
    )
    
    fb_input = FeedbackInput(
        raw_feedbacks=[4.2, 3.8, 4.5, 2.1, 4.9, 3.5],
        auditor_rating=0.75,
        trust_score_history=[0.72, 0.68, 0.75, 0.70]
    )

    model = TrustworthinessModel()
    results = model.compute_trustworthiness(reg_input, risk_input, mc_input, fb_input)

    print("\n--- Layer Scores ---")
    for layer, score in results['layer_scores'].items():
        print(f"  {layer:12s}: {score:.4f}")

    print("\n--- Layer Weights ---")
    for layer, weight in results['weights'].items():
        print(f"  {layer:12s}: {weight:.2f}")

    print("\n--- Final Trustworthiness Score (TW) ---")
    print(f"  TW = {results['trustworthiness_score']:.4f}")

    print("\n--- Interpretation ---")
    tw = results['trustworthiness_score']
    if tw >= 0.8:
        interpretation = "Excellent (Highly Trustworthy System)"
    elif tw >= 0.6:
        interpretation = "Good (Trustworthy System)"
    elif tw >= 0.4:
        interpretation = "Moderate (Needs Improvement)"
    else:
        interpretation = "Poor (Urgent Action Required)"
    print(f"  {interpretation}")

    print("\n" + "=" * 60)
    print("Example Completed")
    print("=" * 60)

    return results

if __name__ == "__main__":
    run_example()
