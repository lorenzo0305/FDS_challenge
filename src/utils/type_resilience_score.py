
from src.utils import get_type_chart

def type_resilience_score(types):
        """Score global du type (nombre de forces - faiblesses)."""
        type_chart = get_type_chart()
        score = 0
        for t in types:
            weaknesses = [v for v in type_chart.get(t, {}).values() if v < 1.0]
            strengths = [v for v in type_chart.get(t, {}).values() if v > 1.0]
            score += len(strengths) - len(weaknesses)
        return score / len(types) if types else 0