from src.utils.get_type_chart import get_type_chart


def get_effectiveness(attacker_types, defender_types):
        """Score moyen d'efficacité entre deux sets de types."""
        type_chart = get_type_chart()
        total, count = 0, 0
        for atk in attacker_types:
            for d in defender_types:
                total += type_chart.get(atk, {}).get(d, 1.0)
                count += 1
        return total / count if count else 1.0

