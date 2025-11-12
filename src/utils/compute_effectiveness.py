from src.utils.get_type_chart import get_type_chart


def compute_effectiveness(attack_type, defender_types):
            
            type_chart = get_type_chart()
            eff = 1.0
            for t in defender_types:
                eff *= type_chart.get(attack_type.lower(), {}).get(t.lower(), 1.0)
            return eff