from src.utils.get_type_chart import get_type_chart
from src.utils.get_effectiveness import get_effectiveness
from src.utils.type_resilience_score import type_resilience_score
from src.utils.build_type_lookup import build_type_lookup
import pandas as pd
import numpy as np


def create_simple_features(data: list[dict], type_lookup: dict) -> pd.DataFrame:
    """
    Extracts features from Pokémon battle data.
    - Team stats
    - Status and boosts
    - Tempo advantage
    - Type vulnerability
    - Survivors (alive count + types + HP)
    - NEW: type_hp_match_score -> comparative advantage between P1 and P2 survivors
    """
    feature_list = []
    type_chart = get_type_chart()
    print("Building Pokémon type lookup table...")
    type_lookup = build_type_lookup(train_data)
    print(f"Type lookup built with {len(type_lookup)} unique Pokémon.")


    for battle in tqdm(data, desc="Extracting features"):
        features = {}
        battle_timeline = battle.get('battle_timeline', [])
        if not battle_timeline:
            continue

        p1_team = battle.get('p1_team_details', [])
        p2_lead = battle.get('p2_lead_details', {})

        # --- Moyennes d’équipe P1 ---
        if p1_team:
            features['p1_mean_hp'] = np.mean([p.get('base_hp', 0) for p in p1_team])
            features['p1_mean_atk'] = np.mean([p.get('base_atk', 0) for p in p1_team])
            features['p1_mean_def'] = np.mean([p.get('base_def', 0) for p in p1_team])
            features['p1_mean_spe'] = np.mean([p.get('base_spe', 0) for p in p1_team])

        # --- Statistiques du lead adverse ---
        if p2_lead:
            features['p2_lead_hp'] = p2_lead.get('base_hp', 0)
            features['p2_lead_atk'] = p2_lead.get('base_atk', 0)
            features['p2_lead_def'] = p2_lead.get('base_def', 0)
            features['p2_lead_spe'] = p2_lead.get('base_spe', 0)

        # --- Statuts (hors fainted) ---
        p1_status, p2_status = [], []
        for turn in battle_timeline[:30]:
            for player, store in [('p1_pokemon_state', p1_status), ('p2_pokemon_state', p2_status)]:
                state = turn.get(player, {})
                if isinstance(state, dict):
                    status = str(state.get('status', '')).lower()
                    if status not in ['nostatus', 'noeffect', '', 'fnt', 'none']:
                        store.append(status)
        features['p1_num_status'] = len(p1_status)
        features['p2_num_status'] = len(p2_status)
        features['status_diff'] = len(p2_status) - len(p1_status)

        # --- TEMPO / ADVANTAGE ---
        p1_adv_turns = p2_adv_turns = 0
        for turn in battle_timeline[:30]:
            p1_hp = turn.get('p1_pokemon_state', {}).get('hp_pct', 1.0)
            p2_hp = turn.get('p2_pokemon_state', {}).get('hp_pct', 1.0)
            if p1_hp > p2_hp:
                p1_adv_turns += 1
            elif p2_hp > p1_hp:
                p2_adv_turns += 1
        features['p1_advantage_ratio'] = p1_adv_turns / 30
        features['p2_advantage_ratio'] = p2_adv_turns / 30
        features['tempo_balance'] = features['p1_advantage_ratio'] - features['p2_advantage_ratio']

        # --- SURVIVANTS À LA FIN DES 30 TOURS ---
        p1_alive, p2_alive = [], []
        p1_hp_alive, p2_hp_alive = {}, {}
        p1_types, p2_types = [], []

        # Joueur 1
        for poke in p1_team:
            name = poke.get('name')
            last_state = None
            for turn in reversed(battle_timeline[:30]):
                state = turn.get('p1_pokemon_state', {})
                if isinstance(state, dict) and state.get('name') == name:
                    last_state = state
                    break
            hp = last_state.get('hp_pct', 1.0) if last_state else 1.0
            status = str(last_state.get('status', 'nostatus')) if last_state else 'nostatus'
            if hp > 0 and 'fnt' not in status:
                p1_alive.append(name.lower())
                p1_hp_alive[name.lower()] = hp
                p1_types.extend([t for t in poke.get('types', []) if t != 'notype'])

        # Joueur 2
        for turn in battle_timeline[:30]:
            state = turn.get('p2_pokemon_state', {})
            if isinstance(state, dict):
                name = state.get('name')
                hp = state.get('hp_pct', 1.0)
                status = str(state.get('status', 'nostatus'))
                if name and hp > 0 and 'fnt' not in status:
                    p2_hp_alive[name.lower()] = hp
                    if 'p2_lead_details' in battle and battle['p2_lead_details'].get('name') == name:
                        p2_types.extend([t for t in battle['p2_lead_details'].get('types', []) if t != 'notype'])
                    p2_alive.append(name.lower())

        # --- Compte et scores simples ---
        features['p1_alive_count'] = len(set(p1_alive))
        features['p2_alive_count'] = len(set(p2_alive))
        features['alive_diff'] = features['p2_alive_count'] - features['p1_alive_count']
        features['p1_alive_type_score'] = type_resilience_score(p1_types)
        features['p2_alive_type_score'] = type_resilience_score(p2_types)
        features['type_alive_diff'] = features['p2_alive_type_score'] - features['p1_alive_type_score']

        # --- 🔹 NOUVEL INDICATEUR : type_hp_match_score ---
        matchup_sum, matchup_count = 0, 0
        for p1_name in p1_alive:
            p1_types_local = type_lookup.get(p1_name, [])
            for p2_name in p2_alive:
                p2_types_local = type_lookup.get(p2_name, [])
                eff = get_effectiveness(p1_types_local, p2_types_local)
                hp_weight = p1_hp_alive.get(p1_name, 1.0) - p2_hp_alive.get(p2_name, 1.0)
                matchup_sum += eff * hp_weight
                matchup_count += 1
        features['type_hp_match_score'] = matchup_sum / matchup_count if matchup_count else 0

        # --- ID et cible ---
        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])

        feature_list.append(features)

    df = pd.DataFrame(feature_list).fillna(0)
    print(f"\n✅ Feature extraction done for {len(df)} battles.")
    display(df.head())
    return df




# Create feature DataFrames for both training and test sets
print("Processing training data...")
train_df = create_simple_features(train_data, type_lookup)


print("\nProcessing test data...")
test_data = []
with open(test_file_path, 'r') as f:
    for line in f:
        test_data.append(json.loads(line))
test_df = create_simple_features(test_data, type_lookup)

print("\nTraining features preview:")
display(train_df.head())