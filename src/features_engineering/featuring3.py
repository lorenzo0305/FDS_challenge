from src.utils.get_type_chart import get_type_chart
from src.utils.get_effectiveness import get_effectiveness
from src.utils.type_resilience_score import type_resilience_score
from src.utils.analyze_global_p2_usage import analyze_global_p2_usage
import pandas as pd
import numpy as np
from IPython.display import display
from tqdm.notebook import tqdm # type: ignore


def create_simple_features(data: list[dict], type_lookup: dict, all_p2_pokemons: set = None) -> pd.DataFrame:
    """
    Extract features from Pokémon battle data.
    
    Args:
        data: list of battle dictionaries
        type_lookup: dict mapping Pokémon name -> stats dict with keys 'base_hp', 'base_atk', etc.
        all_p2_pokemons: optional set of globally seen P2 Pokémon for fallback

    Returns:
        DataFrame with features for each battle
    """
    feature_list = []
    type_chart = get_type_chart()
    print("Building Pokémon type lookup table...")

    for battle in tqdm(data, desc="Extracting features"):
        features = {}
        battle_timeline = battle.get('battle_timeline', [])
        if not battle_timeline:
            continue

        p1_team = battle.get('p1_team_details', [])
        p2_lead = battle.get('p2_lead_details', {})

        # --- P1 Team stats ---
        if p1_team:
            features['p1_mean_hp']  = np.mean([p.get('base_hp', 0) for p in p1_team])
            features['p1_mean_atk'] = np.mean([p.get('base_atk', 0) for p in p1_team])
            features['p1_mean_def'] = np.mean([p.get('base_def', 0) for p in p1_team])
            features['p1_mean_spe'] = np.mean([p.get('base_spe', 0) for p in p1_team])
        else:
            features['p1_mean_hp'] = features['p1_mean_atk'] = features['p1_mean_def'] = features['p1_mean_spe'] = 0

        # --- Approximation P2 Team ---
        p2_seen = set()
        for turn in battle_timeline[:30]:
            state = turn.get('p2_pokemon_state', {})
            if isinstance(state, dict):
                name = state.get('name')
                if name:
                    p2_seen.add(name.lower())

        # Fallback sur la liste globale
        if not p2_seen and all_p2_pokemons:
            p2_seen = set(list(all_p2_pokemons)[:6])

        # --- P2 stats ---
        p2_stats = [type_lookup[name] for name in p2_seen if isinstance(type_lookup.get(name), dict)]
        if p2_stats:
            features['p2_mean_hp']  = np.mean([s.get('base_hp', 0) for s in p2_stats])
            features['p2_mean_atk'] = np.mean([s.get('base_atk', 0) for s in p2_stats])
            features['p2_mean_def'] = np.mean([s.get('base_def', 0) for s in p2_stats])
            features['p2_mean_spe'] = np.mean([s.get('base_spe', 0) for s in p2_stats])
        else:
            features['p2_mean_hp'] = features['p2_mean_atk'] = features['p2_mean_def'] = features['p2_mean_spe'] = 0

        # --- Différences P1 vs P2 ---
        features['hp_team_diff']  = features['p1_mean_hp']  - features['p2_mean_hp']
        features['atk_team_diff'] = features['p1_mean_atk'] - features['p2_mean_atk']
        features['def_team_diff'] = features['p1_mean_def'] - features['p2_mean_def']
        features['spe_team_diff'] = features['p1_mean_spe'] - features['p2_mean_spe']

        # --- Status ---
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

        # --- TEMPO ---
        p1_adv_turns = p2_adv_turns = 0
        for turn in battle_timeline[:30]:
            p1_hp = turn.get('p1_pokemon_state', {}).get('hp_pct', 1.0)
            p2_hp = turn.get('p2_pokemon_state', {}).get('hp_pct', 1.0)
            if p1_hp > p2_hp: p1_adv_turns += 1
            elif p2_hp > p1_hp: p2_adv_turns += 1
        features['p1_advantage_ratio'] = p1_adv_turns / 30
        features['p2_advantage_ratio'] = p2_adv_turns / 30
        features['tempo_balance'] = features['p1_advantage_ratio'] - features['p2_advantage_ratio']

        # --- Survivants ---
        p1_alive, p2_alive = [], []
        p1_hp_alive, p2_hp_alive = {}, {}
        p1_types, p2_types = [], []

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

        features['p1_alive_count'] = len(set(p1_alive))
        features['p2_alive_count'] = len(set(p2_alive))
        features['alive_diff'] = features['p1_alive_count'] - features['p2_alive_count']
        features['p1_alive_type_score'] = type_resilience_score(p1_types)
        features['p2_alive_type_score'] = type_resilience_score(p2_types)
        features['type_alive_diff'] = features['p1_alive_type_score'] - features['p2_alive_type_score']

        # --- type_hp_match_score ---
        matchup_sum = matchup_count = 0
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

