from src.utils.get_type_chart import get_type_chart
from src.utils.compute_effectiveness import compute_effectiveness
import pandas as pd
import numpy as np

from tqdm.notebook import tqdm



def create_simple_features(data: list[dict]) -> pd.DataFrame:
    """
    Extract battle-level features from Pokémon battle data.
    """

    type_chart = get_type_chart()

    feature_list = []

    for battle in tqdm(data, desc="Extracting features"):
        features = {}
        battle_timeline = battle.get('battle_timeline', [])
        p1_team = battle.get('p1_team_details', [])
        p2_lead = battle.get('p2_lead_details', {})

        # --- Player 1 Team Features ---
        if p1_team:
            features['p1_mean_hp'] = np.mean([p.get('base_hp', 0) for p in p1_team])
            features['p1_mean_spe'] = np.mean([p.get('base_spe', 0) for p in p1_team])
            features['p1_mean_atk'] = np.mean([p.get('base_atk', 0) for p in p1_team])
            features['p1_mean_def'] = np.mean([p.get('base_def', 0) for p in p1_team])

        # --- Player 2 Lead Features ---
        if p2_lead:
            features['p2_lead_hp'] = p2_lead.get('base_hp', 0)
            features['p2_lead_spe'] = p2_lead.get('base_spe', 0)
            features['p2_lead_atk'] = p2_lead.get('base_atk', 0)
            features['p2_lead_def'] = p2_lead.get('base_def', 0)

        # --- Nombre de K.O. pour chaque joueur ---
        p1_KO_set, p2_KO_set = set(), set()
        if battle_timeline:
            for turn in battle_timeline[:30]:
                # Joueur 1
                p1_state = turn.get('p1_pokemon_state', {})
                if isinstance(p1_state, dict):
                    name = p1_state.get('name')
                    status = str(p1_state.get('status', '')).lower()
                    hp = p1_state.get('hp_pct', 1.0)
                    if ('fnt' in status) or (float(hp) == 0.0):
                        if name:
                            p1_KO_set.add(name)
                # Joueur 2
                p2_state = turn.get('p2_pokemon_state', {})
                if isinstance(p2_state, dict):
                    name = p2_state.get('name')
                    status = str(p2_state.get('status', '')).lower()
                    hp = p2_state.get('hp_pct', 1.0)
                    if ('fnt' in status) or (float(hp) == 0.0):
                        if name:
                            p2_KO_set.add(name)

        features['p1_num_KO'] = len(p1_KO_set)
        features['p2_num_KO'] = len(p2_KO_set)
        features['ko_diff'] = len(p2_KO_set) - len(p1_KO_set)

        # --- Comptage des statuts (hors 'fnt') sur les 30 premiers tours ---
        p1_status_set, p2_status_set = [], []
        if battle_timeline:
            for turn in battle_timeline[:30]:
                # Joueur 1
                p1_state = turn.get('p1_pokemon_state', {})
                if isinstance(p1_state, dict):
                    status = str(p1_state.get('status', '')).lower()
                    if status not in ['nostatus', 'noeffect', '', 'fnt', 'none']:
                        p1_status_set.append(status)
                # Joueur 2
                p2_state = turn.get('p2_pokemon_state', {})
                if isinstance(p2_state, dict):
                    status = str(p2_state.get('status', '')).lower()
                    if status not in ['nostatus', 'noeffect', '', 'fnt', 'none']:
                        p2_status_set.append(status)

        features['p1_num_status'] = len(p1_status_set)
        features['p2_num_status'] = len(p2_status_set)
        features['status_diff'] = len(p2_status_set) - len(p1_status_set)

        # --- Pourcentage de HP restants à la fin des 30 premiers tours ---
        p1_hp_last, p2_hp_last = {}, {}
        if battle_timeline:
            for turn in battle_timeline[:30]:
                # Joueur 1
                p1_state = turn.get('p1_pokemon_state', {})
                if isinstance(p1_state, dict):
                    name = p1_state.get('name')
                    hp = p1_state.get('hp_pct', None)
                    if name and isinstance(hp, (int, float)):
                        p1_hp_last[name] = hp
                # Joueur 2
                p2_state = turn.get('p2_pokemon_state', {})
                if isinstance(p2_state, dict):
                    name = p2_state.get('name')
                    hp = p2_state.get('hp_pct', None)
                    if name and isinstance(hp, (int, float)):
                        p2_hp_last[name] = hp

        for p in battle.get('p1_team_details', []):
            name = p.get('name')
            if name not in p1_hp_last:
                p1_hp_last[name] = 1.0
        if 'p2_team_details' in battle:
            for p in battle.get('p2_team_details', []):
                name = p.get('name')
                if name not in p2_hp_last:
                    p2_hp_last[name] = 1.0
        else:
            p2_lead = battle.get('p2_lead_details', {})
            if p2_lead:
                name = p2_lead.get('name')
                if name and name not in p2_hp_last:
                    p2_hp_last[name] = 1.0

        features['p1_mean_hp_remaining'] = np.mean(list(p1_hp_last.values())) if p1_hp_last else 1.0
        features['p2_mean_hp_remaining'] = np.mean(list(p2_hp_last.values())) if p2_hp_last else 1.0
        features['hp_remaining_diff'] = features['p2_mean_hp_remaining'] - features['p1_mean_hp_remaining']


        # --- Attaques subies (vulnérabilité) ---
        attack_types_received = []
        for turn in battle_timeline[:30]:
            move_details = turn.get('p2_move_details')
            if isinstance(move_details, dict):  # ✅ Vérifie que c’est bien un dict
                move_type = str(move_details.get('type', '')).lower()
                if move_type not in ['', 'notype', 'none']:
                    attack_types_received.append(move_type)

        if attack_types_received and p1_team:
            total_eff, count = 0, 0
            for atk_type in attack_types_received:
                for poke in p1_team:
                    def_types = [t for t in poke.get('types', []) if t != 'notype']
                    if def_types:
                        total_eff += compute_effectiveness(atk_type, def_types)
                        count += 1
            features['p1_type_vulnerability'] = total_eff / count if count > 0 else 1.0
        else:
            features['p1_type_vulnerability'] = 1.0


        p1_advantage_turns = 0
        p2_advantage_turns = 0

        for turn in battle_timeline[:30]:
            p1_hp = turn.get('p1_pokemon_state', {}).get('hp_pct', 1.0)
            p2_hp = turn.get('p2_pokemon_state', {}).get('hp_pct', 1.0)
            if p1_hp > p2_hp:
                p1_advantage_turns += 1
            elif p2_hp > p1_hp:
                p2_advantage_turns += 1

        features['p1_advantage_ratio'] = p1_advantage_turns / 30
        features['p2_advantage_ratio'] = p2_advantage_turns / 30
        features['tempo_balance'] = features['p1_advantage_ratio'] - features['p2_advantage_ratio']

        
        # --- BOOSTS MOYENS SUR LES 30 PREMIERS TOURS ---
        p1_boost_sum = 0
        p2_boost_sum = 0
        p1_boost_count = 0
        p2_boost_count = 0

        if battle_timeline:
            for turn in battle_timeline[:30]:
                # Joueur 1
                p1_state = turn.get('p1_pokemon_state', {})
                if isinstance(p1_state, dict):
                    boosts = p1_state.get('boosts', {})
                    if isinstance(boosts, dict) and boosts:
                        boost_total = sum(boosts.get(stat, 0) for stat in ['atk', 'def', 'spa', 'spd', 'spe'])
                        p1_boost_sum += boost_total
                        p1_boost_count += 1

                # Joueur 2
                p2_state = turn.get('p2_pokemon_state', {})
                if isinstance(p2_state, dict):
                    boosts = p2_state.get('boosts', {})
                    if isinstance(boosts, dict) and boosts:
                        boost_total = sum(boosts.get(stat, 0) for stat in ['atk', 'def', 'spa', 'spd', 'spe'])
                        p2_boost_sum += boost_total
                        p2_boost_count += 1

        # Moyenne des boosts (ou 0 si aucune donnée)
        features['p1_mean_boosts'] = p1_boost_sum / p1_boost_count if p1_boost_count > 0 else 0
        features['p2_mean_boosts'] = p2_boost_sum / p2_boost_count if p2_boost_count > 0 else 0
        features['boost_diff'] = features['p2_mean_boosts'] - features['p1_mean_boosts']

        
        # --- Battle ID & target ---
        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])

        feature_list.append(features)

    return pd.DataFrame(feature_list).fillna(0)


