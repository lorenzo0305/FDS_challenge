from tqdm.notebook import tqdm # type: ignore

def analyze_global_p2_usage(data: list[dict]) -> tuple[set, set, bool]:
    """
    Compare les Pokémon utilisés globalement par P1 et P2.
    Indique si P2 a utilisé au moins un Pokémon que P1 n'a jamais utilisé.
    
    Retourne :
    - all_p1_pokemons : ensemble de tous les Pokémon vus chez P1
    - all_p2_pokemons : ensemble de tous les Pokémon vus chez P2
    - has_unique_p2_pokemon : booléen indiquant si P2 a au moins un Pokémon inédit
    """
    all_p1_pokemons = set()
    all_p2_pokemons = set()

    for battle in tqdm(data, desc="Analyzing global Pokémon usage (P1 vs P2)"):
        # Pokémon de P1 (équipe initiale)
        p1_team = {p['name'].lower() for p in battle.get('p1_team_details', []) if 'name' in p}
        all_p1_pokemons.update(p1_team)

        # Pokémon de P2 (tous ceux vus dans les 30 premiers tours)
        for turn in battle.get('battle_timeline', [])[:30]:
            p2_state = turn.get('p2_pokemon_state', {})
            if isinstance(p2_state, dict):
                name = p2_state.get('name')
                if name:
                    all_p2_pokemons.add(name.lower())

    # Pokémon uniques à P2
    p2_unique_pokemons = all_p2_pokemons - all_p1_pokemons
    has_unique_p2_pokemon = len(p2_unique_pokemons) > 0

    print(f"Total Pokémon utilisés par P1 : {len(all_p1_pokemons)}")
    print(f"Total Pokémon utilisés par P2 : {len(all_p2_pokemons)}")
    print(f"P2 a utilisé {len(p2_unique_pokemons)} Pokémon que P1 n’a jamais utilisés.")
    print(f"Présence d’au moins un Pokémon inédit chez P2 : {has_unique_p2_pokemon}")

    return all_p1_pokemons, all_p2_pokemons, has_unique_p2_pokemon