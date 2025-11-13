from tqdm.notebook import tqdm # type: ignore

def analyze_global_p2_usage(data: list[dict]) -> tuple[set, set, bool]:
    """
    Compares the Pokémon used globally by P1 and P2.
    Indicates whether P2 has used at least one Pokémon that P1 has never used.
    Returns:
        all_p1_pokemons: set of all Pokémon seen for P1
        all_p2_pokemons: set of all Pokémon seen for P2
        has_unique_p2_pokemon: boolean indicating whether P2 has at least one unique Pokémon
    """
    all_p1_pokemons = set()
    all_p2_pokemons = set()

    for battle in tqdm(data, desc="Analyzing global Pokémon usage (P1 vs P2)"):
        # P1 pokemon 
        p1_team = {p['name'].lower() for p in battle.get('p1_team_details', []) if 'name' in p}
        all_p1_pokemons.update(p1_team)

        # P2 pokemon (appearing on the 30 rounds)
        for turn in battle.get('battle_timeline', [])[:30]:
            p2_state = turn.get('p2_pokemon_state', {})
            if isinstance(p2_state, dict):
                name = p2_state.get('name')
                if name:
                    all_p2_pokemons.add(name.lower())

    # check P2 pokemon unique
    p2_unique_pokemons = all_p2_pokemons - all_p1_pokemons
    has_unique_p2_pokemon = len(p2_unique_pokemons) > 0

    print(f"Total Pokémon utilisés par P1 : {len(all_p1_pokemons)}")
    print(f"Total Pokémon utilisés par P2 : {len(all_p2_pokemons)}")
    print(f"P2 a utilisé {len(p2_unique_pokemons)} Pokémon que P1 n’a jamais utilisés.")
    print(f"Présence d’au moins un Pokémon inédit chez P2 : {has_unique_p2_pokemon}")

    return all_p1_pokemons, all_p2_pokemons, has_unique_p2_pokemon