# Build lookup table of Pokémon types from all known teams
def build_type_lookup(data):
    lookup = {}
    for battle in data:
        for poke in battle.get('p1_team_details', []):
            name = poke.get('name')
            types = [t.lower() for t in poke.get('types', []) if t and t.lower() != 'notype']
            if name and types:
                lookup[name.lower()] = types
    return lookup


