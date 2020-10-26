# Dictionary of piano parts (for MIDI with multiple parts)
piano_parts = {
    'blackbird': 1, 
    'beautiful': 0, 
    'pathetique': 0, 
    'bluebossa1': 0, 
    'bluebossa2': 0, 
    'blameiton': 0, 
    'brazilian': 0, 
    'brazilsuite': 0, 
    'broadway': 0, 
    'cantible': 0, 
    'bymyself': 2, 
    'closeyoureyes': 0, 
    'cubanochant': 0,
    'dearlybeloved': 0,
    'daysofwine': 0,
    'desafinado': 0,
    'desire': 0,
    'effendi': 1,
    'exactlylikeyou': 1,
    'howcomeyoulikeme': 0,
    'girltalk': 0,
    'hymntofreedom': 0,
    'goodbait': 1,
    'ifiwere': 0,
    'inasentimental1': 1,
    'itcouldhappen': 0,
    'ivegrownacc': 0,
    'ineverknew': 0,
    'ishouldcare2': 0,
    'iconcentrate': 0
}

def get_piano_part(name):
    if name not in piano_parts:
        return 0
    else:
        return piano_parts[name]