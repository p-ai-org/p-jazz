import music21
from music21 import converter, note, stream, meter
from util import *

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

''' Convert a single file from midi to numpy '''
# convert_midi_to_numpy_image('alice.mid', verbose=True, piano_part=-1)

''' Convert an entire directory '''
# process_directory(DATA_DIR, piano_parts, verbose=True, override=False)

''' Convert a numpy image to a MIDI file '''
# tensor = load_numpy('alice/alice_gray.npy')
# part = numpy_to_part(tensor)
# part_to_midi(part, 'alice_test_gray')

''' Combine all tensors together '''
# long_tensor = combine_into_long_tensor(mode='bin', verbose=True, size=88)

''' Convert all generated images to MIDI and save in a prespecified folder '''
for i in range(10):
    save_image88('generated_image_' + str(i) + '.png', suffix=i, verbose=False, subdir='trial_3/', rotate=True)