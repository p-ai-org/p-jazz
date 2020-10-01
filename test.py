import music21
from music21 import converter, note, stream, meter
from util import *
from reference import *

''' Convert a single file from midi to numpy '''
# convert_midi_to_numpy_image('alice.mid', verbose=True, piano_part=-1)

''' Convert an entire directory '''
process_directory(INPUT_MIDI_DIR, piano_parts, verbose=True, override=False)

''' Convert a numpy image to a MIDI file '''
# tensor = load_numpy('alice/alice_gray.npy')
# part = numpy_to_part(tensor)
# part_to_midi(part, 'alice_test_gray')

''' Combine all tensors together '''
# long_tensor = combine_into_long_tensor(mode='bin', verbose=True, size=88)

''' Convert all generated images to MIDI and save in a prespecified folder '''
# for i in range(10):
#     save_image88('generated_image_' + str(i) + '.png', suffix=i, verbose=False, subdir='trial_3/', rotate=True)