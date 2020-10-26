from util import *
from reference import *

CHROMATIC_SCALE_SHARPS = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
BACKSLASH = '/'
DELIMITER = ','

def index_to_key(i):
    return CHROMATIC_SCALE_SHARPS[(i + 9) % 12] + str((i + 9) // 12)

def key_to_index(i):
    return 12 * int(i[-1]) + indexOf(CHROMATIC_SCALE_SHARPS(i[:-1]))

def midi_to_text(path):
    ret = 'BEGIN\n'
    arr = np.transpose(np.load(path))
    for step in arr:
        notes = np.where(step == 1)[0]
        if np.size(notes) != 0:
            ret += index_to_key(notes[0])
            for note in notes[1:]:
                ret += DELIMITER + index_to_key(note)
        else:
            ret += 'wait'
        ret += '\n'
    ret += 'END\n'
    return ret

def make_corpus():
    directory = 'input_midi'
    file = open('corpus.txt',"a")
    file.truncate(0)
    for fname in os.listdir(INPUT_NUMPY_DIR):
        print(fname)
        full_path = "{}{}/{}_bin.npy".format(INPUT_NUMPY_DIR, fname, fname)
        file.write(midi_to_text(path=full_path))