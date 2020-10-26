from util import *
from reference import *

# Note names
CHROMATIC_SCALE_SHARPS = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
# Change depending on your OS
BACKSLASH = '/'
# What goes between notes in a chord
DELIMITER = ','

# Convert an index 0-87 to a note (0 --> A0, 3 --> C1, etc)
def index_to_key(index):
    return CHROMATIC_SCALE_SHARPS[(index + 9) % 12] + str((index + 9) // 12)

# Convert a note name to an index 0-87 (A0 --> 0, C1 --> 3, etc)
def key_to_index(key):
    return 12 * int(key[-1]) + CHROMATIC_SCALE_SHARPS.index(key[:-1]) - 9

def numpy_to_text(path):
    ''' Convert a numpy path to text '''
    ret = 'BEGIN\n'
    arr = np.transpose(np.load(path))
    for step in arr:
        # Find where notes are "active"
        notes = np.where(step == 1)[0]
        if np.size(notes) != 0:
            # Add first note
            ret += index_to_key(notes[0])
            # Delimit additional notes
            for note in notes[1:]:
                ret += DELIMITER + index_to_key(note)
        # Rest
        else:
            ret += 'wait'
        ret += '\n'
    ret += 'END\n'
    return ret

def make_corpus(verbose=False):
    ''' Construct corpus from input numpy directory '''
    file = open('corpus.txt',"a")
    # Clear file
    file.truncate(0)
    for fname in tqdm(os.listdir(INPUT_NUMPY_DIR)):
        if verbose:
            print(fname)
        full_path = "{}{}/{}_bin.npy".format(INPUT_NUMPY_DIR, fname, fname)
        file.write(numpy_to_text(path=full_path))