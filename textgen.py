from util import *
from reference import *

# Note names
CHROMATIC_SCALE_SHARPS = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
# Change depending on your OS
BACKSLASH = '/'
# What goes between notes in a chord
DELIMITER = ', '
# What goes between timesteps
TIMESTEP = '\n'
# Rest
REST = 'wait'
#BEGIN and END tokens
BEGIN = 'BEGIN'
END = 'END'

# Convert an index 0-87 to a note (0 --> A0, 3 --> C1, etc)
def index_to_key(index):
    return CHROMATIC_SCALE_SHARPS[(index + 9) % 12] + str((index + 9) // 12)

# Convert a note name to an index 0-87 (A0 --> 0, C1 --> 3, etc)
def key_to_index(key):
    try:
        return 12 * int(key[-1]) + CHROMATIC_SCALE_SHARPS.index(key[:-1]) - 9
    # Return a special index if something can't be parsed
    except:
        return -1

def numpy_to_text(path):
    ''' Convert a numpy path to text '''
    ret = BEGIN + TIMESTEP
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
            ret += REST
        ret += TIMESTEP
    ret += END + '\n'
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

def text_to_numpy(text):
    tokens = text.split(TIMESTEP)
    arr = np.zeros((1, 88))
    for token in tokens:
        if token in [BEGIN, END]:
            continue
        this_timestep = np.zeros(88)
        if token != REST:
            notes = token.split(DELIMITER)
            for note in notes:
                index = key_to_index(note)
                if index != -1:
                    this_timestep[index] = 1
        arr = np.vstack((arr, [this_timestep]))
    return np.transpose(arr[1:])

def text_to_midi(text, suffix):
    parsed = text_to_numpy(text)
    part = numpy_to_part(parsed)
    part_to_midi(part, 'GPT_model_{}'.format(suffix))

# make_corpus()

# text = ''
# text_to_midi(text, '1')