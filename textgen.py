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
REST = 'w'
# BEGIN and END tokens
BEGIN = '<|startoftext|>'
END = '<|endoftext|>'
# Note ON/OFF suffixes
ON = '<'
TAP = '^'
OFF = '>'

def index_to_key(index):
    ''' Convert an index 0-87 to a note (0 --> A0, 3 --> C1, etc) '''
    return CHROMATIC_SCALE_SHARPS[(index + 9) % 12] + str((index + 9) // 12)

def key_to_index(key):
    ''' Convert a note name to an index 0-87 (A0 --> 0, C1 --> 3, etc) '''
    try:
        return 12 * int(key[-1]) + CHROMATIC_SCALE_SHARPS.index(key[:-1]) - 9
    # Return a special index if something can't be parsed
    except:
        return -1

def numpy_to_text(arr, phrases=False):
    '''
    Convert a numpy path to text 

    Details:
        arr: piano roll in (timesteps, keys)
        phrases: a boolean, if True ==> look for end phrase token (89th key)
    
        returns: a string representing the music
    '''
    ret = ''
    for step in arr:
        # Stop early if end of phrase
        if phrases and step[88] == 1:
            break
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
    return ret

def numpy_to_text_onoff(arr, phrases=False):
    '''
    Convert a numpy path to text using on / off notation

    Details:
        arr: piano roll in (timesteps, keys)
        phrases: a boolean, if True ==> look for end phrase token (89th key)
    
        returns: a string representing the music
    '''
    ret = ''
    for i, step in enumerate(arr):
        # Stop early if end of phrase
        if phrases and step[88] == 1:
            break
        # Find where notes are "active"
        notes = np.where(step == 1)[0]
        # String for this timestep
        timestep_string = ''
        # Go through all notes
        for note in notes:
            # Four cases: ON, TAP, OFF, or sustained (nothing)
            # No previous timestep or not active in previous timestep
            if i == 0 or arr[i-1][note] == 0:
                # No next timestep or deactive in next timestep (special case)
                if i == len(arr)-1 or arr[i+1][note] == 0:
                    timestep_string += TAP + index_to_key(note) + DELIMITER
                else:
                    timestep_string += ON + index_to_key(note) + DELIMITER
            # No next timestep or deactive in next timestep
            elif i == len(arr)-1 or arr[i+1][note] == 0:
                timestep_string += OFF + index_to_key(note) + DELIMITER
        if timestep_string == '':
            # Nothing new here
            ret += REST
        else:
            # Cut out the last delimiter
            ret += timestep_string[:-len(DELIMITER)]
        ret += TIMESTEP
    return ret


def make_corpus(verbose=False, input_dir=INPUT_NUMPY_DIR, fname='corpus.txt', onoff=False):
    '''
    Construct corpus from input numpy directory

    Details:
        verbose: boolean, if True ==> print out the files in the input directory
        input_dir: directory to look for npy files
        fname: filename to save text file as
        onoff: boolean, if True ==> use on/off notation
    
        returns: None
    '''
    file = open(fname, "a")
    # Clear file
    file.truncate(0)
    for fname in tqdm(os.listdir(input_dir)):
        if verbose:
            print(fname)
        full_path = "{}{}/{}_bin.npy".format(INPUT_NUMPY_DIR, fname, fname)
        arr = np.transpose(np.load(full_path))
        if onoff:
            file.write(numpy_to_text_onoff(arr))
        else:
            file.write(numpy_to_text(arr))
    file.close()

def make_phrase_corpus(fname='phrases.txt', onoff=False, load_fname='phrases.npy'):
    '''
    Construct corpus from input numpy directory using phrases

    Details:
        fname: filename to save text file as
        onoff: boolean, if True ==> use on/off notation
        load_fname: filename for phrases npy
    
        returns: None
    '''
    file = open(fname, "a")
    file.truncate(0)
    phrases = np.load(load_fname)
    for i, phrase in enumerate(phrases):
        file.write(BEGIN + TIMESTEP)
        if onoff:
            file.write(numpy_to_text_onoff(phrase, phrases=True))
        else:
            file.write(numpy_to_text(phrase, phrases=True))
        file.write(END + TIMESTEP)
    file.close()

def text_to_numpy(text):
    '''
    Converts a string of text into a numpy piano roll

    Details:
        text: a string to be translated

        returns: a numpy array
    '''
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

def text_to_numpy_onoff(text):
    '''
    Converts a string of text into a numpy piano roll using on/off notation

    Details:
        text: a string to be translated

        returns: a numpy array
    '''
    tokens = text.split(TIMESTEP)
    arr = np.zeros((1, 88))
    # Keep track of which notes we're supposed to keep filling in
    obligations = np.zeros(88)
    for token in tokens:
        if token in [BEGIN, END]:
            continue
        this_timestep = np.zeros(88)
        # Set all obligations
        for i in range(88):
            this_timestep[i] = obligations[i]
        # Deal with tokens
        if token != REST:
            notes = token.split(DELIMITER)
            for note in notes:
                # Get prefix (on, off, or tap)
                prefix = note[0]
                index = key_to_index(note[1:])
                # Quick validity check, otherwise ignore
                if index != -1 and prefix in [ON, OFF, TAP]:
                    # If on (and valid; obligation should be off) activate and turn on obligation
                    if prefix == ON and obligations[index] == 0:
                        obligations[index] = 1
                        this_timestep[index] = 1
                    # If off (and valid; obligation should be on) activate and turn off obligation
                    elif prefix == OFF and obligations[index] == 1:
                        obligations[index] = 0
                        this_timestep[index] = 1
                    # If tap (and valid; obligation should be off) activate and turn off
                    elif prefix == TAP and obligations[index] == 0:
                        obligations[index] = 0
                        this_timestep[index] = 1
        arr = np.vstack((arr, [this_timestep]))
    return np.transpose(arr[1:])

def text_to_midi(text, suffix, onoff=False):
    '''
    Convert output text back into a MIDI file

    Details:
        text: a string to be translated
        suffix: a string to be appended to the filename (before the extension)
        onoff: boolean, if True ==> use on/off notation
    '''
    if onoff:
        parsed = text_to_numpy_onoff(text)
    else:
        parsed = text_to_numpy(text)
    part = numpy_to_part(parsed)
    part_to_midi(part, 'GPT_model_{}'.format(suffix))

''' MAKE CORPUS '''
# make_corpus(fname='corpus_onoff.txt', onoff=True)
# make_phrase_corpus(fname='phrases.txt', onoff=True)

''' READ OUTPUT '''
# text = open('output.txt', 'r').read()
# text_to_midi(text, '1', onoff=True)