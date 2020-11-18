import music21
from music21 import converter, corpus, instrument, midi, note, chord, pitch, stream
from PIL import Image
from tqdm import tqdm
from reference import *
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import sys
import os
import copy
from sklearn.metrics.pairwise import cosine_similarity

''' SAVE DIRECTORIES '''
# Where input MIDI files are stored
INPUT_MIDI_DIR = 'input_midi/'
# Where input MIDI images are stored
INPUT_IMAGE_DIR = 'input_images/'
# Where input numpy arrays are stored
INPUT_NUMPY_DIR = 'input_numpy/'
# Where output MIDI files are stored
OUTPUT_MIDI_DIR = 'output_midi/'
# Where keras models are stored
MODEL_SAVE_DIR = 'keras_models/'
# Where batched arrays are stored
BATCH_SAVE_DIR = 'batched_data/'
# Where generated images are stored
OUTPUT_IMAGE_DIR = 'output_images/'

''' OTHER CONSTANTS '''
# Quarter note scale (1 quarter note is split up into 12 pixels for accuracy)
SCALE = 12
# Maximum velocity
MAX_VEL = 127
# Highest key
MAX_KEY = 127
# Highest pixel value (greyscale and RGB)
MAX_PIXEL_STRENGTH = 255
# Number of piano keys
NUM_KEYS = 88
# np.set_printoptions(threshold=sys.maxsize)

def load_numpy(fname):
    '''Load a numpy file in the numpy save dir
    
        Details:
            fname: filename (in INPUT_NUMPY_DIR)

            returns: numpy array'''
    return np.load(INPUT_NUMPY_DIR + fname)

def open_midi(midi_path):
    '''Open a midi file and return a Stream object
    
        Details:
            midi_path: MIDI filename (in INPUT_MIDI_DIR)

            returns: Stream object'''
    mf = midi.MidiFile()
    mf.open(INPUT_MIDI_DIR + midi_path)
    mf.read()
    mf.close()
    return midi.translate.midiFileToStream(mf)

def key_properties(nt, off=-1):
    '''Get the properties of a particular note (pitch, offset, duration, and velocity)
    
        Details:
            nt: a Note object
            off: offset (-1 means find it yourself, uses offset explicitly if given)

            returns: an array of values'''
    # Use a specific offset (useful for chords, where the chord has an offset but the notes do not)
    if off != -1:
        return [nt.pitch.midi, off, nt.duration.quarterLength, nt.volume.velocity]
    else:
        return [nt.pitch.midi, nt.offset, nt.duration.quarterLength, nt.volume.velocity]

def extract_relevant_data(elements, verbose=False):
    '''Takes a bunch of elements and returns the normalized versions
    
        Details:
            elements: a collection of notes and chords

            returns: a list of notes (an array of arrays)'''
    all_notes = []
    for element in elements:
        # Treat note
        if isinstance(element, note.Note):
            all_notes.append(key_properties(element))
        # Treat chord
        elif isinstance(element, chord.Chord):
            # Get chord offset
            off = element.offset
            # Treat each note in chord
            for nt in element.notes:
                all_notes.append(key_properties(nt, off=off))
    # Normalize
    all_notes = normalize(all_notes, verbose=verbose)
    return all_notes

def normalize(all_notes, verbose=False):
    '''Some standardization (scale, translate so it starts at 0, round to int)
    
        Details:
            all_notes: an array of notes represented by arrays

            returns: an array of notes represented by arrays'''
    # Scale all note offsets and durations
    for nt in all_notes:
        nt[1] = nt[1] * SCALE
        nt[2] = nt[2] * SCALE
    offsets = [nt[1] for nt in all_notes]
    minimum = min(offsets)
    # Translate and cast to int
    for nt in all_notes:
        nt[0] = int(max(0, nt[0]))
        nt[1] = int(nt[1] - minimum)
        nt[2] = int(max(1, nt[2]))
        nt[3] = int(nt[3])
    # Key normalize
    all_notes = normalize_key(all_notes, verbose=verbose)
    return all_notes

def normalize_key(all_notes, verbose=False):
    '''Predicts the key from note distribution and transposes to C
    
    Details:
            all_notes: an array of notes represented by arrays

            returns: an array of notes represented by arrays'''
    frequencies = get_frequencies(all_notes)
    key = detect_key(frequencies, verbose=verbose)
    if key > 6:
        key = 12 - key
    all_notes = transpose(all_notes, -key)
    return all_notes

def get_frequencies(all_notes):
    '''Counts the frequency of each letter note
    
        Details:
            all_notes: an array of notes represented by arrays
            
            returns: a array containing the frequency of each note (0:C ... 11:B)'''
    note_frequencies = np.zeros(12)

    # loop through all_notes and tally the number of times each note occurs
    for note in all_notes:
        # First element of note is pitch
        pitch = note[0]
        # C1 = 24 and 24 % 12 = 0 so this works out
        scale_degree = pitch % 12
        # Add to dictionary
        note_frequencies[scale_degree] += 1

    # Divide every value in note_frequencies by the total number of notes to get the frequency
    for i in range(12):
        note_frequencies[i] /= len(all_notes)

    return note_frequencies

def transpose(all_notes, transpose_amt):
    '''Transposes (shifts up/down) all notes by a given amount
    
        Details:
            all_notes: an array of notes represented by arrays
            transpose_amt: integer representing number of semitones to transpose by (+ for up, - for down)
            
            returns: an array of notes represented by arrays'''
    for note in all_notes:
        note[0] += transpose_amt
    return all_notes

def _shift_by (arr, k):
    '''Shift all elements in an array in a circular manner

    Details:
        arr: an array
        k: number of "shifts"
        
        returns: an array'''
    return np.concatenate((arr[-k:], arr[:-k]))

# Helper function for reference of the entire list of canonical functions
def _get_canonical_scale():
    arr = np.zeros((12,12))
    major_scale_form = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]) / 7
    for i in range(12):
        key_frequencies = _shift_by(major_scale_form, i)
        arr[i] = key_frequencies
    return arr

def detect_key(input_frequencies, verbose=False):
    '''
    Detects which key is implied by the given note frequencies

    Details:
        input_frequencies: an array containing the frequency of each note (0:C ... 11:B)

        returns: an integer 0 - 11 representing the key of the music (0:C ... 11:B)
    '''
    canonical_scales = _get_canonical_scale() 

    max_index = -1
    max_similarity = -1
    
    for i, scale in enumerate(canonical_scales):
        similarity = cosine_similarity([input_frequencies], [scale])
        if similarity > max_similarity:
            max_index = i
            max_similarity = similarity
    if verbose:
        print("Key: {}".format(max_index))
    return max_index


def print_instruments(parts):
    '''Helpful I/O function
    
        Details:
            parts: a collection of Parts

            returns: None, just prints'''
    print('PRINTING PARTS')
    for i in range(len(parts)):
        print('Part ' + str(i) + ': ' + str(parts[i].getInstrument(returnDefault=False).instrumentName))

def featurize_notes(midi, mode='binary', piano_part=-1, verbose=False):
    '''Turns MIDI into a tensor (matrix, in this case)
    
        Details:
            midi: a collection of Parts
            mode: binary velocity or greyscale velocity (0-127)
            piano_part: index of piano Part (-1 if it doesn't need to be specified)

            returns: a 2D matrix, like a piano roll'''
    # If there's more than one part and we're not told which to use
    if len(midi.parts) != 1 and piano_part == -1:
        print_instruments(midi.parts)
        raise Exception('Expected only 1 part but got ' + str(len(midi.parts)) + ' instead')
    # Only one part
    elif len(midi.parts) == 1:
        elements = midi.parts[0].flat.notes
    # Use specified part
    else:
        elements = midi.parts[piano_part].flat.notes
    # Get tensor
    return get_tensor_from_part(elements, mode=mode, verbose=verbose)

def get_tensor_from_part(elements, mode, verbose=False):
    '''Transform a bunch of elements to a tensor
    
        Details:
            elements: a collection of Notes and Chords
            mode: binary velocity or greyscale velocity (0-127)

            returns: a 2D matrix, like a piano roll'''
    # Get notes (including notes from chords)
    all_notes = extract_relevant_data(elements, verbose=verbose)
    max_len = get_max_length(all_notes)
    tensor = np.zeros(shape=(128, max_len))
    # Add each note in
    for nt in all_notes:
        for i in range(nt[2]):
            # Include velocity or not
            if mode == 'binary':
                tensor[nt[0]][nt[1] + i] = 1
            elif mode == 'grayscale':
                tensor[nt[0]][nt[1] + i] = nt[3] / MAX_VEL
    # Bad for model
    tensor = remove_big_gaps(tensor)
    # Pianos don't have 127 notes, so we remove the 'fake' ones above and below
    tensor = cut_non_piano_notes(tensor, inverted=False)
    return tensor

def get_max_length(notes):
    '''Gets the the length of the song (max of offset + length)
    
        Details:
            notes: a list of notes
            returns: what the width of the final tensor must be'''
    maximum = max(notes, key=lambda x: x[1] + x[2])
    return maximum[1] + maximum[2]

def combine_into_rgb(tensor):
    '''Combine every three binary digits to an RGB value
    
        Details:
            tensor: a 2D matrix (piano roll)

            returns: a 3D tensor (height, width, 3 RGB values)'''
    shape = tensor.shape
    offset = 3 - (shape[1] % 3)
    # Need to add column(s) so width is divisible by 3
    if offset != 3:
        extra = np.zeros(shape=(shape[0], offset))
        tensor = np.hstack((tensor, extra))
    new_shape = tensor.shape
    tensor = np.reshape(tensor, (new_shape[0], int(new_shape[1] / 3), 3))
    return tensor

def get_image(tensor, show=False, mode='L'):
    '''Turn a tensor into an image
    
        Details:
            tensor: a tensor
            show: if True, displays the image
            mode: 'L' for B/W, 'RGB' for RGB

            returns: an image'''
    img = Image.fromarray(np.uint8(tensor * 255), mode)
    if show:
        img.show()
    return img

def save_image(image, fname):
    '''Save an image
    
        Details:
            image: an image
            fname: a filename

            returns: None'''
    image.save(fname)

def convert_midi_to_numpy_image(fname, piano_part=-1, verbose=False, images=True, do_grayscale=False):
    '''Top-level method: runs most of the pipeline from filename to numpy and images
    
        Details:
            fname: a filename (in INPUT_MIDI_DIR)
            piano_part: index of the piano Part (-1 means there's no need to specify)
            verbose: if True, shows debug
            images: if True, also saves images

            returns: None'''
    name = fname.split('.')[0]
    numpy_dir = INPUT_NUMPY_DIR + name + '/'
    image_dir = INPUT_IMAGE_DIR + name + '/'
    # Create relevant directories
    create_dir(numpy_dir, verbose=verbose)
    create_dir(image_dir, verbose=verbose)
    midi = open_midi(fname)
    
    # GET TENSORS
    binary_tensor = featurize_notes(midi, mode='binary', piano_part=piano_part, verbose=verbose)
    if verbose:
        print('Binary tensor shape:', binary_tensor.shape)
    
    binary_rgb_tensor = combine_into_rgb(binary_tensor)
    if verbose:
        print('Binary compressed (RGB) tensor shape:', binary_rgb_tensor.shape)

    if do_grayscale:
        grayscale_tensor = featurize_notes(midi, mode='grayscale', piano_part=piano_part, verbose=verbose)
        if verbose:
            print('Grayscale tensor size:', grayscale_tensor.shape)

        grayscale_rgb_tensor = combine_into_rgb(grayscale_tensor)
        if verbose:
            print('Grayscale compressed (RGB) tensor shape:', grayscale_rgb_tensor.shape)

    # GET IMAGES
    if images:
        binary_image = get_image(binary_tensor, mode='L')
        binary_rgb_image = get_image(binary_rgb_tensor, mode='RGB')
        if do_grayscale:
            grayscale_image = get_image(grayscale_tensor, mode='L')
            grayscale_rgb_image = get_image(grayscale_rgb_tensor, mode='RGB')
    
    # SAVE
    np.save(numpy_dir + name + '_bin.npy', binary_tensor)
    np.save(numpy_dir + name + '_bin_rgb.npy', binary_rgb_tensor)
    if do_grayscale:
        np.save(numpy_dir + name + '_gray.npy', grayscale_tensor)
        np.save(numpy_dir + name + '_gray_rgb.npy', grayscale_rgb_tensor)

    if images:
        save_image(binary_image, image_dir + name + '_bin.png')
        save_image(binary_rgb_image, image_dir + name + '_bin_rgb.png')
        if do_grayscale:
            save_image(grayscale_image, image_dir + name + '_gray.png')
            save_image(grayscale_rgb_image, image_dir + name + '_gray_rgb.png')

def create_dir(dirname, suppress_warnings=False, verbose=False):
    '''Create a directory if it doesn't exist
    
        Details:
            dirname: a directory name
            suppress_warnings: if True, doesn't warn about overriding

            returns: None'''
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    else:
        if not suppress_warnings and verbose:
            print('NOTE: Directory \'' + dirname + '\' already exists and may be overridden.')

def remove_big_gaps(tensor, threshold=SCALE*4):
    '''Remove entire measures of nothing
    
        Details:
            tensor: a 2D matrix (piano roll)
            threshold: how many pixels of nothing before cutting (defaults to a measure)
            
            returns: a 2D matrix (piano roll)'''
    shape = tensor.shape
    width = shape[1]
    for i in range(width):
        if np.all((tensor[:, i:(i + threshold)] == 0)):
            tensor = np.hstack((tensor[:, :i], tensor[:, (i+threshold):]))
    return tensor

def process_directory(dirname, verbose=False, override=False):
    '''High-level method for processing an entire directory of midi files
    
        Details:
            dirname: a directory name to crawl through
            verbose: if True, prints debug statements
            override: if True, doesn't replace existing output folders

            returns: None'''
    fnames = os.listdir(dirname)
    for fname in tqdm(fnames):
        name = fname.split('.')[0]
        if verbose:
            print('\nProcessing ' + name + '...')
        if not override and os.path.isdir(INPUT_NUMPY_DIR + name) and len(os.listdir(INPUT_NUMPY_DIR + name)) != 0:
            if verbose:
                print('Skipping \'' + name + '\' because there exists a directory in its name with files inside.')
            continue
        if name in piano_parts:
            piano_part = piano_parts[name]
        else:
            piano_part = -1
        convert_midi_to_numpy_image(fname, verbose=verbose, piano_part=piano_part)

def numpy_to_part(tensor, is88=True):
    '''Backwards process: convert a tensor to a midi part
    
        Details:
            tensor: a 2D matrix (piano roll)

            returns: a Part'''
    # Create a stream
    pt = stream.Part()
    count = 0
    shape = tensor.shape
    for key in range(shape[0]):
        for timestep in range(shape[1]):
            # Note or continuation of a note
            if tensor[key][timestep] != 0:
                if timestep == 0 or tensor[key][timestep - 1] == 0:
                    if is88:
                        current_note = note.Note(key + 21)
                    else:
                        current_note = note.Note(key)
                    current_note.volume.velocity = tensor[key][timestep] * MAX_VEL
                    current_note.offset = timestep / SCALE
                count += 1
            # End of a note
            if tensor[key][timestep] == 0 or timestep == shape[1] - 1:
                if count != 0:
                    current_note.duration.quarterLength = count / SCALE
                    pt.insert(current_note.offset, current_note)
                    count = 0
    return pt

def part_to_midi(part, fname):
    '''Turn a part into a MIDI file and save it
    
        Details:
            part: a Part

            returns: None'''
    mf = midi.translate.streamToMidiFile(part)
    mf.open(OUTPUT_MIDI_DIR + fname + '.mid', 'wb')
    mf.write()
    mf.close()

def spread_out_rgb(tensor):
    '''Flatten a tensor with RGB values into a 2D B/W matrix (3x as long)
    
        Details:
            tensor: a 3D tensor (height, width, 3 RGB values)

            returns: a 2D matrix (piano roll)'''
    shape = tensor.shape
    tensor = np.reshape(tensor, (shape[0], shape[1] * shape[2]))
    return tensor

def build_dataset(verbose=False, size=NUM_KEYS, save=False, fname='data'):
    '''Concatenate a bunch of files (in the numpy save dir) into one binary dataset
    
        Details:
            verbose: if True, prints debug statements
            size: height of the piano roll (defaults to 88)
            save: save the dataset to file
            fname: name of file if saved

            returns: a long 2D matrix (piano roll)'''
    long_tensor = np.zeros(shape=(size, 1))
    for dirname in tqdm(os.listdir(INPUT_NUMPY_DIR)):
        long_tensor = np.hstack((long_tensor, load_numpy(dirname + '/' + dirname + '_bin.npy')))
    if verbose:
        print('Read', len(os.listdir(INPUT_NUMPY_DIR)), 'files and concatenated', long_tensor.shape[1] - 1, 'vectors.')
    long_tensor = long_tensor[:, 1:]
    if save:
        np.save('{}{}.npy'.format(BATCH_SAVE_DIR, fname), long_tensor)
    return long_tensor

def cut_non_piano_notes(tensor, inverted=False):
    '''Remove keys that aren't on a real piano
    
        Note that piano rolls with 0-indexing look inverted by default, but are NOT inverted

        Details:
            tensor: a 2D matrix (piano roll)
            inverted: if True, this means the piano roll has been up-down inverted

            returns: a 2D matrix (piano roll)'''
    if not inverted:
        return tensor[21:109, ...]
    else:
        return tensor[19:107, ...]

def image_to_np(image, verbose=False, rotate=True):
    '''Convert an image back into a numpy array
    
        Details:
            image: an image
            verbose: if True, shows the image (and other debugs)
            rotate: if True, rotates the tensor once counterclockwise

            returns: a 2D tensor (piano roll)'''
    if verbose:
        print(image.format)
        print(image.size)
        print(image.mode)
        image.show()
    tensor = np.asarray(image)
    if verbose:
        print(tensor.shape)
    if image.mode == 'RGB':
        tensor = flatten_rgb(tensor)
    if rotate:
        tensor = np.transpose(tensor)
    tensor = pad_to_128(tensor, inverted=False)
    return tensor
    
def flatten_rgb(tensor):
    '''Ignore color and just make each pixel B/W
    
        Details:
            tensor: a 3D tensor (height, width, 3 RGB values)
            
            returns: a 2D matrix (piano roll)'''
    shape = tensor.shape
    new_tensor = np.zeros(shape=(shape[0], shape[1]))
    for i, row in enumerate(tensor):
        for j, pixel in enumerate(row):
            # Somewhat arbitrary threshold that R, G, and B values are all > half the max pixel strength
            if (pixel[0] > MAX_PIXEL_STRENGTH/2) and (pixel[1] > MAX_PIXEL_STRENGTH/2) and (pixel[2] > MAX_PIXEL_STRENGTH/2):
                new_tensor[i][j] = 1
            else:
                new_tensor[i][j] = 0
    return new_tensor

def pad_to_128(tensor, inverted=False):
    '''Pad to the top and bottom of the array so it's 128 tall again (add non-piano notes back in)
    
        Note that piano rolls with 0-indexing look inverted by default, but are NOT inverted

        Details:
            tensor: a 2D matrix (piano roll)
            inverted: if True, this means the piano roll has been up-down inverted

            returns: a 2D matrix (piano roll)'''
    bottom = np.zeros(shape=(21, tensor.shape[1]))
    top = np.zeros(shape=(19, tensor.shape[1]))
    if not inverted:
        return np.vstack((bottom, tensor, top))
    else:
        return np.vstack((top, tensor, bottom))

def save_image88(fname, suffix=0, verbose=False, subdir='', rotate=True):
    '''Convert an image (88 pixels tall) to a midi file and save it
    
        Details:
            fname: a filename for the input image
            suffix: a suffix to append to each saved filename
            verbose: if True, shows the image
            subdir: a directory name for a folder inside OUTPUT_IMAGE_DIR
            rotate: if True, rotates the piano roll once counterclockwise

            returns: None'''
    image = Image.open(OUTPUT_IMAGE_DIR + subdir + fname)
    tensor = image_to_np(image, verbose=verbose, rotate=rotate)
    part = numpy_to_part(tensor)
    part_to_midi(part, subdir + '/' + 'generated_midi_' + str(suffix))

def plot_history(history):
    '''Plots the loss from a keras history
    
        Details:
            history: a history dictionary from keras model.fit()

            returns: None'''
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def remove_leading_zeros(arr):
    '''Removes rows of all 0s at the beginning of an array
    
        Details:
            arr: a numpy array (likely in (timestep, features))

            returns: a numpy array'''
    cut_amt = -1
    for i in range(len(arr)):
        if max(arr[i]) == 0:
            cut_amt = i
        else:
            break
    return arr[cut_amt+1:]

def get_phrases(piano_roll, min_quarter_notes=2, max_quarter_notes=8, quarter_note_split = 0.25):
    '''Converts a piano roll into an array of phrases using the Hartono method

        Note that the phrases have 89 features to account for an endsequence bit (bit 88)

        TODO: Trash phrases that go over max and wait for a new break?
    
        Details:
            piano_roll: a numpy array (likely in (features, timestep))
            min_quarter_notes: the minimum number of quarter notes a phrase can be
            max_quarter_notes: the maximum number of quarter notes a phrase can be
            quarter_note_split: how many quarter notes of rest before calling a new phrase

            returns: a numpy array in (phrases, timesteps, features)'''
    # (timestep, features)
    p_r = np.transpose(piano_roll)
    # Count no. of times we hit cutoff
    cutoff_counter = 0
    # boolean to see what state (cut-off or not) we are in
    in_cutoff_state = False
    # Pad piano roll to 89
    piano_roll = np.zeros((p_r.shape[0], NUM_KEYS + 1))
    piano_roll[:p_r.shape[0], :p_r.shape[1]] = p_r
    # Amount after which we look for a gap
    threshold = SCALE * min_quarter_notes
    # Arbitrarily choose an eighth note as enough of a rest
    gap_threshold = int(SCALE * quarter_note_split)
    # At some point you need to stop
    cutoff = SCALE * max_quarter_notes
    phrases = np.zeros(shape=(1, cutoff, NUM_KEYS + 1))
    # End token
    endtoken = np.zeros((1, NUM_KEYS + 1))
    endtoken[0, NUM_KEYS] = 1
    # Iterate through timesteps
    pause_counter = 0
    current_phrase = np.zeros((1, NUM_KEYS + 1))
    for i in range(len(piano_roll)):
        # Get vector representing all the notes at this timestep
        timestep = piano_roll[i]
        # Get highest value (interested in if it's 0)
        maximum = max(timestep)
        hit_cutoff = len(current_phrase) >= cutoff - 1   

        if maximum == 0: 
            pause_counter += 1
            # If gap is long enough to make a phrase and phrase iteself is long enough OR hit cutoff
            long_break = (pause_counter >= gap_threshold and len(current_phrase) >= threshold)
            if long_break or hit_cutoff:
                if long_break and in_cutoff_state:
                    in_cutoff_state = False

                else:
                    # Take out zeros at beginning
                    current_phrase = remove_leading_zeros(current_phrase)
                    # If we stopped from a long break, remove that long break
                    current_phrase = current_phrase[1:-gap_threshold+1]
                    # Add endtoken
                    current_phrase = np.concatenate((current_phrase, endtoken), axis=0)
                    # Pad to cutoff
                    padded_current_phrase = np.zeros((cutoff, NUM_KEYS + 1))
                    padded_current_phrase[:current_phrase.shape[0], :current_phrase.shape[1]] = current_phrase
                    # Add this phrase to running phrases
                    phrases = np.concatenate((phrases, [padded_current_phrase]), axis=0) 
                    if hit_cutoff:
                        in_cutoff_state = True
                # Reset variables
                current_phrase = np.zeros((1, NUM_KEYS + 1))
                pause_counter = 0
        else:
            # Just reset the counter
            pause_counter = 0

        if not in_cutoff_state:
            # Add this timestep to running timesteps
            current_phrase = np.concatenate((current_phrase, [timestep]), axis=0)

    # print('Fraction at cutoff: ', cutoff_counter / (len(phrases)-1))
    return phrases[1:]

def get_melody(path):
    """
    Extracts the melody from an array of notes. 

    Details:
        path: path of numpy array to extract the melody from
        returns: an array containing only the melody
    """
    # lower allowed melody note (middle C)
    cutoff_number = 60
    # largest allowed melodic jump (one octave)
    largest_jump = 12
    melody = np.transpose(np.zeros(np.load(path).shape))
    arr = np.transpose(np.load(path))
    count = 0

    for step in arr:
        # get the highest note played at the given timestep
        high_note = np.max(np.where(step == 1.0)[0]) if 1.0 in step else 0
        # create a timestep for the melody containing only the highest note
        melody_step = np.zeros((88,))
        if high_note != 0:
            melody_step[high_note] = 1

        if count == 0:
            # append highest note to melody only if note is at or above cutoff note
            if high_note >= cutoff_number:
                melody[count] = melody_step
            else:
                melody[count] = np.zeros((88,))
            count += 1
        else:
            # get the melody note from the previous timestep
            prev_note = np.max(np.where(melody[count-1] == 1.0)[0]) if 1.0 in melody[count-1] else 0
            # append highest note to melody if its higher than the previous melody note
            if high_note >= prev_note and high_note >= cutoff_number:
                melody[count] = melody_step
            # append highest note to melody if its an octave or less below the previous melody note 
            elif prev_note - high_note <= largest_jump and high_note >= cutoff_number:
                melody[count] = melody_step
            else:
                melody[count] = np.zeros((88,))
            count += 1
            
    return np.transpose(melody)
        

        


    
