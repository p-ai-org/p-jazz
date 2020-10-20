import music21
from music21 import converter, corpus, instrument, midi, note, chord, pitch, stream
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import sys
import os
import copy

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

def extract_relevant_data(elements):
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
    all_notes = normalize(all_notes)
    return all_notes

def normalize(all_notes):
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
    return all_notes

def get_frequencies(all_notes):
    '''Counts the frequency of each letter note
    
        Details:
            all_notes: an array of notes represented by arrays
            
            returns: a dictionary containing the frequency of each note (0:C ... 11:B)'''
    note_frequencies = {}
    
    # initialize values in note_frequencies
    for i in range(12):
        note_frequencies[i] = 0

    # loop through all_notes and tally the number of times each note occurs
    for note in all_notes:
        for i in range(12):
            if note[0] in range(21 + i, 127, 12):
                note_frequencies[(i + 9) % 12] += 1

    # divide every value in note_frequencues by the total number of notes to get the frequency
    for i in range(12):
        note_frequencies[i] /= len(all_notes)

    return note_frequencies

def transpose(all_notes, transpose_amt):
    '''Transposes (shifts up/down) all notes by a given amount
    
        Details:
            all_notes: an array of notes represented by arrays
            transpose_amt: integer representing number of semitones to transpose by (+ for up, - for down)
            
            returns: an array of notes represented by arrays'''
    return all_notes

def detect_key(frequencies):
    '''Detects which key is implied by the given note frequencies
    
        Details:
            frequencies: a dictionary containing the frequency of each note (0:C ... 11:B)
            
            reeturns: an integer 0 - 11 representing the key of the music (0:C ... 11:B)'''
    return 0

def print_instruments(parts):
    '''Helpful I/O function
    
        Details:
            parts: a collection of Parts

            returns: None, just prints'''
    print('PRINTING PARTS')
    for i in range(len(parts)):
        print('Part ' + str(i) + ': ' + str(parts[i].getInstrument(returnDefault=False).instrumentName))

def featurize_notes(midi, mode='binary', piano_part=-1):
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
    return get_tensor_from_part(elements, mode=mode)

def get_tensor_from_part(elements, mode):
    '''Transform a bunch of elements to a tensor
    
        Details:
            elements: a collection of Notes and Chords
            mode: binary velocity or greyscale velocity (0-127)

            returns: a 2D matrix, like a piano roll'''
    # Get notes (including notes from chords)
    all_notes = extract_relevant_data(elements)
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

def convert_midi_to_numpy_image(fname, piano_part=-1, verbose=False, images=True):
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
    create_dir(numpy_dir)
    create_dir(image_dir)
    midi = open_midi(fname)
    
    # GET TENSORS
    binary_tensor = featurize_notes(midi, mode='binary', piano_part=piano_part)
    if verbose:
        print('Binary tensor shape:', binary_tensor.shape)
    
    binary_rgb_tensor = combine_into_rgb(binary_tensor)
    if verbose:
        print('Binary compressed (RGB) tensor shape:', binary_rgb_tensor.shape)

    grayscale_tensor = featurize_notes(midi, mode='grayscale', piano_part=piano_part)
    if verbose:
        print('Grayscale tensor size:', grayscale_tensor.shape)

    grayscale_rgb_tensor = combine_into_rgb(grayscale_tensor)
    if verbose:
        print('Grayscale compressed (RGB) tensor shape:', grayscale_rgb_tensor.shape)

    # GET IMAGES
    if images:
        binary_image = get_image(binary_tensor, mode='L')
        binary_rgb_image = get_image(binary_rgb_tensor, mode='RGB')
        grayscale_image = get_image(grayscale_tensor, mode='L')
        grayscale_rgb_image = get_image(grayscale_rgb_tensor, mode='RGB')
    
    # SAVE
    np.save(numpy_dir + name + '_bin.npy', binary_tensor)
    np.save(numpy_dir + name + '_bin_rgb.npy', binary_rgb_tensor)
    np.save(numpy_dir + name + '_gray.npy', grayscale_tensor)
    np.save(numpy_dir + name + '_gray_rgb.npy', grayscale_rgb_tensor)

    if images:
        save_image(binary_image, image_dir + name + '_bin.png')
        save_image(binary_rgb_image, image_dir + name + '_bin_rgb.png')
        save_image(grayscale_image, image_dir + name + '_gray.png')
        save_image(grayscale_rgb_image, image_dir + name + '_gray_rgb.png')

def create_dir(dirname, suppress_warnings=False):
    '''Create a directory if it doesn't exist
    
        Details:
            dirname: a directory name
            suppress_warnings: if True, doesn't warn about overriding

            returns: None'''
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    else:
        if not suppress_warnings:
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

def process_directory(dirname, piano_parts, verbose=False, override=False):
    '''High-level method for processing an entire directory of midi files
    
        Details:
            dirname: a directory name to crawl through
            piano_parts: a dictionary that matches filenames to their respective piano part
            verbose: if True, prints debug statements
            override: if True, doesn't replace existing output folders

            returns: None'''
    fnames = os.listdir(dirname)
    for fname in fnames:
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

def numpy_to_part(tensor):
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

def combine_into_long_tensor(mode='bin', verbose=False, size=88):
    '''Concatenate a bunch of files (in the numpy save dir) into one big file
    
        Details:
            mode:
                'bin' for binary (0 or 255)
                'bin rgb' for RGB with only min or max values (e.g. 255 R, 0 G, 255 B)
                'gray' for grayscale (0 - 255)
                'gray rgb' for continuous RGB values (e.g. 244 R, 21 G, 93 B)
            verbose: if True, prints debug statements
            size: height of the piano roll (defaults to 88)

            returns: a long 2D matrix (piano roll)'''
    long_tensor = np.zeros(shape=(size, 1))
    for dirname in os.listdir(INPUT_NUMPY_DIR):
        if mode == 'bin':
            long_tensor = np.hstack((long_tensor, load_numpy(dirname + '/' + dirname + '_bin.npy')))
        elif mode == 'bin rgb':
            long_tensor = np.hstack((long_tensor, load_numpy(dirname + '/' + dirname + '_bin_rgb.npy')))
        elif mode == 'gray':
            long_tensor = np.hstack((long_tensor, load_numpy(dirname + '/' + dirname + '_gray.npy')))
        elif mode == 'gray rgb':
            long_tensor = np.hstack((long_tensor, load_numpy(dirname + '/' + dirname + '_gray_rgb.npy')))
        else:
            raise Exception('Unrecognized mode \'' + mode + '\'')
    if verbose:
        print('Read', len(os.listdir(INPUT_NUMPY_DIR)), 'files and concatenated', long_tensor.shape[1] - 1, 'vectors.')
    return long_tensor[:, 1:]

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
        tensor = np.rot90(tensor, k=1, axes=(0, 1))
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