import music21
from music21 import converter, corpus, instrument, midi, note, chord, pitch, stream
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import sys
import os
import copy

# Save directories
DATA_DIR = 'midi_files/'
IMAGE_SAVE_DIR = 'output_images/'
NUMPY_SAVE_DIR = 'numpy_files/'
MIDI_SAVE_DIR = 'output_midi/'
MODEL_SAVE_DIR = 'keras_models/'
BATCH_SAVE_DIR = 'processed_data/'
GENERATED_SAVE_DIR = 'generated_images/'
# Quarter note scale (1 quarter note is split up into 12 pixels for accuracy)
SCALE = 12
# Maximum velocity
MAX_VEL = 127
# Highest key
MAX_KEY = 127
# Highest pixel value (greyscale and RGB)
MAX_PIXEL_STRENGTH = 255
# np.set_printoptions(threshold=sys.maxsize)

# Load a numpy file in the numpy save dir
def load_numpy(fname):
    return np.load(NUMPY_SAVE_DIR + fname)

# Open a midi file and return a Stream object
def open_midi(midi_path, remove_drums=False):
    mf = midi.MidiFile()
    mf.open(DATA_DIR + midi_path)
    mf.read()
    mf.close()
    if (remove_drums):
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]
    return midi.translate.midiFileToStream(mf)

# Get the properties of a particular note (pitch, offset, duration, and velocity)
def key_properties(nt, off=-1):
    # Use a specific offset (useful for chords, where the chord has an offset but the notes do not)
    if off != -1:
        return [nt.pitch.midi, off, nt.duration.quarterLength, nt.volume.velocity]
    else:
        return [nt.pitch.midi, nt.offset, nt.duration.quarterLength, nt.volume.velocity]

# Takes a bunch of elements and returns the normalized versions
def extract_relevant_data(elements):
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
    return normalize(all_notes)

# Some standardization
def normalize(all_notes):
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

# Helpful I/O function
def print_instruments(parts):
    print('PRINTING PARTS')
    for i in range(len(parts)):
        print('Part ' + str(i) + ': ' + str(parts[i].getInstrument(returnDefault=False).instrumentName))

# Transform midi to a tensor
def featurize_notes(midi, mode='binary', piano_part=-1):
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

# Transform a bunch of elements to a tensor
def get_tensor_from_part(elements, mode):
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
    maximum = max(notes, key=lambda x: x[1] + x[2])
    return maximum[1] + maximum[2]

# Combine every three binary digits to an RGB value
def combine_into_rgb(tensor):
    shape = tensor.shape
    offset = 3 - (shape[1] % 3)
    # Need to add column(s) so width is divisible by 3
    if offset != 3:
        extra = np.zeros(shape=(shape[0], offset))
        tensor = np.hstack((tensor, extra))
    new_shape = tensor.shape
    tensor = np.reshape(tensor, (new_shape[0], int(new_shape[1] / 3), 3))
    return tensor

# Turn a tensor into an image
def get_image(tensor, show=False, mode='L'):
    img = Image.fromarray(np.uint8(tensor * 255), mode)
    if show:
        img.show()
    return img

def save_image(image, fname):
    image.save(fname)

# Top-level method- runs most of the pipeline from filename to numpy and images
def convert_midi_to_numpy_image(fname, piano_part=-1, verbose=False, images=True):
    name = fname.split('.')[0]
    numpy_dir = NUMPY_SAVE_DIR + name + '/'
    image_dir = IMAGE_SAVE_DIR + name + '/'
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

# Create a directory if it doesn't exist
def create_dir(dirname):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    else:
        print('NOTE: Directory \'' + dirname + '\' already exists and may be overridden.')

# Remove entire measures of nothing
def remove_big_gaps(tensor, threshold=SCALE*4):
    shape = tensor.shape
    width = shape[1]
    for i in range(width):
        if np.all((tensor[:, i:(i + threshold)] == 0)):
            tensor = np.hstack((tensor[:, :i], tensor[:, (i+threshold):]))
    return tensor

# High-level method for processing an entire directory of midi files
def process_directory(dirname, piano_parts, verbose=False, override=False):
    fnames = os.listdir(dirname)
    for fname in fnames:
        name = fname.split('.')[0]
        if verbose:
            print('\nProcessing ' + name + '...')
        if not override and os.path.isdir(NUMPY_SAVE_DIR + name) and len(os.listdir(NUMPY_SAVE_DIR + name)) != 0:
            if verbose:
                print('Skipping \'' + name + '\' because there exists a directory in its name with files inside.')
            continue
        if name in piano_parts:
            piano_part = piano_parts[name]
        else:
            piano_part = -1
        convert_midi_to_numpy_image(fname, verbose=verbose, piano_part=piano_part)

# Backwards process: convert a tensor to a midi part
def numpy_to_part(tensor):
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
                    current_note.volume.velocity = tensor[key][timestep] * 127
                    current_note.offset = timestep / 12
                count += 1
            # End of a note
            if tensor[key][timestep] == 0 or timestep == shape[1] - 1:
                if count != 0:
                    current_note.duration.quarterLength = count / 12
                    pt.insert(current_note.offset, current_note)
                    count = 0
    return pt

# Turn a part into a MIDI file and save it
def part_to_midi(part, fname):
    mf = midi.translate.streamToMidiFile(part)
    mf.open(MIDI_SAVE_DIR + fname + '.mid', 'wb')
    mf.write()
    mf.close()

# Flatten a tensor with RGB values into a 2D matrix with just BW (3x as long)
def spread_out_rgb(tensor):
    shape = tensor.shape
    tensor = np.reshape(tensor, (shape[0], shape[1] * shape[2]))
    return tensor

# Concatenate a bunch of files (in the numpy save dir) into one big file
def combine_into_long_tensor(mode='bin', verbose=False, size=88):
    long_tensor = np.zeros(shape=(size, 1))
    for dirname in os.listdir(NUMPY_SAVE_DIR):
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
        print('Read', len(os.listdir(NUMPY_SAVE_DIR)), 'files and concatenated', long_tensor.shape[1] - 1, 'vectors.')
    return long_tensor[:, 1:]

# Remove keys that aren't on a real piano (note: 0 is low pitch and 127 is high pitch, so images look upside down to 
# you are NOT inverted)
def cut_non_piano_notes(tensor, inverted=False):
    if not inverted:
        return tensor[21:109, ...]
    else:
        return tensor[19:107, ...]

# Convert an image back into a numpy array
def image_to_np(image, verbose=False, rotate=True):
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
    
# Ignore color and just make it 0 or 1
def flatten_rgb(tensor):
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

# Pad to the top and bottom of the array so it's 128 tall again (add non-piano notes back in)
def pad_to_128(tensor, inverted=False):
    bottom = np.zeros(shape=(21, tensor.shape[1]))
    top = np.zeros(shape=(19, tensor.shape[1]))
    if not inverted:
        return np.vstack((bottom, tensor, top))
    else:
        return np.vstack((top, tensor, bottom))

# Convert an image (88 pixels tall) to a midi file and save it
def save_image88(fname, suffix=0, verbose=False, subdir='', rotate=True):
    image = Image.open(GENERATED_SAVE_DIR + subdir + fname)
    tensor = image_to_np(image, verbose=verbose, rotate=rotate)
    part = numpy_to_part(tensor)
    part_to_midi(part, subdir + '/' + 'generated_midi_' + str(suffix))