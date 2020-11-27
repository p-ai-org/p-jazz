from util import *
import argparse
import os

parser = argparse.ArgumentParser(description='Setup the p-jazz repository')
parser.add_argument('mode', help="Use 'directories' to setup the necessary folders and 'process' to process MIDI files")
parser.add_argument('-v', '--verbose', help="Set verbosity of output", action='store_true')
parser.add_argument('-o', '--overwrite', help="Overwrite existing numpy and image directories when processing", action='store_true')
parser.add_argument('--midi', help="Location of MIDI files. Defaults to INPUT_MIDI_DIR specified in util.py", default=INPUT_MIDI_DIR)
args = parser.parse_args()

mode = args.mode

if mode == 'directories':
  print(':: BUILDING REQUIRED DIRECTORIES ::')

  dirs = [INPUT_MIDI_DIR, INPUT_IMAGE_DIR, INPUT_NUMPY_DIR, INPUT_TEXT_DIR, OUTPUT_MIDI_DIR, MODEL_SAVE_DIR, NUMPY_CORPUS_DIR, OUTPUT_IMAGE_DIR]

  for dirname in dirs:
    if not os.path.isdir(dirname):
      os.mkdir(dirname)
    else:
      print("{} already exists, skipping".format(dirname))

  print(':: DONE ::')
elif mode == 'process':
  print(':: PROCESSING MIDI FILES ::')
  process_directory(args.midi, verbose=args.verbose, override=args.overwrite)
  print(':: BUILDING NUMPY CORPUS ::')
  build_dataset(verbose=args.verbose, size=88, save=True, fname='corpus')
  print(':: BUILDING PHRASE CORPUS ::')
  tensor = np.load('{}{}'.format(NUMPY_CORPUS_DIR, 'corpus.npy'))
  phrases = get_phrases(tensor, max_quarter_notes=8, quarter_note_split=(1/2))
  np.save('phrases.npy', phrases)
  print(':: DONE ::')
else:
  print("Mode '{}' not recognized. Please use 'directories' or 'process'.".format(mode))