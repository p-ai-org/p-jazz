# Import statements
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from tqdm import tqdm
from util import *
import math
import numpy as np

# How far back the LSTM gets to see
n_steps = 24

def split_sequences(sequences, n_steps):
    '''
    Split a multivariate sequence into samples

    e.g. [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]] with n = 2 returns:

    First X: [[1, 2, 3], [4, 5, 6]]
    First y: [7, 8, 9]

    Second X: [[4, 5, 6], [7, 8, 9]]
    Second y: [10, 11, 12]

    Returns (X, y)
    '''
    X, y = list(), list()
    for i in range(len(sequences)):
		# Find the end of this pattern
        end_ix = i + n_steps
        # Check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def load_dataset(fname):
    dataset = np.load('{}{}.npy'.format(BATCH_SAVE_DIR, fname))
    # Rotate so it's [timesteps, features]
    dataset = np.rot90(dataset, k=3)
    return dataset

def define_model(n_features):
    # Define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(100, activation='relu'))
    # Which activation function to use?
    model.add(Dense(n_features, activation='relu'))
    return model

def train_model(save=False, dataset='full_1', fname='model', epochs=10):
    # Load data
    data = load_dataset(dataset)
    # Split into X and y
    X, y = split_sequences(data, n_steps)
    # n_features is always 88 for us
    n_features = X.shape[2]
    model = define_model(n_features)
    # Compile and train
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=1)
    if save:
        model.save('{}{}'.format(MODEL_SAVE_DIR, fname))

def activate(vec, threshold=0.1):
    '''
    Predictions are not binary, so we set some linear threshold above which we round to 1
    '''
    return np.array([(lambda x: 1 if x>threshold else 0)(x) for x in vec])

def make_predictions(modelname, dataset='full_1', start=0, predict_amount=(4*n_steps), show=False):
    # Load model and data
    model = load_model('{}{}'.format(MODEL_SAVE_DIR, modelname))
    data = load_dataset(dataset)
    # Starting point for model to start predictin
    seed = data[start:start+n_steps, :]
    # Stochastically add timesteps
    for i in tqdm(range(predict_amount)):
        # Get seed
        X_input = seed[i:i+n_steps, :]
        X_input = np.reshape(X_input, newshape=(1, X_input.shape[0], X_input.shape[1]))
        # Apply activation and append to seed
        yhat = [activate(model.predict(X_input)[0])]
        seed = np.vstack((seed, yhat))
    if show:
        get_image(seed, show=True)
    return seed

train_model(save=True, fname='lstm1', epochs=1)

# make_predictions('lstm1', dataset='full_1', start=242, predict_amount=96, show=True)