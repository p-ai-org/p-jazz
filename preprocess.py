from util import *
import math
# np.set_printoptions(threshold=sys.maxsize)

# '''
height=88
width=88
CAP = 200000

full_dataset = combine_into_long_tensor(mode='bin', verbose=True, size=height)
full_shape = full_dataset.shape
print('Shape:', full_shape)

step = 1
# batch_size = 19600
batch_size = math.floor((full_shape[1] - width) / step)
batch_size = min(CAP, batch_size)

batched_data = np.zeros(shape=(batch_size, width, height))

for i in range(batch_size):
    index = i * step
    sample = full_dataset[:, index : index+width]
    # Rotate
    batched_data[i] = np.rot90(sample, k=3, axes=(0, 1))
print(batched_data.shape)

np.save('processed_data/batches_3.npy', batched_data)
# '''

# tensor = np.load('processed_data/batches_1.npy')
# get_image(tensor[0], show=True)