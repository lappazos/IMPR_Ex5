import random
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

FILE_PROBLEM = "File Problem"

GREYSCALE = 1

MAX_INTENSITY = 255


def normalize_0_to_1(im):
    """
    normalize picture
    :param im: image in range 0-255
    :return: image in range [0,1]
    """
    if im.dtype != np.float64:
        im = im.astype(np.float64)
        im /= MAX_INTENSITY
    return im


def read_image(filename, representation):
    """
    This function returns an image, make sure the output image is represented by a matrix of type
    np.float64 with intensities (either grayscale or RGB channel intensities) normalized to the range [0, 1].
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2).
    :return: an image
    """
    im = None
    try:
        im = imread(filename)
    except Exception:  # internet didnt have specific documentation regarding the exceptions this func throws
        print(FILE_PROBLEM)
        exit()
    im = normalize_0_to_1(im)
    if representation == GREYSCALE:
        return rgb2gray(im)
    return im.astype(np.float64)


def crop(im, height, width, coupled_im=None):
    im_height, im_width, _ = im.shape
    max_row = im_height - height
    max_column = im_width - width
    row = random.randrange(max_row + 1)
    column = random.randrange(max_column + 1)
    if not coupled_im:
        return im[row:row + height, column:column + height, :]
    return im[row:row + height, column:column + height, :], coupled_im[row:row + height, column:column + height, :]


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    height = crop_size[0]
    width = crop_size[1]
    im_dict = {}
    while True:
        source_batch = np.empty((batch_size, height, width, 1))
        target_batch = np.empty((batch_size, height, width, 1))
        for i in range(batch_size):
            file = random.choice(filenames)
            if file in im_dict:
                im = im_dict[file]
            else:
                im = read_image(file, GREYSCALE)
                im_dict[file] = read_image(file, GREYSCALE)
            corrupted_im = corruption_func(crop(im, 3 * height, 3 * width))
            target_batch[i, :, :, :], source_batch[i, :, :, :] = crop(im, height, width, corrupted_im)
        yield source_batch, target_batch


def resblock(input_tensor, num_channels):
    a = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    b = Activation('relu')(a)
    c = Conv2D(num_channels, (3, 3), padding='same')(b)
    d = Add()([input_tensor, c])
    return Activation('relu')(d)


def build_nn_model(height, width, num_channels, num_res_blocks):
    a = Input(shape=(height, width, 1))
    b = Conv2D(num_channels, (3, 3), padding='same')(a)
    res_block_input = Activation('relu')(b)
    for block in range(num_res_blocks):
        res_block_input = resblock(res_block_input, num_channels)
    c = Conv2D(num_channels, (3, 3), padding='same')(res_block_input)
    d = Add()([a, c])
    return Model(inputs=a, outputs=d)


def train_model(model, images, corruption_func, batch_size,
                steps_per_epoch, num_epochs, num_valid_samples):
    images = random.shuffle(images)
    fraction = int(0.8 * len(images))
    train_set = images[0:fraction]
    test_set = images[fraction:]
    train_data_set_generator = load_dataset(train_set, batch_size, corruption_func,
                                            (model.input_shape[1], model.input_shape[2]))
    test_data_set_generator = load_dataset(test_set, batch_size, corruption_func,
                                           (model.input_shape[1], model.input_shape[2]))
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(train_data_set_generator, steps_per_epoch, num_epochs, validation_data=test_data_set_generator,
                        validation_steps=num_valid_samples)


