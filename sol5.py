import random
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from tensorflow.keras.layers import Input, Conv2D, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import sol5_utils
from scipy.ndimage.filters import convolve

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
    """
    crop a specific size window from an image or a couple of images with same size, which is chosen in a random location
    :param im: image to crop
    :param height: window height
    :param width: window width
    :param coupled_im: coupled image to crop
    :return: if coupled_im is None - cropped image, otherwise - a couple of cropped images
    """
    im_height, im_width = im.shape
    max_row = im_height - height
    max_column = im_width - width
    row = random.randrange(max_row + 1)
    column = random.randrange(max_column + 1)
    if coupled_im is None:
        return im[row:row + height, column:column + height]
    return im[row:row + height, column:column + height], coupled_im[row:row + height, column:column + height]


im_dict = {}


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    outputs a data_generator, a Python’s generator object which outputs random tuples of the form
    (source_batch, target_batch), where each output variable is an array of shape (batch_size, height,
    width, 1) , target_batch is made of clean images, and source_batch is their respective randomly corrupted version
    according to corruption_func(im)
    :param filenames: A list of filenames of clean images
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent
    :param corruption_func: A function receiving a numpy’s array representation of an image as a single argument,
    and returns a randomly corrupted version of the input image
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract
    :return: generator
    """
    height = crop_size[0]
    width = crop_size[1]
    while True:
        source_batch = np.empty((batch_size, height, width, 1))
        target_batch = np.empty((batch_size, height, width, 1))
        for i in range(batch_size):
            file = random.choice(filenames)
            if file in im_dict:
                im = im_dict[file]
            else:
                im = read_image(file, GREYSCALE)
                im_dict[file] = im
            im = crop(im, 3 * height, 3 * width)
            corrupted_im = corruption_func(im)
            target_batch[i, :, :, 0], source_batch[i, :, :, 0] = crop(im, height, width, corrupted_im)
        yield source_batch - 0.5, target_batch - 0.5


def resblock(input_tensor, num_channels):
    """
    takes as input a symbolic input tensor and the number of channels for each of its
    convolutional layers, and returns the symbolic output tensor of the layer configuration
    :param input_tensor: input tensor
    :param num_channels: num of channels for  convolution
    :return: residual block sub-model
    """
    a = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    b = Activation('relu')(a)
    c = Conv2D(num_channels, (3, 3), padding='same')(b)
    d = Add()([input_tensor, c])
    return Activation('relu')(d)


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    return an untrained Keras model (do not call the compile() function!),
    with input dimension the shape of (height, width, 1), and all convolutional layers (including residual blocks) with
    number of output channels equal to num_channels, except the very last convolutional
    layer which should have a single output channel.
    :param height: input shape height
    :param width: input shape width
    :param num_channels: num of channels for  convolution
    :param num_res_blocks: The number of residual blocks
    :return: Model
    """
    a = Input(shape=(height, width, 1))
    b = Conv2D(num_channels, (3, 3), padding='same')(a)
    res_block_input = Activation('relu')(b)
    for block in range(num_res_blocks):
        res_block_input = resblock(res_block_input, num_channels)
    c = Conv2D(1, (3, 3), padding='same')(res_block_input)
    d = Add()([a, c])
    return Model(inputs=a, outputs=d)


def train_model(model, images, corruption_func, batch_size,
                steps_per_epoch, num_epochs, num_valid_samples):
    """
    train a given neural network model with given images
    :param model: a general neural network model for image restoration
    :param images: a list of file paths pointing to image files. You should assume these paths are complete, and
    should append anything to them
    :param corruption_func: A function receiving a numpy’s array representation of an image as a single argument,
    and returns a randomly corrupted version of the input image
    :param batch_size: the size of the batch of examples for each iteration of SGD
    :param steps_per_epoch: The number of update steps in each epoch
    :param num_epochs: The number of epochs for which the optimization will run
    :param num_valid_samples: The number of samples in the validation set to test on after every epoch.
    """
    random.shuffle(images)
    fraction = int(0.8 * len(images))
    train_set = images[0:fraction]
    test_set = images[fraction:]
    train_data_set_generator = load_dataset(train_set, batch_size, corruption_func,
                                            (model.input_shape[1], model.input_shape[2]))
    test_data_set_generator = load_dataset(test_set, batch_size, corruption_func,
                                           (model.input_shape[1], model.input_shape[2]))
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(train_data_set_generator, steps_per_epoch, num_epochs, validation_data=test_data_set_generator,
                        validation_steps=num_valid_samples / batch_size)


def restore_image(corrupted_image, base_model):
    """
    given a trained model, use it ro restore a corrupted image
    :param corrupted_image: – a grayscale image of shape (height, width) and with values in the [0, 1] range of
    type float64
    :param base_model: a neural network trained to restore small patches
    :return: fixed image
    """
    height, width = corrupted_image.shape
    a = Input(shape=(height, width, 1))
    b = base_model(a)
    new_model = Model(inputs=a, outputs=b)
    return (new_model.predict(np.expand_dims(corrupted_image.reshape((height, width, 1)) - 0.5, axis=0)) + 0.5).clip(
        min=0, max=1).reshape(
        (height, width)).astype(np.float64)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    randomly sample a value of sigma, uniformly distributed between min_sigma and
    max_sigma, followed by adding to every pixel of the input image a zero-mean gaussian random variable with standard
    deviation equal to sigma.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing the maximal
    variance of the gaussian distribution.
    :return: corrupted image
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    image_with_noise = image + np.random.normal(0, sigma, image.shape)
    return (np.around(image_with_noise * 255) / 255).clip(0, 1)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    train a denoising model
    :param num_res_blocks: The number of residual blocks in the net
    :param quick_mode: bool. set other params in order to train the model faster
    :return: trained denoising model
    """
    image_paths_list = sol5_utils.images_for_denoising()
    model = build_nn_model(height=24, width=24, num_channels=48, num_res_blocks=num_res_blocks)

    def inner(image):
        return add_gaussian_noise(image, min_sigma=0, max_sigma=0.2)

    if not quick_mode:
        train_model(model, image_paths_list, inner, batch_size=100, steps_per_epoch=100, num_epochs=5,
                    num_valid_samples=1000)
    else:
        train_model(model, image_paths_list, inner, batch_size=10, steps_per_epoch=3, num_epochs=2,
                    num_valid_samples=30)
    return model


def add_motion_blur(image, kernel_size, angle):
    """
    simulate motion blur on the given image using a square kernel of size kernel_size where the line has the given
    angle in radians, measured relative to the positive horizontal axis, e.g. a horizontal line would have a zero angle
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param kernel_size: an odd integer specifying the size of the kernel (even integers are ill-defined).
    :param angle: an angle in radians in the range [0, π).
    :return: corrupted image
    """
    corr_image = convolve(image, sol5_utils.motion_blur_kernel(kernel_size, angle))
    return (np.around(corr_image * 255) / 255).clip(0, 1)


def random_motion_blur(image, list_of_kernel_sizes):
    """
    samples an angle at uniform from the range [0, π), and choses a kernel size at uniform from the list
    list_of_kernel_sizes, followed by applying the previous function with the given image and the randomly sampled
    parameters
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param list_of_kernel_sizes: a list of odd integers
    :return: corrupted image
    """
    return add_motion_blur(image, np.random.choice(list_of_kernel_sizes), np.random.uniform(0, np.pi))


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    train a debluring model
    :param num_res_blocks: The number of residual blocks in the net
    :param quick_mode: bool. set other params in order to train the model faster
    :return: trained debluring model
    """
    image_paths_list = sol5_utils.images_for_deblurring()
    model = build_nn_model(height=16, width=16, num_channels=32, num_res_blocks=num_res_blocks)

    def inner(image):
        return random_motion_blur(image, [7])

    if not quick_mode:
        train_model(model, image_paths_list, inner, batch_size=100, steps_per_epoch=100, num_epochs=10,
                    num_valid_samples=1000)
    else:
        train_model(model, image_paths_list, inner, batch_size=10, steps_per_epoch=3, num_epochs=2,
                    num_valid_samples=30)
    return model
