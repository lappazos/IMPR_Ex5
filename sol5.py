import random
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from tensorflow.keras.layers import Input, Conv2D, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import sol5_utils
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt

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
    c = Conv2D(1, (3, 3), padding='same')(res_block_input)
    d = Add()([a, c])
    return Model(inputs=a, outputs=d)


def train_model(model, images, corruption_func, batch_size,
                steps_per_epoch, num_epochs, num_valid_samples):
    random.shuffle(images)
    fraction = int(0.8 * len(images))
    train_set = images[0:fraction]
    test_set = images[fraction:]
    train_data_set_generator = load_dataset(train_set, batch_size, corruption_func,
                                            (model.input_shape[1], model.input_shape[2]))
    test_data_set_generator = load_dataset(test_set, batch_size, corruption_func,
                                           (model.input_shape[1], model.input_shape[2]))
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    # todo check devision in batch size
    model.fit_generator(train_data_set_generator, steps_per_epoch, num_epochs, validation_data=test_data_set_generator,
                        validation_steps=num_valid_samples / batch_size)


def restore_image(corrupted_image, base_model):
    height, width = corrupted_image.shape
    a = Input(shape=(height, width, 1))
    b = base_model(a)
    new_model = Model(inputs=a, outputs=b)
    return (new_model.predict(np.expand_dims(corrupted_image.reshape((height, width, 1)) - 0.5, axis=0)) + 0.5).clip(
        min=0, max=1).reshape(
        (height, width))


def add_gaussian_noise(image, min_sigma, max_sigma):
    sigma = np.random.uniform(min_sigma, max_sigma)
    image_with_noise = image + np.random.normal(0, sigma, image.shape)
    return (np.around(image_with_noise * 255) / 255).clip(0, 1)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
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
    corr_image = convolve(image, sol5_utils.motion_blur_kernel(kernel_size, angle))
    return (np.around(corr_image * 255) / 255).clip(0, 1)


def random_motion_blur(image, list_of_kernel_sizes):
    return add_motion_blur(image, np.random.choice(list_of_kernel_sizes), np.random.uniform(0, np.pi))


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
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


def make_graphs():
    validation_error_denoise = []
    validation_error_deblur = []
    for i in range(1, 6):
        denoise_model = learn_denoising_model(i)
        deblur_model = learn_deblurring_model(i)
        validation_error_denoise.append(denoise_model.history.history['val_loss'][-1])
        validation_error_deblur.append(deblur_model.history.history['val_loss'][-1])

    arr = np.arange(1, 6)

    plt.plot(arr, validation_error_denoise)
    plt.title('validation error - denoise')
    plt.xlabel('number res blocks')
    plt.ylabel('validation loss denoise')
    plt.savefig('denoise.png')
    plt.show()

    plt.plot(arr, validation_error_deblur)
    plt.title('validation error - deblur')
    plt.xlabel('number res blocks')
    plt.ylabel('validation loss deblur')
    plt.savefig('deblur.png')
    plt.show()


if __name__ == '__main__':
    # make_graphs()
    model = learn_denoising_model()
    im = read_image('current\\image_dataset\\train\\15004.jpg',GREYSCALE)
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    plt.show()
    im_corr = add_gaussian_noise(im,0,0.2)
    plt.imshow(im_corr, cmap='gray', vmin=0, vmax=1)
    plt.show()
    res = restore_image(im_corr,model)
    plt.imshow(res, cmap='gray', vmin=0, vmax=1)
    plt.show()

    model = learn_deblurring_model()
    im = read_image('current\\text_dataset\\train\\0000196_orig.png', GREYSCALE)
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    plt.show()
    im_corr = random_motion_blur(im,[7])
    plt.imshow(im_corr, cmap='gray', vmin=0, vmax=1)
    plt.show()
    res = restore_image(im_corr, model)
    plt.imshow(res, cmap='gray', vmin=0, vmax=1)
    plt.show()
