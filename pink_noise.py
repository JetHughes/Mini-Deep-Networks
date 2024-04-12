import numpy as np
import matplotlib.pyplot as plt

def pink_noise_2d(shape):
    """
    Generate an image of pink noise of a specified size.

    Parameters:
        width (int): Width of the image.
        height (int): Height of the image.
        num_octaves (int): Number of octaves to use for pink noise generation.

    Returns:
        numpy.ndarray: An array representing the generated pink noise image.
    """

    white_noise = np.random.randn(*shape)

    fft_noise = np.fft.fftn(white_noise)

    x_freq = np.fft.fftfreq(shape[0], d=1./shape[0])
    y_freq = np.fft.fftfreq(shape[1], d=1./shape[1])

    freq_mag = np.sqrt(x_freq[:,None]**2 + y_freq[:,None]**2)
    freq_mag[0,0] = 1
    pink_filter = 1/freq_mag
    pink_fft = fft_noise * pink_filter
    pink_noise = np.fft.ifftn(pink_fft).real
    pink_noise = (pink_noise-np.min(pink_noise))/(np.max(pink_noise)-np.min(pink_noise))

    return pink_noise

def pink_noise(shape):
    pink_noise=np.zeros(shape)
    for i in range(shape[2]):
        pink_noise[:,:,i] = pink_noise_2d(shape[:2])
    return pink_noise

def gen_pink_noise(shape, noise_level=0.2):
    initial_image = np.ones(shape) * noise_level
    noise = pink_noise(shape)
    noisy_image = initial_image + noise
    noisy_image = np.clip(noisy_image, 0., 1.)
    return noisy_image
        


def display_image(image):
    """
    Display the given image.

    Parameters:
        image (numpy.ndarray): Image to display.
    """
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Example usage:
pink_noise_img = gen_pink_noise((96,96,3))
display_image(pink_noise_img)
