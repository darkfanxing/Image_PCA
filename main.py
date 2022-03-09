from os.path import getsize
from os import listdir

import re
from math import log10

from typing import Tuple, List

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import seaborn as sns
sns.set_theme(style="ticks")

from numpy import array, where, save, ndarray, mean
import matplotlib.pyplot as plt
import cv2

def get_data() -> Tuple[ndarray, ndarray]:
    mnist = load_digits()
    training_data = []
    test_data = []

    for number in range(10):
        number_indices = where(mnist.target==number)[0]
        training_data.append(mnist.images[number_indices[:16]])
        test_data.append(mnist.images[number_indices[16:26]])

    training_data = array(training_data).reshape((160, 64))
    test_data = array(test_data).reshape((100, 64))

    return training_data, test_data

def show_images(original_images, reconstructed_images, scale=24):
    for index in range(100):
        cv2.imshow("pca", cv2.resize(reconstructed_images, (0, 0), fx=scale, fy=scale))
        cv2.imshow("original", cv2.resize(test_data[index].reshape((8, 8, 1)), (0, 0), fx=scale, fy=scale))
        cv2.waitKey(0)

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def get_compression_ratio() -> List[float]:
    weights_filename = [file_ for file_ in listdir("pca_weights/") if file_.endswith('.npy')]
    weights_filename.sort(key=natural_keys)
    weights_size = []
    for weight_filename in weights_filename:
        weights_size.append(getsize(f"pca_weights/{weight_filename}")/1024)
    
    return weights_size[-1] / array(weights_size)

def convert_range(images):
    for image_index in range(images.shape[0]):
        images[image_index] = normalize(images[image_index])
    
    return images * 255

def psnr(images_true, images_test):
    psnr_count = 0
    for index in range(images_true.shape[0]):
        mse = mean((images_true[index] - images_test[index])**2)
        if mse < 1e-10:
            psnr_count += 100
        else:
            psnr_count += 10 * log10(255**2/mse)

    return psnr_count / images_true.shape[0]

if __name__ == "__main__":
    training_data, test_data = get_data()
    test_data_c = convert_range(test_data.reshape((-1, 8, 8)))
    psnrs = []
    for component_number in range(training_data.shape[1]):
        pca = PCA(n_components=component_number)
        pca.fit(training_data)
        # save(f"pca_weights/component_{component_number}.npy", pca.components_)\
        test_data_pca = pca.transform(test_data)
        reconstructed_images = pca.inverse_transform(test_data_pca)
        reconstructed_images = convert_range(reconstructed_images.reshape((-1, 8, 8)))
        
        psnrs.append(psnr(test_data_c, reconstructed_images))

    # plot relationship between compression ratio and number of components
    compression_ratios = get_compression_ratio()
    # compression_ratio_plot = sns.lineplot(x=list(range(64)), y=compression_ratios)
    # compression_ratio_plot.set_xlabel("number of components", fontsize=16)
    # compression_ratio_plot.set_ylabel("compression ratio", fontsize=16)
    # compression_ratio_plot.set_title("compression ratio vs. number of components", fontsize=18)
    
    psnr_plot = sns.lineplot(x=list(range(64)), y=psnrs)
    plt.show()

    
