from os.path import getsize
from os import listdir
from typing import List
from sklearn.decomposition import PCA
from numpy import array, vstack, where, save, mean, log10, float16
from utils import natural_keys
import matplotlib.pyplot as plt
import cv2
import idx2numpy

import seaborn as sns
sns.set_theme(style="ticks")

class MNISTPCAAnalysis():
    def __init__(self, training_data_number: int = 160, test_data_number: int = 100) -> None:
        self.traing_data_number = training_data_number
        self.test_data_number = test_data_number
        self.initialize_data()


    def initialize_data(self) -> None:
        self.training_data = []
        self.test_data = []

        training_data_labels = idx2numpy.convert_from_file("src/data/train-labels.idx1-ubyte")
        training_data_images = idx2numpy.convert_from_file("src/data/train-images.idx3-ubyte")
        test_data_labels = idx2numpy.convert_from_file("src/data/t10k-labels.idx1-ubyte")
        test_data_images = idx2numpy.convert_from_file("src/data/t10k-images.idx3-ubyte")

        for number in range(10):
            training_data_number_indices = where(training_data_labels==number)[0]
            test_data_number_indices = where(test_data_labels==number)[0]
            
            self.training_data.append(training_data_images[training_data_number_indices[:16]])
            self.test_data.append(test_data_images[test_data_number_indices[:10]])

        self.training_data = array(self.training_data).reshape((-1, 28*28))
        self.test_data = array(self.test_data).reshape((-1, 28*28))


    def fit(self) -> None:
        self.psnrs = []
        for component_number in range(self.training_data.shape[1]):
            try:
                pca = PCA(n_components=component_number, svd_solver="full")
                pca.fit(self.training_data)
                save(
                    f"src/pca_weights/component_{component_number}.npy", vstack((
                        pca.components_.reshape(784, -1),
                        pca.transform(self.training_data)
                    )).astype(float16)
                )
                reconstructed_images = pca.inverse_transform(pca.transform(self.test_data))
                self.psnrs.append(self._get_psnr(self.test_data, reconstructed_images))
            except:
                pca = PCA(n_components=component_number-1, svd_solver="full")
                pca.fit(self.training_data)
                
                self.eigenvectors = pca.components_
                self.eigenvalues = pca.explained_variance_
                break


    def plot_compression_ratios(self) -> None:
        def _get_compression_ratios() -> List[float]:
            weights_filename = [file_ for file_ in listdir("src/pca_weights/") if file_.endswith('.npy')]
            weights_filename.sort(key=natural_keys)
            weights_size = []
            for weight_filename in weights_filename:
                weights_size.append(getsize(f"src/pca_weights/{weight_filename}"))
            print(weights_size[:-1])
            return getsize("src/pca_weights/full_component.npy") / (array(weights_size[:-1]))

        compression_ratios = _get_compression_ratios()
        sns.lineplot(
            x=list(range(len(compression_ratios))),
            y=compression_ratios
        )
        plt.xlabel("number of components", fontsize=16)
        plt.ylabel("Compression ratio", fontsize=16)
        plt.show()

    
    def plot_psnr(self) -> None:
        sns.lineplot(
            x=list(range(len(self.psnrs))),
            y=array(log10(self.psnrs)),
        )
        plt.xlabel("number of components", fontsize=16)
        plt.ylabel("PSNR", fontsize=16)
        plt.show()

    
    def plot_eigenvalue_superpositions(self) -> None:
        def _get_eigenvalue_superpositions():
            superposition_values = []
            for index in range(len(self.eigenvalues)):
                superposition_values.append(sum(self.eigenvalues[:index+1]))

            return array(superposition_values) / superposition_values[-1]
        
        eigenvalue_superpositions = _get_eigenvalue_superpositions()

        sns.lineplot(
            x=list(range(len(eigenvalue_superpositions))),
            y=eigenvalue_superpositions
        )
        plt.xlabel("number of components", fontsize=16)
        plt.ylabel("eigenvalue superposition", fontsize=16)
        plt.show()


    def plot_eigenvectors(self, n_components=5, scale=10) -> None:
        for index in range(n_components):
            cv2.imshow(
                "Eigenvector",
                cv2.resize(
                    (self.eigenvectors[index]*self.eigenvalues[index]).reshape(28, 28, 1), (0, 0), fx=scale, fy=scale)
                )
            cv2.waitKey(0)


    def show_reconstructed_images(self, n_components=5, display_images_number=5, scale=10) -> None:
        if display_images_number > 0 and display_images_number <= self.test_data.shape[0]:
            # is_escape = False
            for component_number in range(n_components):
                pca = PCA(n_components=component_number, svd_solver="full")
                pca.fit(self.training_data)
                reconstructed_images = pca.inverse_transform(pca.transform(self.test_data))

                for index in range(display_images_number):
                    cv2.imshow(
                        "Original image",
                        cv2.resize(
                            self.test_data[index].reshape(28, 28, 1),
                            (0, 0),
                            fx=scale,
                            fy=scale
                        )
                    )
                    cv2.imshow(
                        "Images with PCA reconstuction",
                        cv2.resize(reconstructed_images[index].reshape(28, 28, 1), (0, 0), fx=scale, fy=scale)
                    )
                    cv2.waitKey(0)

    def _get_psnr(self, images_true, images_test) -> List[float]:
            psnr_count = 0
            for index in range(images_true.shape[0]):
                mse = mean((images_true[index] - images_test[index])**2)
                psnr_count += 100 if mse < 1e-10 else (10 * log10(255**2/mse))

            return psnr_count / images_true.shape[0]

if __name__ == "__main__":
    mnist_pca_analysis = MNISTPCAAnalysis()
    mnist_pca_analysis.fit()

    mnist_pca_analysis.plot_compression_ratios()
    # mnist_pca_analysis.plot_eigenvalue_superpositions()
    # mnist_pca_analysis.plot_psnr()
    # mnist_pca_analysis.show_reconstructed_images(n_components=5, display_images_number=5, scale=10)
    # mnist_pca_analysis.plot_eigenvectors(n_components=5, scale=10)