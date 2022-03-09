from sklearn.datasets import load_digits
from numpy import array, where
from sklearn.decomposition import PCA
import cv2

mnist = load_digits()
training_data = []
test_data = []

for number in range(10):
    number_indices = where(mnist.target==number)[0]
    training_data.append(mnist.images[number_indices[:16]])
    test_data.append(mnist.images[number_indices[16:26]])

training_data = array(training_data).reshape((160, 64))
test_data = array(test_data).reshape((100, 64))



pca = PCA(n_components=49)
pca.fit(training_data)
reconstructed_images = pca.transform(test_data)
reconstructed_images = reconstructed_images.reshape((100, 7, 7, 1))
cv2.imshow("123", reconstructed_images[0])
cv2.waitKey(0)
# weights = pca.components_
# print(weights)