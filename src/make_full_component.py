from np import save
import idx2numpy
import os

training_data_images = idx2numpy.convert_from_file("src/data/train-images.idx3-ubyte")

np.save(f"src/pca_weights/full_component.npy", training_data_images)
print(os.path.getsize("src/pca_weights/full_component.npy"))