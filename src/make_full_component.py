from numpy import save, float16
import idx2numpy
import os

training_data_images = idx2numpy.convert_from_file("src/data/train-images.idx3-ubyte").astype(float16)
save(f"src/pca_weights/full_component.npy", training_data_images[:160])
print(os.path.getsize("src/pca_weights/full_component.npy"))