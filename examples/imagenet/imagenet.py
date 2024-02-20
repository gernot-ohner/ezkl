import json
import numpy as np
from PIL import Image


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load an image and resize it to the target size."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    return np.array(img) / 255.0  # Normalize pixel values to [0, 1]


# Path to your Imagenette images

image1_path = "/Users/gernotohner/dev/rust/ezkl/examples/imagenet/imagenette2/train/n01440764/ILSVRC2012_val_00000293.JPEG"
imagenette_image_paths = [image1_path]

# Load and preprocess images
sample_inputs = [load_and_preprocess_image(path) for path in imagenette_image_paths]

# Convert to list for JSON serialization
sample_inputs_list = [input_image.tolist() for input_image in sample_inputs]

# Serialize the list of image data to JSON
with open("input.json", "w") as f:
    json.dump(sample_inputs_list, f)

# Check if the file was created and its content
with open("input.json", "r") as f:
    loaded_input = json.load(f)
    print(loaded_input)  # Optional: Print to verify
