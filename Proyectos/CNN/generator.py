import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up the image size and directories
IMAGE_SIZE = 128
train_dir = 'orignal'  # Your source directory with images
save_directory = 'processed_images'  # Directory to save augmented images

# Create the save directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Define the ImageDataGenerator with augmentations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True
)

# Set up the image data generator to flow from the directory and save augmented images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode="sparse",
    save_to_dir=save_directory,  # Directory to save augmented images
    save_prefix='aug',  # Prefix for the augmented images
    save_format='jpeg',  # Save as jpeg
    batch_size=32,  # Batch size for the generator
)

# Check if the generator is working
for i in range(2):  # Change to the number of steps you want to iterate
    x_batch, y_batch = next(train_generator)  # Get the next batch of images
    print(f"Batch {i + 1} of {train_generator.samples // train_generator.batch_size} - Saved images to {save_directory}")
