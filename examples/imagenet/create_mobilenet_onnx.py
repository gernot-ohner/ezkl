
from keras.optimizers.legacy import Adam
from keras.applications import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input

train_dir = "./imagenette2/train"
val_dir = "./imagenette2/val"

# For training, you'll need to setup data generators that perform data augmentation and preprocessing
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load the MobileNet model without the top layer
model = MobileNet(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')


eval_generator = eval_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Ensure this matches your dataset
    shuffle=False)  # Important for evaluation to not shuffle the data

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# eval_loss, eval_accuracy = model.evaluate_generator(eval_generator)
# print("Evaluation loss:", eval_loss)
# print("Evaluation accuracy:", eval_accuracy)
model.save("./model")