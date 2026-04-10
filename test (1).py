from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = load_model("/Users/vaibhavmishra/Desktop/face mask detection/mask_model.keras")

datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

test_data = datagen.flow_from_directory(
    "/Users/vaibhavmishra/Desktop/face mask detection/Dataset",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    subset='validation'  # <--- validation split ka subset yahan use karo
)

loss, acc = model.evaluate(test_data)
print("Accuracy:", acc)
print("Loss:", loss)