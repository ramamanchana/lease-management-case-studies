import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Sample image (Assuming we have an image of a lease document)
image_path = 'sample_lease_document.png'  # Path to sample image

# Load and preprocess the image
image = load_img(image_path, target_size=(128, 128))
image_array = img_to_array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Predict (Using a pre-trained model for demonstration)
# model.fit(training_data, training_labels, epochs=10)  # Example training step
prediction = model.predict(image_array)
print("Predicted Class:", np.argmax(prediction))

# Visualize the input image
plt.imshow(load_img(image_path))
plt.title('Sample Lease Document')
plt.show()
