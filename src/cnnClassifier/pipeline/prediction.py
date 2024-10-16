import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # load model
        model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)

        # Normalize the image
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        # Get predictions
        result = model.predict(test_image)
        print("Raw prediction output:", result)

        result = np.argmax(result, axis=1)
        print("Predicted class index:", result[0])

        if result[0] == 0:
            prediction = 'Cyst'
        elif result[0] == 1:
            prediction = 'Normal'
        elif result[0] == 2:
            prediction = 'Stone'
        else:
            prediction = 'Tumor'

        return [{"image": prediction}]
