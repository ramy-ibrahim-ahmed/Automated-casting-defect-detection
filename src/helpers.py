import numpy as np
import torch
from torchvision import transforms


def predict_resnet(model, image, device="cpu", img_shape=(224, 224), threshold=0.1):
    transform = transforms.Compose(
        [
            transforms.Resize(img_shape),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5642121434211731, 0.5642121434211731, 0.5642121434211731],
                std=[0.23441407084465027, 0.23441407084465027, 0.23441407084465027],
            ),
        ]
    )

    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs)
        predicted = (probabilities > threshold).int()

    if predicted[0][0] == 1:
        return "Defective", round(float(probabilities[0][0]), 2) * 100

    return "OK", round(1 - float(probabilities[0][0]), 2) * 100


def predict_inception(model, image, img_shape=(299, 299), threshold=0.1):
    image = image.resize(img_shape)
    image_array = np.expand_dims(np.array(image), axis=0)

    probability = model.predict(image_array, verbose=0)
    probability = float(probability[0][0])

    predicted = int(probability > threshold)
    if predicted == 1:
        return "Defective", round(probability, 2) * 100
    return "OK", round(1 - probability, 2) * 100
