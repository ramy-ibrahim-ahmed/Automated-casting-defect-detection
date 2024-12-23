from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

import torch
from .ResNet.ResNet import ResNet18
from .ResNet.DeepResNet import ResNet50

from keras.api.models import load_model
from .Inception.train import F1Score

from .helpers import predict_resnet, predict_inception

app = FastAPI(
    title="🛠️🔍 Automated Casting Defect Detection API",
    description=(
        "* An API designed to automate quality inspection of casting products using deep learning.\n"
        "* The system classifies images of submersible pump impellers as either **Defective** or **Ok** based on custom-trained models: ResNet-18, ResNet-50, and a fine-tuned Inception V3.\n"
        "* This solution improves manufacturing efficiency by reducing manual inspection errors and identifying defects like blowholes, shrinkage, and metallurgical irregularities."
    ),
    version="1.0.0",
    contact={
        "name": "Ramy Ibrahim",
        "url": "https://github.com/ramy-ibrahim-ahmed/Automated-Casting-Defect-Detection-with-ResNet",
        "email": "ramyibrahim987@gmail.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18 = ResNet18(out_neurons=1).to(device)
resnet18.load_state_dict(
    torch.load(
        r"src/best_resnet_18.pth",
        map_location=device,
        weights_only=True,
    )
)
resnet18.eval()

resnet50 = ResNet50(out_neurons=1).to(device)
resnet50.load_state_dict(
    torch.load(
        r"src/best_resnet_50.pth",
        map_location=device,
        weights_only=True,
    )
)
resnet50.eval()

inception = load_model(
    r"src/best_inception_v3.keras", custom_objects={"F1Score": F1Score}
)


@app.post("/resnet-18/")
async def predict_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image.")

        image = Image.open(file.file).convert("RGB")
        label, confidence = predict_resnet(model=resnet18, image=image, device=device)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"label": label, "confidence": f"{confidence:.2f}%"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resnet-50/")
async def predict_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image.")

        image = Image.open(file.file).convert("RGB")
        label, confidence = predict_resnet(
            model=resnet50, image=image, threshold=0.1, device=device
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"label": label, "confidence": f"{confidence:.2f}%"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inception-v3/")
async def predict_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image.")

        image = Image.open(file.file).convert("RGB")
        label, confidence = predict_inception(
            model=inception, image=image, threshold=0.1
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"label": label, "confidence": f"{confidence:.2f}%"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
