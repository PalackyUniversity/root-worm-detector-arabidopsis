from sahi.slicing import slice_image
from tqdm import tqdm
from glob import glob
from PIL import Image
import numpy as np
import cv2
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm


TRAIN_METRICS = {
    "accuracy": "accuracy",
    "dice_loss": sm.losses.dice_loss,
    "precision": sm.metrics.precision,
    "recall": sm.metrics.recall,
    "f1-score": sm.metrics.f1_score,
    "f2-score": sm.metrics.f2_score
}

THRESHOLD_FOR_COUNTING = 0.1

# Parameters for slicing
IMAGE_SIZE = 512
IMAGE_OVERLAP = 0.2

DIR_MODELS = "models"
DIR_RESULT = "results"
DIR_DATA = "data"

BATCH_SIZE = 8

os.makedirs(DIR_RESULT, exist_ok=True)

# Create model
model = sm.Unet("seresnet18", classes=2, encoder_weights=None, input_shape=(512, 512, 3))
model.compile("Adam", sm.losses.DiceLoss(), metrics=list(TRAIN_METRICS.values()))

# Load model if already trained
model.load_weights(os.path.join(DIR_MODELS, "best_model.h5"))

for image in tqdm(glob(os.path.join(DIR_DATA, "*"))):
    original_image = cv2.imread(image)
    sliced = slice_image(
        image=Image.fromarray(original_image),
        slice_height=IMAGE_SIZE,
        slice_width=IMAGE_SIZE,
        overlap_height_ratio=IMAGE_OVERLAP,
        overlap_width_ratio=IMAGE_OVERLAP
    )

    x = np.array([i["image"] for i in sliced], dtype=np.float32)
    y = model.predict(x, batch_size=BATCH_SIZE)

    image_pred = np.zeros(original_image.shape[:2], dtype=y.dtype)
    image_count = np.zeros(original_image.shape[:2], dtype=y.dtype)

    # Reconstruct final prediction image
    for n, i in enumerate(sliced):
        px, py = i["starting_pixel"]

        image_pred[py:py + IMAGE_SIZE, px:px + IMAGE_SIZE] += y[n, :, :, 1]
        image_count[py:py + IMAGE_SIZE, px:px + IMAGE_SIZE] += 1

        cv2.rectangle(original_image, (px, py), (px + IMAGE_SIZE, py + IMAGE_SIZE), (0, 255, 0), 1)

    image_count[image_count == 0] = 1
    image_pred /= image_count

    # Detect contours in the image
    contours_pred, _ = cv2.findContours(
        (image_pred >= THRESHOLD_FOR_COUNTING).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(original_image, contours_pred, -1, (0, 0, 255), 2)
    cv2.imwrite(image.replace(DIR_DATA, DIR_RESULT) + f"_count={len(contours_pred)}.jpg", original_image)
