import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess


IMG_SIZE = 224
BATCH_SIZE = 32


def crop_black_borders(image, threshold=10):
    """Crop black borders from a retinal fundus image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray_image > threshold

    if not np.any(mask):
        return image

    coordinates = np.column_stack(np.where(mask))
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)

    return image[y_min:y_max + 1, x_min:x_max + 1]


def enhance_contrast_clahe(image):
    """Apply CLAHE contrast enhancement"""
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    return enhanced_rgb


def load_and_preprocess_image(image_path, img_size=IMG_SIZE, use_crop=True, use_clahe=True):
    """Read, crop, enhance, resize, and convert image to float32."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if use_crop:
        image = crop_black_borders(image)

    if use_clahe:
        image = enhance_contrast_clahe(image)

    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)

    return image


class FundusDataGenerator(Sequence):
    def __init__(
        self,
        dataframe,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        shuffle=True,
        augment=False,
        model_type="baseline"
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.model_type = model_type
        self.indices = np.arange(len(self.dataframe))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, batch_index):
        batch_indices = self.indices[
            batch_index * self.batch_size:(batch_index + 1) * self.batch_size
        ]

        batch_df = self.dataframe.iloc[batch_indices]

        images = []
        labels = []

        for _, row in batch_df.iterrows():
            image = load_and_preprocess_image(
                row["image_path"],
                img_size=self.img_size
            )

            if self.augment:
                image = self.apply_augmentation(image)

            image = self.apply_model_preprocessing(image)

            images.append(image)
            labels.append(row["diagnosis"])

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def apply_augmentation(self, image):
        """Apply random training augmentation."""
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)

        if np.random.rand() < 0.5:
            angle = np.random.uniform(-20, 20)
            height, width = image.shape[:2]

            rotation_matrix = cv2.getRotationMatrix2D(
                (width / 2, height / 2),
                angle,
                1.0
            )

            image = cv2.warpAffine(
                image,
                rotation_matrix,
                (width, height),
                borderMode=cv2.BORDER_REFLECT_101
            )

        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.85, 1.15)
            beta = np.random.uniform(-15, 15)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        return image.astype(np.float32)

    def apply_model_preprocessing(self, image):
        """Apply normalization/preprocessing depending on the model."""
        if self.model_type == "resnet":
            return resnet_preprocess(image.copy())

        if self.model_type == "efficientnet":
            return efficientnet_preprocess(image.copy())

        return image / 255.0
