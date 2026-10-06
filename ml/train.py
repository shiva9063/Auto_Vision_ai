import tensorflow as tf
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from tensorflow.keras import layers, models


def model_training():
    # ==========================================
    # 1. Project paths
    # ==========================================

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_PATH = PROJECT_ROOT / "data" / "preprocessed" / "dataset"


    # ==========================================
    # 2. Parameters
    # ==========================================

    IMAGE_SIZE = (180, 180)
    BATCH_SIZE = 32
    SEED = 42


    # ==========================================
    # 3. Load training dataset
    # ==========================================

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_PATH,
        validation_split=0.25,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    # ==========================================
    # 4. Load validation dataset
    # ==========================================

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_PATH,
        validation_split=0.25,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # ==========================================
    # 5. Class names
    # ==========================================
    # Class names
    import json

    CLASS_PATH=PROJECT_ROOT/'models'/'class_names.json'


    class_names = train_dataset.class_names

    with open(CLASS_PATH, "w") as f:
        json.dump(class_names, f)




    # ==========================================
    # 6. Normalize images
    # ==========================================

    normalization_layer = layers.Rescaling(1.0 / 255)

    train_dataset = train_dataset.map(
        lambda images, labels: (normalization_layer(images), labels)
    )

    validation_dataset = validation_dataset.map(
        lambda images, labels: (normalization_layer(images), labels)
    )


    # ==========================================
    # 7. Improve performance
    # ==========================================

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(AUTOTUNE)


    # ==========================================
    # 8. Build CNN model
    # ==========================================

    model = models.Sequential([

        layers.Input(shape=(180, 180, 3)),

        # CNN Block 1
        layers.Conv2D(16,(3, 3),padding="same",activation="relu"),
        layers.MaxPool2D(),

        # CNN Block 2
        layers.Conv2D(32,(3, 3),padding="same",activation="relu"),
        layers.MaxPool2D(),

        # CNN Block 3
        layers.Conv2D(64,(3, 3),padding="same",activation="relu"),
        layers.MaxPool2D(),

        # Classification
        layers.Flatten(),

        layers.Dense(64,activation="relu"),

        layers.Dense(len(class_names),activation="softmax")
    ])


    # ==========================================
    # 9. Compile model
    # ==========================================

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


    # ==========================================
    # 10. Model summary
    # ==========================================

    model.summary()

    model.fit(train_dataset,validation_data=validation_dataset,epochs=15)

    MODEL_PATH = PROJECT_ROOT / "models" / "model.keras"

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model.save(MODEL_PATH)

if __name__=='__main__':
    model_training()