from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
# AutoVision AI project folder




def preprocessing_dataset():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RAW_PATH = PROJECT_ROOT / "data" / "raw" / "flower_photos"
    PROCESSED_PATH = PROJECT_ROOT / "data" / "preprocessed" / "dataset"
    if not RAW_PATH.exists():
        return "ERROR: Raw dataset folder not found!"

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    for class_folder in RAW_PATH.iterdir():

        if not class_folder.is_dir():
            continue

        processed_class = PROCESSED_PATH / class_folder.name
        processed_class.mkdir(parents=True, exist_ok=True)

        for image_path in class_folder.iterdir():

            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            try:
                image = Image.open(image_path)
                image = image.convert("RGB")
                image = image.resize((180, 180))
                image.save(processed_class / image_path.name)

            except Exception as e:
                return (f"Error processing {image_path}: {e}")
    return "Data Preprocessing completed"
if __name__ == "__main__":
    preprocessing_dataset()