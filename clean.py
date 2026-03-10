from PIL import Image
import os

SOURCE = "img_Dataset"
DEST = "img_Dataset_clean"

count = 0

for folder_path, _, files in os.walk(SOURCE):

    for filename in files:

        src_path = os.path.join(folder_path, filename)

        # replicate folder structure
        relative_path = os.path.relpath(folder_path, SOURCE)
        dest_folder = os.path.join(DEST, relative_path)
        os.makedirs(dest_folder, exist_ok=True)

        dest_path = os.path.join(dest_folder, filename)

        try:
            with Image.open(src_path) as img:

                img = img.convert("RGB")
                img.save(dest_path, "JPEG", quality=95)

                count += 1

        except Exception as e:
            print("Skipping bad image:", src_path, "|", e)

print("Total cleaned images:", count)