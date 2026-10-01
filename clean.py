from PIL import Image
import os

folders = ["dataset/train/cats", "dataset/train/dogs", "dataset/test/cats", "dataset/test/dogs"]
count = 0
deleted = 0

for folder in folders:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
                # re-encode as clean JPEG to fix header
                img.save(path, "JPEG", quality=95)
                count += 1
        except Exception as e:
            try:
                os.remove(path)
                deleted += 1
            except:
                pass

print(f"Re-encoded {count} images")
print(f"Deleted {deleted}")