import os
from PIL import Image
from torch.utils.data import Dataset


class ArtistClassificationDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform
        self.valid_classes = ["Jackson_Pollock", "Andy_Warhol", "Vincent_van_Gogh", "Pablo_Picasso", "Francisco_Goya",
                              "Marc_Chagall", "Salvador_Dali", "Leonardo_da_Vinci", "Mikhail_Vrubel", "Amedeo_Modigliani"]
        # Filter out unwanted classes
        self.classes = [cls for cls in os.listdir(root) if cls in self.valid_classes]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for filename in os.listdir(class_path):
                if filename.endswith(".jpg"):
                    image_path = os.path.join(class_path, filename)
                    class_idx = self.class_to_idx[class_name]
                    samples.append((image_path, class_idx))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, class_idx = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, class_idx
