import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class AnnotatedImageDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None):
        """
        Args:
            root_dir (string): Directory with all the subfolders containing images.
            label_file (string): Path to the Excel file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.labels_df = pd.read_excel(label_file)
        self.labels_df = self.labels_df[self.labels_df['Presence'] != 0]
        self.labels_df.reset_index(drop=True, inplace=True)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        pat_id = row['Pat_ID']
        section_id = row['Section_ID']
        window_id = str(row['Window_ID']).zfill(5)
        label = row['Presence']
        
        folder_name = f"{pat_id}_{section_id}"
        folder_path = os.path.join(self.root_dir, folder_name)
        
        image_file, is_augmented = self.find_image_file(folder_path, window_id)
        if image_file is None:
            raise FileNotFoundError(f"No matching image found for Window ID {window_id} in folder {folder_name}")

        # If the image is augmented, keep the label as -1 but retrieve the original pat_id
        if is_augmented:
            label = -1
            base_window_id = image_file.split('_')[0]
            pat_id = self.get_original_pat_id(base_window_id, section_id)
        
        image = Image.open(image_file).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label, pat_id

    def find_image_file(self, folder_path, window_id):
        """
        Searches for an image file in the folder that matches the given window_id,
        including augmented versions.
        """
        for file_name in os.listdir(folder_path):
            if file_name.startswith(window_id) and file_name.endswith('.png'):
                is_augmented = "_Aug" in file_name
                return os.path.join(folder_path, file_name), is_augmented
        return None, False

    def get_original_pat_id(self, base_window_id, section_id):
        """
        Finds the original patient ID for an augmented image by matching the base
        window_id and section_id in the labels DataFrame.
        """
        match = self.labels_df[
            (self.labels_df['Window_ID'] == int(base_window_id)) &
            (self.labels_df['Section_ID'] == section_id)
        ]
        if not match.empty:
            return match.iloc[0]['Pat_ID']
        return None  # Or handle case where match is not found


def show_image(image_tensor, label):
    image = image_tensor.permute(1, 2, 0).numpy()
    plt.imshow(image)
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = AnnotatedImageDataset(
        root_dir='Annotated',
        label_file='HP_WSI-CoordAnnotatedPatches.xlsx',
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    count = 0
    for images, labels, pat_ids in dataloader:
        for i in range(len(images)):
            if labels[i].item() == -1:
                if count >= 15:
                    break
                print(f"Patient ID: {pat_ids[i]}, Label: {labels[i].item()}")
                show_image(images[i], labels[i].item())
                count += 1
        if count >= 15:
            break
