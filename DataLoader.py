import pandas as pd
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MedicalImageDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, presence, pat_id = self.data[idx]
        
        # Convert the image to a tensor and normalize (assuming image is in BGR format)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
        else:
            image = torch.zeros((3, 256, 256), dtype=torch.float32)  # Placeholder for missing images
        
        # Convert presence to a tensor
        label = torch.tensor(presence, dtype=torch.float32)
        
        return image, label, pat_id


def Load_Data(excel_path, base_folder):
    data = []
    df = pd.read_excel(excel_path)
    
    # Filter out rows with Presence == 0
    df = df[df['Presence'] != 0].reset_index(drop=True)
    
    # Loop through each subfolder and file in the Annotated folder
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            # Only process .png files
            if file.endswith(".png"):
                # Extract folder name and file name
                subfolder = os.path.basename(root)
                pat_id, section_id = subfolder.split("_")
                window_id = file.split(".")[0]

                # Check if window_id length meets criteria for processing
                if len(window_id) > 5:
                    image_path = os.path.join(root, file)
                    image = cv2.imread(image_path)
                    presence = 1  # Since presence is 1 when window_id length > 5

                    # Create a single list for this file
                    image_data = [image, presence, pat_id]
                    data.append(image_data)
                
                else:
                    window_id = int(window_id)
                    # Find the matching row in DataFrame
                    matching_row = df[(df['Pat_ID'] == pat_id) & (df['Section_ID'] == int(section_id)) & (df['Window_ID'] == window_id)]

                    if not matching_row.empty:
                        presence = matching_row.iloc[0]['Presence']
                        image_path = os.path.join(root, file)
                        image = cv2.imread(image_path)

                        # Create a single list for this file
                        image_data = [image, presence, pat_id]
                        data.append(image_data)

                    else: 
                        image_path = os.path.join(root, file)
                        image = cv2.imread(image_path)
                        presence = -1

                        # Create a single list for this file
                        image_data = [image, presence, pat_id]
                        data.append(image_data)
    
    return data