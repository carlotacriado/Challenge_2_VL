import csv
import cv2
import os
import numpy as np

IMAGE_PATH = r"C:\Users\oleks\Downloads\Annotated"

list_name = []
list_presence = []
images_path = []

dataset = []
label = []

SIZE = 64

# Open the CSV file and read the values: list_name = Name of patient and name of the photo | list_presence = 1, -1, 0 depending of the presence
CSV_PATH = r'C:\Users\oleks\Downloads\HP_WSI-CoordAnnotatedPatches.csv'
with open(CSV_PATH) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        elif line_count>0:
            list_name.append(f"{row[0]}/{row[2].zfill(5)}")
            list_presence.append(row[7])
            line_count += 1
    print(f'Processed {line_count} lines.')

# Read all the image, check if they are annotated and match them with the presence from the CSV file
number_error = 0
for folder in os.listdir(IMAGE_PATH):
    if "_0" in folder:
        for filename in os.listdir(os.path.join(IMAGE_PATH, folder)):
            if filename.endswith(".png"):
                image = cv2.imread(os.path.join(os.path.join(IMAGE_PATH, folder), filename))
                image = image.resize((SIZE, SIZE))
                dataset.append(np.array(image))
                filename = filename[:5]
                ref_label = folder + '/' + filename
                ref_label = ref_label.replace('_0', '')

                try:
                    index = list_name.index(ref_label)
                except ValueError:
                    #print(f"'{ref_label}' is not in the list.")
                    number_error +=1

                label.append(list_presence[index])


print(f"Number of image not in the list: {number_error}")
print(f"Length of our dataset: {len(dataset)}")
print(f"Length of our label: {len(label)}")
