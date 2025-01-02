# extract_from_image.py

import os
import pytesseract
from PIL import Image
import pandas as pd

# Configure Tesseract path if not in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path as necessary

# Folder where images are stored
input_folder = 'files'
output_file = 'extracted_text_from_images.xlsx'

# List to store extracted data
data = []

# Get all image files in the folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    print(f"Processing {image_file}...")

    # Open the image
    with Image.open(image_path) as img:
        # Use OCR to extract text
        text = pytesseract.image_to_string(img, lang='eng')
        data.append({
            'Image File': image_file,
            'Extracted Text': text.strip()
        })

# Save the data to an Excel file
df = pd.DataFrame(data)
df.to_excel(output_file, index=False)
print(f"Text extraction complete. Data saved to {output_file}.")
