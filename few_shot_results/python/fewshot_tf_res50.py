## few-shot results, training required, res50 backbone.
# PLOT: PROMPT LEARNING WITH OPTIMAL TRANS- PORT FOR VISION-LANGUAGE MODELS


import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import os

def get_datasets_mean(all_list):
    # ipdb.set_trace()
    to_tensor = torch.Tensor(all_list)
    mean_tensor = to_tensor.mean(0)
    return mean_tensor.tolist()


datasets = ['ImageNet', 'Flowers102', 'DTD', 'OxfordPets', 'StanfordCars', 'UCF101', 'Caltech101', 'Food101', 'SUN397', 'FGVCAircraft', 'EuroSAT', 'Mean']
methods = ['Tip-X', 'Tip-Adapter', 'APE', 'DMN', 'OURS']
excel_path = '../excel/training_free_res50.xlsx'
results_path = './figs/'

# Load the Excel file into a DataFrame
df = pd.read_excel(excel_path, header=None)

# Function to find the index of the next empty row after a given start index
def find_next_empty_row(start_index):
    empty_rows = df[df.isnull().all(axis=1)].index
    for idx in empty_rows:
        if idx > start_index:
            return idx
    return None

all_cate = {}
for method in methods:
    print(method)
    method_start = df[df[0] == method].index[0]
    method_end = find_next_empty_row(method_start)

    # Extract results data
    method_results = df.iloc[method_start+1: method_end, 1:13].values.tolist()
    method_results = np.array(method_results).T.tolist()  ## [11 tasks+1mean * 5 shot]
    method_results[-1] = get_datasets_mean(method_results[:-1])
    all_cate[method] = method_results

shots = [0, 1, 2, 4, 8, 16]
shots_calip_zero_shot = [0]  


zero_shot_clip_results = [60.32, 66.10, 40.07, 85.83, 55.71, 61.33, 83.94, 77.32, 58.53, 17.10, 37.54, 58.53]
# zero_shot_clip_mean = []
for i in range(len(datasets)):
    dataset_name = datasets[i]
    dmn = all_cate['DMN'][i]
    ape = all_cate['APE'][i]
    tip = all_cate['Tip-Adapter'][i]
    tip_x = all_cate['Tip-X'][i]
    LP = all_cate['OURS'][i]
    zero_shot_clip = [zero_shot_clip_results[i]]

    # Plotting
    plt.figure(figsize=(4,3))
    plt.plot(shots[1:], LP, '*--', label='ECALP (Ours)', color='red')
    plt.plot(shots[1:], dmn, 'o--', label='DMN', color='green')
    plt.plot(shots[1:], ape, 'v--', label='APE', color='cyan')
    plt.plot(shots[1:], tip, 'p--', label='TIP-Adapter', color='magenta')
    plt.plot(shots[1:], tip_x, 's--', label='TIP-X', color='black')

    # Plotting only points for Zero-shot CLIP
    plt.scatter(shots_calip_zero_shot, zero_shot_clip, marker='D', label='Zero-shot CLIP', color='brown')
    
    # Set labels, title, legend, etc.
    plt.xlabel('Shots Number')
    plt.ylabel('Accuracy (%)')
    if dataset_name == 'Mean':
        plt.title('Average Accuracy of 11 Datasets')
    else:
        plt.title('Accuracy on ' + dataset_name)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks([0, 1, 2, 4, 8, 16])
    plt.tight_layout()
    plt.savefig(results_path+'resnet_tf_' + dataset_name + '.pdf', dpi=500, bbox_inches='tight', transparent=True)





# List of PDF files in the desired order
pdf_files = [
    
    "resnet_tf_Mean.pdf",
    "resnet_tf_ImageNet.pdf",
    "resnet_tf_Flowers102.pdf",
    "resnet_tf_DTD.pdf",
    # "resnet_tf_OxfordPets.pdf",
    "resnet_tf_StanfordCars.pdf",
    "resnet_tf_UCF101.pdf",
    # "resnet_tf_Caltech101.pdf",
    # "resnet_tf_Food101.pdf",
    "resnet_tf_SUN397.pdf",
    "resnet_tf_FGVCAircraft.pdf",
    "resnet_tf_EuroSAT.pdf"
]

# Convert the first page of each PDF to an image
images = []
for pdf in pdf_files:
    doc = fitz.open('./figs/'+pdf)  # Open the PDF
    page = doc.load_page(0)  # Load the first page
    mat = fitz.Matrix(10, 10)
    pix = page.get_pixmap(matrix=mat)  # Render the page to an image with the matrix
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    images.append(img)

# Resize all images to the same size (if needed)
base_width, base_height = images[0].size
images = [img.resize((base_width, base_height)) for img in images]

columns = 3
# Create a blank image to paste the 3x4 grid
grid_img = Image.new('RGB', (base_width * columns, base_height * len(pdf_files) // columns))

# Paste the images into the grid
for i, img in enumerate(images):
    x = (i % columns) * base_width
    y = (i // columns) * base_height
    grid_img.paste(img, (x, y))

# Save the final image
grid_img.save('./figs/'+"few-shot.pdf", resolution=1600)
# grid_img.show()
