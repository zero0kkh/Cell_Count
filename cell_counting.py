# Version: 1.0.0_2024.Mar.14
# Purpose: Count the number of cells(NeuN) within a certain distance from the probe boundary
# Written by: Kanghyeon Kim
# Acknowledgement: I would like to thank Changhoon Sung for his active discussion and prompt feedback throughout this process
# Last updated: 2024.Mar.14

# README
# You should run the following command in your terminal:
# pip install scikit-image opencv-python

# References
# 1. (Korean) using scikit-image
#  ref: https://engineer-mole.tistory.com/15
# 2. (Korean) using ndimage (especially for labeling/regionprops)
#  ref: https://cumulu-s.tistory.com/38 

# Import the necessary libraries
import argparse
import os
import subprocess
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import (
    color,
    draw,
    exposure,
    filters,
    io,
    measure,
    morphology,
    segmentation,
)
from skimage.color import gray2rgb
from skimage.draw import polygon_perimeter


# process_image function is the main function that processes the tiff image and returns the histogram data and the number of cells
# thresh = 30 # (probe_mask) threshold value for the background/probe in the fourth channel
# um_per_px = 0.62  # micrometers per pixel
# ROI_um = 400  # 400 micrometers from cell to probe boundary
# radius = int(300/um_per_px)  # Adjust this radius to match the size of your probe area.
def process_image(directory_path, filename, thresh = 30, um_per_px = 0.62, ROI_um = 400, radius = 300, distanceMethod = 'Center') -> (int):
    hist_bin=10
    labeled_image_path = os.path.join(directory_path, f"{filename}_labeled.png")
    probe_image_path = os.path.join(directory_path, f"{filename}_probe.png")
    detected_image_path = os.path.join(directory_path, f"{filename}_detected.png")
    histogram_image_path = os.path.join(directory_path, f"{filename}_hist.png")
    histogram_csv_path = os.path.join(directory_path, f"{filename}_hist.csv")
    NeuNdata_csv_path = os.path.join(directory_path, f"{filename}_NeuNdata.csv") # NeuN data: CellCount, DistanceFromProbeBoundary
    
    radius = int(radius/um_per_px)
    tiff_path = './'+filename+'.tif'
    tiff_image = io.imread(tiff_path)
    first_channel = tiff_image[:, :, 0] # DAPI
    fourth_channel = tiff_image[:, :, 3] # NeuN

    # 1. Detect cells: Labeling the regions with preprocessing steps

    def detectImg(img):
        # Apply the preprocessing steps
        # equalized_image = cv2.equalizeHist(img)
        blurred_image = cv2.GaussianBlur(img, (5, 5), 0)
        thresh_value, binary_image = cv2.threshold(img, 0, 4095, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
        cleaned_image = morphology.remove_small_objects(binary_image > thresh_value, min_size=50)
        labeled_image, _ = ndimage.label(cleaned_image)
        props = measure.regionprops(labeled_image)
        return labeled_image, props

        """
        Histogram Equalization (cv2.equalizeHist): This step is meant to improve the contrast of the image. It can be very useful when you have an image with backgrounds and features that are close in intensity. However, if it's making your results worse, it's likely that your data does not benefit from this type of contrast enhancement, or it might be exacerbating noise. It’s good that you’ve removed it if it was not helping.
        Gaussian Blurring (cv2.GaussianBlur): Blurring helps to reduce image noise and detail. If your images have a lot of high-frequency noise, blurring can help make the regions of interest more distinct. If removing this step makes it harder to distinguish the cells from the background, then it's necessary. If the cells are still distinct without it, you might not need this step.
        Thresholding (cv2.threshold) with Otsu's method: This is used to separate the foreground (cells) from the background. Otsu’s method automatically determines a threshold value from the image histogram that minimizes the variance within the classes (foreground/background), which is generally a good approach when you have bimodal histograms. This step is almost always necessary for segmentation tasks.
        Removing Small Objects (morphology.remove_small_objects): This step removes artifacts from the image that are smaller than a specified size, which can help to clean up the image. It's usually necessary if you have small points of noise that are being incorrectly labeled as cells.
        """

    # labeled: labeled 2D image, props: region properties
    DAPI_labeled, DAPI_props = detectImg(first_channel) # DAPI_labeled: labeled 2D image, DAPI_props: region properties
    NeuN_labeled, NeuN_props = detectImg(fourth_channel) # NeuN_labeled: labeled 2D image, NeuN_props: region properties
    NeuN_cellcnt = len(NeuN_props)*[1] # Array to store the number of cells: NeuN_cellcnt[i] = number of cells in i-th region
    NeuN_dist = len(NeuN_props)*[0]  # Array to store the distances from the probe boundary: NeuN_dist[i] = distance of i-th cell from the probe boundary
    # Display the labeled images
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(DAPI_labeled, cmap='nipy_spectral')
    axes[0].set_title('DAPI Labeled')
    axes[0].axis('off')
    axes[1].imshow(NeuN_labeled, cmap='nipy_spectral')
    axes[1].set_title('NeuN Labeled')
    axes[1].axis('off')
    # plt.show()
    plt.savefig(labeled_image_path)



    # Count the number of cells NeuN considering DAPI as the reference
    # Check if there are matching regions in NeuN and DAPI

    if len(DAPI_props) > 0:
        new_NeuN_props = []

        # Iterate through NeuN regions
        for i, region_NeuN in enumerate(NeuN_props):
            n_DAPI = 0
            # Find matching DAPI regions for current NeuN region
            for region_DAPI in DAPI_props:
                if region_NeuN.bbox[0] >= region_DAPI.bbox[0] and region_NeuN.bbox[1] >= region_DAPI.bbox[1] and region_NeuN.bbox[2] <= region_DAPI.bbox[2] and region_NeuN.bbox[3] <= region_DAPI.bbox[3]:
                    n_DAPI += 1
            NeuN_cellcnt[i] = max(n_DAPI,1)


    # 2. Probe Mask

    # Since we know the fourth channel can have different types, normalize and convert it to uint8
    if fourth_channel.dtype != np.uint8:
        fourth_channel_normalized = cv2.normalize(fourth_channel, None, alpha=0, beta=255,
                                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        fourth_channel_uint8 = fourth_channel_normalized.astype(np.uint8)
    else:
        fourth_channel_uint8 = fourth_channel

    if first_channel.dtype != np.uint8:
        first_channel_normalized = cv2.normalize(first_channel, None, alpha=0, beta=255,
                                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        first_channel_uint8 = first_channel_normalized.astype(np.uint8)
    else:
        first_channel_uint8 = first_channel

    # Center the probe and create a mask (largest black area including the center)
    first_channel_uint8 += 1  # To seperate the inner region from the outside of the circel
    height, width = first_channel_uint8.shape
    image_center = np.array(first_channel_uint8.shape) // 2
    center_y, center_x = height // 2, width // 2
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    circle_mask = dist_from_center <= radius
    central_region = first_channel_uint8 * circle_mask
    # thresh = filters.threshold_otsu(central_region[circle_mask]) # If you want to use the threshold value through aitonmatic otsu method
    binary_probe_mask = np.logical_and(central_region > 0, central_region < thresh)
    labeled_array, num_features = ndimage.label(binary_probe_mask)
    central_region_label = labeled_array[image_center[0], image_center[1]]
    probe_mask = labeled_array == central_region_label
    # probe_mask = segmentation.expand_labels(labeled_array == central_region_label, distance=5)
    # probe_mask = filled_probe_mask

    # visualize the probe area
    probe_image = gray2rgb(fourth_channel_uint8)
    probe_image[probe_mask, 0] = 255  # R 채널
    probe_image[probe_mask, 1] = 0    # G 채널
    probe_image[probe_mask, 2] = 0    # B 채널
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(probe_image, cmap='gray')
    ax.set_title('Fourth Channel with Probe Area')
    ax.axis('off')
    # plt.show()
    plt.savefig(probe_image_path)

    # 3. Distance
    # Calculate the distance of each cell from the probe boundary
    probe_mask_coords = np.argwhere(probe_mask)
    # probe_center = np.array(fourth_channel_uint8.shape) // 2 # Assert the probe center is equal to the image center

    for i, prop in enumerate(NeuN_props):
        if distanceMethod == 'Center':
            cell_centroid = prop.centroid
            distances = cdist([cell_centroid], probe_mask_coords, metric='euclidean').flatten()
        elif distanceMethod == 'Boundary':
            cell_coords = prop.coords
            distance_transform = ndimage.distance_transform_edt(~probe_mask)
            distances = distance_transform[cell_coords[:, 0], cell_coords[:, 1]]
        distance_to_probe_edge_um = np.min(distances) * um_per_px
        NeuN_dist[i] = distance_to_probe_edge_um  # 계산된 거리를 리스트에 저장
        # NeuN_dist[i] = min_distance(prop.centroid, probe_mask_coords)
        if distance_to_probe_edge_um > ROI_um: NeuN_cellcnt[i] = 0  # 프로브에서 400 마이크로미터 이상 떨어진 세포는 제외

    # Collect these distances and filter for those within 400 micrometers
    distances_within_ROI = []
    for i, d in enumerate(NeuN_dist):
        for j in range(NeuN_cellcnt[i]):
            distances_within_ROI.append(d)

    # Create the histogram with 50 micrometer bins
    plt.hist(distances_within_ROI, bins=np.arange(0, 401, hist_bin), color='blue', edgecolor='black')
    plt.title('Histogram of Cell Distances from Probe Boundary')
    plt.xlabel('Distance (micrometers)')
    plt.ylabel('Number of Cells')
    # plt.show()
    plt.savefig(histogram_image_path)
    # Save the histogram data to a CSV file
    # We count the number of occurrences per bin
    counts, bin_edges = np.histogram(distances_within_ROI, bins=np.arange(min(distances_within_ROI), max(distances_within_ROI) + hist_bin, hist_bin))
    histogram_data = pd.DataFrame({'BinStart': bin_edges[:-1], 'Count': counts})
    histogram_data.to_csv(histogram_csv_path, index=False)

    # 4. Visualize the detected results with the boundaries of the cells
    # Color Reference
    colors = [
        [255, 0, 0],  # Red
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [255, 255, 0], # Yellow
        [255, 165, 0], # Orange
        [255, 0, 255], # Magenta
        [0, 255, 255], # Cyan
        [192, 192, 192] # Silver
    ]

    boundaries_image = gray2rgb(fourth_channel_uint8)
    for i, region in enumerate(NeuN_props):
        if NeuN_cellcnt[i] == 0: continue
        else:
            # Draw boundaries around each cell
            coordinates = region.coords
            for coord in coordinates:
                if NeuN_dist[i]<100: boundaries_image[coord[0], coord[1]] = colors[0]  # Red color
                elif NeuN_dist[i]<200: boundaries_image[coord[0], coord[1]] = colors[4] # Orange
                elif NeuN_dist[i]<300: boundaries_image[coord[0], coord[1]] = colors[3] # Yellow
                else: boundaries_image[coord[0], coord[1]] = colors[1]  # Green

    plt.imshow(boundaries_image, cmap='gray')
    plt.title('Fourth Channel: ROI'+str(ROI_um)+'um')
    plt.axis('off')
    # plt.show()
    plt.savefig(detected_image_path)


    # 5. Count the total number of NeuN cells
    total_NeuN_cells = sum(NeuN_cellcnt)
    print(filename+"'s Total number of NeuN cells:", total_NeuN_cells)

    data = {'CellCount': NeuN_cellcnt, 'DistanceFromProbeBoundary': NeuN_dist}
    df = pd.DataFrame(data)
    df.to_csv(NeuNdata_csv_path, index=False)
    return total_NeuN_cells

def main(args):

    directory_path = './' # If this code file is on the same directory with the tiff files. Otherwise, you should change the directory path like './subfolder'
    total_cells_count = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.tif'):
            file_path = os.path.join(directory_path, filename)
            base_filename = os.path.splitext(filename)[0] # Extract the base filename (without the extension 'tif')
            cell_count = process_image(directory_path, base_filename, thresh = args.thresh, um_per_px = args.um_per_px, ROI_um = args.ROI_um, radius = args.radius, distanceMethod = args.distanceMethod)
            total_cells_count.append(cell_count)

    # 모든 파일에 대한 총 세포 수를 CSV 파일로 저장합니다.
    total_cells_df = pd.DataFrame({'Total_NeuN_Cells': [total_cells_count]})
    total_cells_csv_path = os.path.join(directory_path, 'total_NeuN_cells.csv')
    total_cells_df.to_csv(total_cells_csv_path, index=False)
    print('Average number of NeuN cells:', np.mean(total_cells_count))


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--distanceMethod', default="Center", type=str, help="Calculate distance from the probe boundary to Center / Boundary of the cells")
    parser.add_argument('--thresh', type=int, default=30, help="Threshold value for the background/probe in the fourth channel")
    parser.add_argument('--um_per_px', type=float, default=0.62, help="Micrometers per pixel")
    parser.add_argument('--ROI_um', type=int, default=400, help="400 micrometers from cell to probe boundary")
    parser.add_argument('--radius', type=int, default=300, help="Adjust this radius to detect the probe area [micrometer]")
    # Feel free to add more arguments here if you need!

    args = parser.parse_args()
    main(args)