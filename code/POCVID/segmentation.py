import os
import re
import numpy as np
from skimage.transform import resize
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

os.makedirs("output", exist_ok=True)


def resize_image(img, block_size):
    resized_image = resize(
        img,
        (
            img.shape[0] // block_size * block_size,
            img.shape[1] // block_size * block_size,
        ),
    )

    return resized_image


def get_patches(image, block_size):
    coords = [
        (y, x)
        for y in range(0, image.shape[0], block_size)
        for x in range(0, image.shape[1], block_size)
    ]
    patch_list = np.array(
        [image[y : y + block_size, x : x + block_size] * 255 for y, x in coords]
    ).astype(np.uint8)
    return np.array(coords), patch_list


def extract_glcm_features(patches, distances):
    diss, corr, energy, contrast, homo = [], [], [], [], []
    for patch in patches:
        glcm = graycomatrix(
            patch,
            distances=distances,
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            normed=True,
        )
        diss.append(graycoprops(glcm, "dissimilarity")[0, 0])
        corr.append(abs(graycoprops(glcm, "correlation")[0, 0]))
        energy.append(graycoprops(glcm, "energy")[0, 0])
        contrast.append(graycoprops(glcm, "contrast")[0, 0])
        homo.append(graycoprops(glcm, "homogeneity")[0, 0])

    return (
        np.array(diss),
        np.array(corr),
        np.array(energy),
        np.array(contrast),
        np.array(homo),
    )


def threshold_values(xs, ys, zs, contrast_vals, homo_vals):
    def calculate_iqr_threshold(data, upper=True):
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        return q3 + 1.5 * iqr if upper else q1 - 1.5 * iqr

    xs_thr = calculate_iqr_threshold(xs, upper=True)
    ys_thr = calculate_iqr_threshold(ys, upper=False)
    zs_thr = 0.055
    contrast_thr = calculate_iqr_threshold(contrast_vals, upper=True)
    homo_thr = calculate_iqr_threshold(homo_vals, upper=False)

    return xs_thr, ys_thr, zs_thr, contrast_thr, homo_thr


def determine_optimal_clusters(coordinates, condition, max_clusters):
    optimal_score = -1
    optimal_k = 2
    score_list = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        cluster_labels = kmeans.fit_predict(coordinates[condition])
        score = silhouette_score(coordinates[condition], cluster_labels)
        score_list.append(score)

        if score > optimal_score:
            optimal_score = score
            optimal_k = k

    return optimal_k


def save_segmented_image(
    image, coordinates, condition, labels, block_size, output_path
):
    unique_labels = np.unique(labels)

    markers = np.zeros_like(image, dtype=np.int32)
    for i, cluster_id in enumerate(unique_labels, start=1):
        cluster_coords = coordinates[condition][labels == cluster_id]
        for y, x in cluster_coords:
            markers[y : y + block_size, x : x + block_size] = cluster_id

    watershed_markers = cv2.watershed(
        cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), markers
    )

    resized_markers = cv2.resize(
        watershed_markers, (512, 512), interpolation=cv2.INTER_NEAREST
    )

    output_image = np.clip(resized_markers, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, output_image)


def process_image(image_file, block_size, distances, max_clusters):
    print(f"Processing Image: {image_file}")

    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    img = resize_image(img, block_size)
    coordinates, patches = get_patches(img, block_size)
    xs, ys, zs, contrast_vals, homo_vals = extract_glcm_features(patches, distances)

    xs_thr, ys_thr, zs_thr, contrast_thr, homo_thr = threshold_values(
        xs, ys, zs, contrast_vals, homo_vals
    )

    condition = (
        (xs < xs_thr)
        & (ys > ys_thr)
        & (zs > zs_thr)
        & (contrast_vals < contrast_thr)
        & (homo_vals > homo_thr)
    )

    optimal_clusters = determine_optimal_clusters(
        coordinates, condition, max_clusters=max_clusters
    )
    kmeans = KMeans(n_clusters=optimal_clusters, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(coordinates[condition])

    base_name = os.path.basename(image_file)
    output_name = base_name.replace("tm", "seg")
    output_path = os.path.join("output", output_name)

    save_segmented_image(img, coordinates, condition, labels, block_size, output_path)


def process_directory(input_directory, block_size, distances, max_clusters):
    def extract_number(filename):
        match = re.search(r"\d+", filename)
        return int(match.group()) if match else -1

    image_files = [
        f
        for f in os.listdir(input_directory)
        if f.startswith("tm") and f.endswith(".png")
    ]

    sorted_image_files = sorted(image_files, key=extract_number)

    for image_file in sorted_image_files:
        image_path = os.path.join(input_directory, image_file)
        process_image(image_path, block_size, distances, max_clusters)


process_directory("data", 30, [1, 2], 12)
print("Segmentation Complete")
