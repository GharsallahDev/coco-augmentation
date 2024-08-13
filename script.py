import cv2
import json
import os
import numpy as np
import argparse
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_args():
    parser = argparse.ArgumentParser(description='Augment image dataset and update annotations using configuration file')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file')
    return parser.parse_args()

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image

def apply_noise(image, salt_vs_pepper_ratio=0.1):
    noisy_image = np.copy(image)
    salt = np.random.rand(*image.shape) < salt_vs_pepper_ratio / 2
    pepper = np.random.rand(*image.shape) < salt_vs_pepper_ratio / 2
    noisy_image[salt] = 255
    noisy_image[pepper] = 0
    return noisy_image

def enhance_contrast(image, alpha=1.5, beta=0):
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced_image

def adjust_brightness(image, alpha=1.5, beta=50):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def adjust_saturation(image, saturation_factor=1.5):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * saturation_factor, 0, 255)
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return adjusted_image

def apply_histogram_equalization(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab_image))
    lab_planes[0] = cv2.equalizeHist(lab_planes[0])
    equalized_lab_image = cv2.merge(lab_planes)
    equalized_image = cv2.cvtColor(equalized_lab_image, cv2.COLOR_LAB2BGR)
    return equalized_image

def apply_color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    delta_brightness = np.random.uniform(-brightness, brightness)
    hsv_image[..., 2] = np.clip(hsv_image[..., 2] + delta_brightness * 255, 0, 255)

    delta_contrast = np.random.uniform(1 - contrast, 1 + contrast)
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * delta_contrast, 0, 255)

    delta_saturation = np.random.uniform(1 - saturation, 1 + saturation)
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * delta_saturation, 0, 255)

    jittered_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return jittered_image

def apply_perspective_warp(image, warp_factor=0.3):
  height, width = image.shape[:2]

  max_warp_range = int(warp_factor * min(height, width))

  offset = int(max_warp_range * 0.1)
  src_points = np.float32([[offset, offset], [width - offset, offset], [width - offset, height - offset], [offset, height - offset]])

  warp_range_x = np.random.randint(-max_warp_range, max_warp_range, size=2)
  warp_range_y = np.random.randint(-max_warp_range, max_warp_range, size=2)
  dst_points = np.float32([[warp_range_x[0], warp_range_y[0]], 
                           [width + warp_range_x[1], warp_range_y[0]], 
                           [width + warp_range_x[1], height + warp_range_y[1]], 
                           [warp_range_x[0], height + warp_range_y[1]]])

  transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
  warped_image = cv2.warpPerspective(image, transform_matrix, (width, height))
  return warped_image, src_points, dst_points

def adjust_bbox_for_apply_perspective_warp(bbox, src_points, dst_points):
  x_min, y_min, width, height = bbox
  points = np.float32([[x_min, y_min], [x_min + width, y_min], [x_min + width, y_min + height], [x_min, y_min + height]])
  warped_points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points)))
  warped_points = warped_points.reshape(4, 2)

  x_mins = np.amin(warped_points[:, 0], axis=0)
  x_maxs = np.amax(warped_points[:, 0], axis=0)
  y_mins = np.amin(warped_points[:, 1], axis=0)
  y_maxs = np.amax(warped_points[:, 1], axis=0)

  adjusted_bbox = [int(x_mins), int(y_mins), int(x_maxs - x_mins), int(y_maxs - y_mins)]
  return adjusted_bbox

def apply_random_erasing(image, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3):
    if np.random.rand() > p:
        return image, None 

    h, w, _ = image.shape
    s = np.random.uniform(s_l, s_h) * h * w
    r = np.random.uniform(r_1, r_2)
    
    eraser_w = int(np.sqrt(s / r))
    eraser_h = int(np.sqrt(s * r))
    
    eraser_w = min(w, eraser_w)
    eraser_h = min(h, eraser_h)

    x = np.random.randint(0, w - eraser_w)
    y = np.random.randint(0, h - eraser_h)

    eraser = np.random.randint(0, 255, (eraser_h, eraser_w, 3), dtype=np.uint8)

    image[y:y + eraser_h, x:x + eraser_w, :] = eraser

    return image, (x, y, eraser_w, eraser_h)

def adjust_bbox_for_occlusion(bbox, occlusion_rect):
    bx, by, bw, bh = bbox
    ox, oy, ow, oh = occlusion_rect

    ix = max(bx, ox)
    iy = max(by, oy)
    ix2 = min(bx+bw, ox+ow)
    iy2 = min(by+bh, oy+oh)

    if ix < ix2 and iy < iy2:
        if ix == bx and iy == by:
            return [ix2, iy2, (bx + bw) - ix2, (by + bh) - iy2]
        elif ix2 == (bx + bw) and iy2 == (by + bh):
            return [bx, by, ix - bx, iy - by]
    return bbox

def apply_shear(image, shear_factor, direction='horizontal'):
    h, w = image.shape[:2]
    if direction == 'horizontal':
        M = np.array([[1, shear_factor, 0],
                      [0, 1, 0]], dtype=np.float32)
    else:
        M = np.array([[1, 0, 0],
                      [shear_factor, 1, 0]], dtype=np.float32)
    sheared_image = cv2.warpAffine(image, M, (w, h))
    return sheared_image

def adjust_bbox_for_shearing(bbox, shear_factor, direction='horizontal'):
    x_min, y_min, width, height = bbox
    
    corners = np.array([
        [x_min, y_min],
        [x_min + width, y_min],
        [x_min, y_min + height],
        [x_min + width, y_min + height]
    ])

    if direction == 'horizontal':
        M = np.array([
            [1, shear_factor, 0],
            [0, 1, 0]
        ], dtype=np.float32)
    else:
        M = np.array([
            [1, 0, 0],
            [shear_factor, 1, 0]
        ], dtype=np.float32)
    
    ones = np.ones(shape=(len(corners), 1))
    corners_ones = np.hstack([corners, ones])
    transformed_corners = M.dot(corners_ones.T).T
    
    x_coords = transformed_corners[:, 0]
    y_coords = transformed_corners[:, 1]
    new_x_min = np.min(x_coords)
    new_y_min = np.min(y_coords)
    new_x_max = np.max(x_coords)
    new_y_max = np.max(y_coords)
    
    new_bbox = [int(new_x_min), int(new_y_min), int(new_x_max - new_x_min), int(new_y_max - new_y_min)]
    return new_bbox

def apply_scaling(image, scale_factor):
    new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    scaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return scaled_image

def adjust_bbox_for_scaling(bbox, scale_factor):
    x_min, y_min, width, height = bbox
    return [int(x_min * scale_factor), int(y_min * scale_factor), int(width * scale_factor), int(height * scale_factor)]

def calculate_dynamic_zoom_factor(image, bbox):
    img_height, img_width = image.shape[:2]
    x_min, y_min, bbox_width, bbox_height = bbox

    focus_scale = 0.9
    zoom_factor_width = img_width / (bbox_width / focus_scale)
    zoom_factor_height = img_height / (bbox_height / focus_scale)
    zoom_factor = min(zoom_factor_width, zoom_factor_height, 2.0)

    return min(max(zoom_factor, 1.0), 2.0)

def apply_zoom(image, zoom_factor):
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    radius_x, radius_y = int(center_x / zoom_factor), int(center_y / zoom_factor)

    cropped = image[center_y - radius_y:center_y + radius_y, center_x - radius_x:center_x + radius_x]
    zoomed_image = cv2.resize(cropped, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return zoomed_image

def adjust_bbox_for_zooming(bbox, zoom_factor, original_width, original_height):
    x_min, y_min, width, height = bbox
    center_x, center_y = original_width // 2, original_height // 2
    radius_x, radius_y = int(center_x / zoom_factor), int(center_y / zoom_factor)

    new_x_min = max(0, int((x_min - (center_x - radius_x)) * zoom_factor))
    new_y_min = max(0, int((y_min - (center_y - radius_y)) * zoom_factor))
    new_width = min(int(width * zoom_factor), original_width - new_x_min)
    new_height = min(int(height * zoom_factor), original_height - new_y_min)

    return [new_x_min, new_y_min, new_width, new_height]

def rotate_image_90_degrees(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def adjust_bbox_for_rotation_90(bbox, img_width):
    x_min, y_min, width, height = bbox
    new_x_min = y_min
    new_y_min = img_width - x_min - width
    new_width = height
    new_height = width
    return [new_x_min, new_y_min, new_width, new_height]

def adjust_bbox_for_rotation_270(bbox, img_height):
    x_min, y_min, width, height = bbox
    new_x_min = img_height - y_min - height
    new_y_min = x_min
    new_width = height
    new_height = width
    return [new_x_min, new_y_min, new_width, new_height]

def apply_horizontal_flip(image):
    return cv2.flip(image, 1)

def adjust_bbox_for_apply_horizontal_flip(bbox, img_width):
    x_min, y_min, width, height = bbox
    return [img_width - x_min - width, y_min, width, height]

def apply_vertical_flip(image):
    return cv2.flip(image, 0)

def adjust_bbox_for_apply_vertical_flip(bbox, img_height):
    x_min, y_min, width, height = bbox
    return [x_min, img_height - y_min - height, width, height]

def adjust_bbox_for_flip_both(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    new_x_min = img_width - x_min - width
    new_y_min = img_height - y_min - height
    return [new_x_min, new_y_min, width, height]

def adjust_bbox_for_rotation_90_and_apply_horizontal_flip(bbox, img_width, img_height):
    rotated_x_min = bbox[1]
    rotated_y_min = img_width - bbox[0] - bbox[2]
    rotated_width = bbox[3]
    rotated_height = bbox[2]
    
    flipped_x_min = img_width - rotated_x_min - rotated_width
    
    return [flipped_x_min, rotated_y_min, rotated_width, rotated_height]

def adjust_bbox_for_rotation_90_and_apply_vertical_flip(bbox, img_width, img_height):
    rotated_x_min = bbox[1]
    rotated_y_min = img_width - bbox[0] - bbox[2]
    rotated_width = bbox[3]
    rotated_height = bbox[2]
    
    flipped_y_min = img_height - rotated_y_min - rotated_height
    
    return [rotated_x_min, flipped_y_min, rotated_width, rotated_height]

def main():
    opt = get_args()

    try:
        with open(opt.config, 'r') as file:
            config = json.load(file)
    except Exception as e:
        logging.error(f"Failed to load configuration file: {e}")
        return

    if not os.path.exists(config["images_dir"]) or not os.listdir(config["images_dir"]):
        logging.error(f"Specified image directory does not exist or is empty: {config['images_dir']}")
        return
    
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"], exist_ok=True)
        logging.info(f"Created output directory: {config['output_dir']}")

    try:
        with open(config["annotations_json"], "r") as file:
            json_data = json.load(file)
        if "images" not in json_data or "annotations" not in json_data:
            logging.error("JSON file is missing required 'images' or 'annotations' keys")
            return
    except Exception as e:
        logging.error(f"Failed to read JSON file: {e}")
        return
    
    transforms = config["transforms"]
    directory_path = config["images_dir"]
    output_dir = config["output_dir"]
    json_path = config["annotations_json"]

    file_names = [f for f in os.listdir(directory_path) if f.lower().endswith(".jpg")]

    try:
        with open(json_path, "r") as file:
            json_data = json.load(file)
    except IOError as e:
        logging.error(f"Failed to read JSON file: {e}")
        return

    last_id_images = int(json_data["images"][-1]["id"]) if json_data["images"] else 0
    last_id_annotations = int(json_data["annotations"][-1]["id"]) if json_data["annotations"] else 0

    name_list = [
    "_rotated_90.jpg", "_rotated_270.jpg", "_flipped_vertical.jpg", "_flipped_horizontal.jpg",
    "_flipped_both.jpg", "_rotated_90_flipped_horizontal.jpg", "_rotated_90_flipped_vertical.jpg",
    "_scaled.jpg", "_zoomed.jpg", "_sheared_x.jpg", "_sheared_y.jpg", "_warped.jpg", 
    "_color_jittered.jpg", "_gaussian_blurred.jpg","_salt_and_pepper_noise.jpg", "_contrast_enhanced.jpg",
    "_brightness_adjusted.jpg", "_saturation_adjusted.jpg", "_histogram_equalized.jpg",
    "_erased.jpg",
    ]

    flag_list = [
        transforms['rotate_90'], transforms['rotate_270'], transforms['flip_vertical'], transforms['flip_horizontal'],
        transforms['flip_both'], transforms['rotate_90_flip_horizontal'], transforms['rotate_90_flip_vertical'],
        transforms['scale'], transforms['zoom'], transforms['shear_x'], transforms['shear_y'],
        transforms['perspective_warp'], transforms['color_jitter'], transforms['gaussian_blur'], transforms['noise'],
        transforms['contrast'],transforms['brightness'], transforms['saturation'], transforms['histogram_equalization'], 
        transforms['erase'],
    ]
    
    new_images = []
    new_annotations = []
    image_id_converted = {}

    for file_name in file_names:
        path = os.path.join(directory_path, file_name)
        try:
            original_image = cv2.imread(path)
            if original_image is None:
                raise ValueError("Image could not be read or is not a valid image file.")
        except Exception as e:
            logging.warning(f"Failed to read image {file_name}, skipping. Error: {e}")
            continue
        
        height, width, _ = original_image.shape
        base_name = file_name[:-4]
        image_id = find_image_id_by_name(base_name + ".jpg", json_data["images"])

        if image_id is None:
            continue

        for annotation in json_data["annotations"]:
            if annotation["image_id"] == image_id:
                original_bbox = annotation["bbox"]
                scale_factor = random.uniform(0.75, 1.25)
                zoom_factor = calculate_dynamic_zoom_factor(original_image, original_bbox)
                shear_factor = random.uniform(-0.2, 0.2)

                for suffix, flag in zip(name_list, flag_list):
                    if flag:
                        transformed_image, adjusted_bbox = transform_and_adjust_bbox(
                            original_image, original_bbox, suffix, width, height,
                            zoom_factor if suffix == "_zoomed.jpg" else None,
                            scale_factor if suffix == "_scaled.jpg" else None,
                            shear_factor if suffix in ("_sheared_x.jpg", "_sheared_y.jpg") else None
                        )

                        new_height, new_width = transformed_image.shape[:2]
                        new_id = last_id_images + 1
                        new_file_name = base_name + suffix
                        cv2.imwrite(os.path.join(output_dir, new_file_name), transformed_image)
                        new_images.append({
                            "id": new_id,
                            "license": json_data["images"][0]["license"],
                            "file_name": new_file_name,
                            "height": new_height,
                            "width": new_width,
                            "date_captured": json_data["images"][0]["date_captured"],
                        })
                        image_id_converted[base_name + suffix] = new_id
                        last_id_images += 1

                        new_annotation = annotation.copy()
                        new_annotation["id"] = last_id_annotations + 1
                        new_annotation["image_id"] = new_id
                        new_annotation["bbox"] = adjusted_bbox

                        new_annotations.append(new_annotation)
                        last_id_annotations += 1

    json_data["images"].extend(new_images)
    json_data["annotations"].extend(new_annotations)

    try:
        with open(json_path, "w") as file:
            json.dump(json_data, file, indent=4)
        logging.info("Dataset augmented successfully.")
    except IOError as e:
        logging.error(f"Failed to write JSON file: {e}")

def find_image_id_by_name(image_name, images):
    for image in images:
        if image["file_name"] == image_name:
            return image["id"]
    return None

def transform_and_adjust_bbox(image, bbox, suffix, img_width, img_height, zoom_factor=None, scale_factor=None, shear_factor=None):
    if suffix == "_rotated_90.jpg":
        image = rotate_image_90_degrees(image)
        bbox = adjust_bbox_for_rotation_90(bbox, img_width)
    elif suffix == "_rotated_270.jpg":
        image = rotate_image_90_degrees(rotate_image_90_degrees(rotate_image_90_degrees(image)))
        bbox = adjust_bbox_for_rotation_270(bbox, img_height)
    elif suffix == "_flipped_horizontal.jpg":
        image = apply_horizontal_flip(image)
        bbox = adjust_bbox_for_apply_horizontal_flip(bbox, img_width)
    elif suffix == "_flipped_vertical.jpg":
        image = apply_vertical_flip(image)
        bbox = adjust_bbox_for_apply_vertical_flip(bbox, img_height)
    elif suffix == "_flipped_both.jpg":
        image = apply_vertical_flip(apply_horizontal_flip(image))
        bbox = adjust_bbox_for_flip_both(bbox, img_width, img_height)
    elif suffix == "_rotated_90_flipped_horizontal.jpg":
        image = apply_horizontal_flip(rotate_image_90_degrees(image))
        bbox = adjust_bbox_for_rotation_90_and_apply_horizontal_flip(bbox, img_width, img_height)
    elif suffix == "_rotated_90_flipped_vertical.jpg":
        image = apply_vertical_flip(rotate_image_90_degrees(image))
        bbox = adjust_bbox_for_rotation_90_and_apply_vertical_flip(bbox, img_width, img_height)
    elif suffix == "_scaled.jpg":
        image = apply_scaling(image, scale_factor)
        bbox = adjust_bbox_for_scaling(bbox, scale_factor)
    elif suffix == "_zoomed.jpg":
        image = apply_zoom(image, zoom_factor)
        bbox = adjust_bbox_for_zooming(bbox, zoom_factor, img_width, img_height)
    elif suffix == "_sheared_x.jpg":
        image = apply_shear(image, shear_factor, direction='vertical')
        bbox = adjust_bbox_for_shearing(bbox, shear_factor,'vertical')
    elif suffix == "_sheared_y.jpg":
        image = apply_shear(image, shear_factor, direction='horizontal')
        bbox = adjust_bbox_for_shearing(bbox, shear_factor,'horizontal')
    elif suffix == "_warped.jpg":
        image, src_points, dst_points = apply_perspective_warp(image, warp_factor=0.3)
        bbox = adjust_bbox_for_apply_perspective_warp(bbox, src_points, dst_points)
    elif suffix == "_color_jittered.jpg":
        image = apply_color_jitter(image)
    elif suffix == "_gaussian_blurred.jpg":
        image = apply_gaussian_blur(image)
    elif suffix == "_salt_and_pepper_noise.jpg":
        image = apply_noise(image)
    elif suffix == "_contrast_enhanced.jpg":
        image = enhance_contrast(image)
    elif suffix == "_brightness_adjusted.jpg":
        image = adjust_brightness(image)
    elif suffix == "_saturation_adjusted.jpg":
        image = adjust_saturation(image)
    elif suffix == "_histogram_equalized.jpg":
        image = apply_histogram_equalization(image)
    elif suffix == "_erased.jpg":
        image, occlusion_rect = apply_random_erasing(image)
        if occlusion_rect:
            bbox = adjust_bbox_for_occlusion(bbox, occlusion_rect)
    return image, bbox

if __name__ == '__main__':
    main()