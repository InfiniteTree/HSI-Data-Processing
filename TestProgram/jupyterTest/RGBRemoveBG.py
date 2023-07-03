from PIL import Image, ImageFilter
import numpy as np
import cv2

def rgb_to_hsv(rgb_img):
    hsv_img = rgb_img.convert('HSV')
    return hsv_img

def threshold_segmentation(hsv_img, hue_threshold, saturation_threshold, value_threshold):
    # Do Image segmentation by using the domain range of HSV to get the background
    np_img = np.array(hsv_img)
    hue_channel = np_img[:, :, 0]
    saturation_channel = np_img[:, :, 1]
    value_channel = np_img[:, :, 2]

    # Create a binary mask
    mask = ((hue_channel <= hue_threshold[1]) & (hue_channel >= hue_threshold[0]) &
            (saturation_channel <= saturation_threshold[1]) & (saturation_channel >= saturation_threshold[0]) &
            (value_channel <= value_threshold[1]) & (value_channel >= value_threshold[0]))

    # Use the mask
    segmented_img = np.zeros_like(np_img)
    segmented_img[mask] = np_img[mask]

    # Convert an array to an image
    segmented_img = Image.fromarray(segmented_img, 'HSV')

    return segmented_img

def remove_background(rgb_img, hue_threshold, saturation_threshold, value_threshold):
    image = Image.open(rgb_img)
    # change to HSV
    hsv_img = rgb_to_hsv(image)
    # Threshold splitting to get the background
    segmented_img = threshold_segmentation(hsv_img, hue_threshold, saturation_threshold, value_threshold)
    # change back to RGB
    segmented_img = segmented_img.convert('RGB')
    #filtered_img = cv2.GaussianBlur(segmented_img, (5,5), 0, 0)
    #filtered_img = cv2.blur(segmented_img,(5,5))
    segmented_img = segmented_img.filter(ImageFilter.SMOOTH)
    segmented_img.show()
    segmented_img.save('figures/wheat/test/test_new.jpg')

if __name__ == "__main__":
    # Define the threshold of HSV
    hue_threshold = (20, 100)  # Domain of H (0, 360)
    saturation_threshold = (60, 255)  # Domain of S (0, 255)
    value_threshold = (60, 255)  # Domain of OF V (0, 255)

    rgb_image_path = 'figures/wheat/test/test.jpg'

    remove_background(rgb_image_path, hue_threshold, saturation_threshold, value_threshold)

