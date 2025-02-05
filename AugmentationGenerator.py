import cv2
import torch
import urllib.request
import os
import random
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

# TODO:
# - add filepath for depth map if repeated augs of same photo
# - 

class AugmentationGenerator():
    def __init__(self, path_to_image, model_type="DPT_Large"):
        self.image = cv2.imread(path_to_image)
        
        self.image = cv2.copyMakeBorder(
            self.image, 150, 150, 150, 150, cv2.BORDER_CONSTANT, value=(0, 0, 0) # see if this affecs the depth map
        )
        self.model_type = model_type
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def generateDepthMap(self):
        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.cpu().numpy()

    def rotate(self, image=None, displacement_x=-50, displacement_y=-50, depth_map=None):
        """Apply depth-based displacement (rotation simulation)."""
        if image is None:
            image = self.image  # Use self.image if no image is provided
        if depth_map is None:
            depth_map = self.generateDepthMap()

        depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
        h, w = depth_map.shape
        image = cv2.resize(image, (w, h))

        scaled_depth = np.sqrt(depth_map)
        displaced_image = np.zeros_like(image)
        valid_mask = np.zeros((h, w), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                shift_x = int(displacement_x * scaled_depth[y, x])
                shift_y = int(displacement_y * scaled_depth[y, x])
                new_x = min(w - 1, max(0, x + shift_x))
                new_y = min(h - 1, max(0, y + shift_y))
                
                displaced_image[new_y, new_x] = image[y, x]
                valid_mask[new_y, new_x] = 1 

        return displaced_image

    def zoom(self, image=None, zoom_factor=1.2):
        """Apply zoom to an image."""
        if image is None:
            image = self.image  # Use self.image if no image is provided

        h, w = image.shape[:2]
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        zoomed_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if zoom_factor >= 1:
            start_x, start_y = (new_w - w) // 2, (new_h - h) // 2
            return zoomed_image[start_y:start_y + h, start_x:start_x + w]
        else:
            top_pad, bottom_pad = (h - new_h) // 2, h - new_h - (h - new_h) // 2
            left_pad, right_pad = (w - new_w) // 2, w - new_w - (w - new_w) // 2
            return cv2.copyMakeBorder(zoomed_image, top_pad, bottom_pad, left_pad, right_pad, 
                                      borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def apply_augmentation(self, displacement_x=-350, displacement_y=-100, zoom_factor=1.2):
        """Apply rotation first, then zoom on the rotated image."""
        rotated = self.rotate(displacement_x=displacement_x, displacement_y=displacement_y)
        augmented_image = self.zoom(rotated, zoom_factor)
        return augmented_image

if __name__ == '__main__':
    path_to_image="Downloads/Subject_2.png"
    output_dirs = ["output/combined_s2"] # "output/rotation", "output/zoom",
    
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)
    
    generator = AugmentationGenerator(path_to_image)
    
    for i in range(300):
        disp_x = random.randint(-650, 650)
        disp_y = random.randint(-650, 650) # random for now. will probably take some tweaking
        zoom_factor = random.uniform(0.35, 1.35)

        # rotated_img = generator.rotate(displacement_x=disp_x, displacement_y=disp_y)
        # zoomed_img = generator.zoom(zoom_factor=zoom_factor)
        combined_img = generator.apply_augmentation(disp_x, disp_y, zoom_factor)

        # cv2.imwrite(f"output/rotation/rotated_{i}.png", rotated_img)
        # cv2.imwrite(f"output/zoom/zoomed_{i}.png", zoomed_img)
        cv2.imwrite(f"output/combined_s2/combined_{i}.png", combined_img)
    
    print("Augmented images saved successfully!")
    # python3 AugmentationGenerator.py

    # Displays:

    # Initialize the generator with the image path
    # gen = AugmentationGenerator(path_to_image="Downloads/Subject_2.png")

    # # Test with different parameters for rotation and zoom
    # img_rotated = gen.rotate(displacement_x=-50, displacement_y=50)
    # img_zoomed = gen.zoom(zoom_factor=1.2)
    # img_augmented = gen.apply_augmentation(displacement_x=-50, displacement_y=50, zoom_factor=1.2)

    # # Show the augmented images
    # plt.figure(figsize=(10, 5))

    # plt.subplot(1, 4, 1)
    # plt.imshow(cv2.cvtColor(gen.image, cv2.COLOR_BGR2RGB))
    # plt.title("Original")
    
    # plt.subplot(1, 4, 2)
    # plt.imshow(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))
    # plt.title("Augmented (Rotation)")

    # plt.subplot(1, 4, 3)
    # plt.imshow(cv2.cvtColor(img_zoomed, cv2.COLOR_BGR2RGB))
    # plt.title("Augmented (Zoom)")

    # plt.subplot(1, 4, 4)
    # plt.imshow(cv2.cvtColor(img_augmented, cv2.COLOR_BGR2RGB))
    # plt.title("Combined")
    
    # plt.show()