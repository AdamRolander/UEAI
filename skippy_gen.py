from AugmentationGenerator import AugmentationGenerator
import random
import os
import cv2

if __name__ == "__main__":
    input_folder = "extracted_frames"
    output_folder = "augmented/skippy_test"
    os.makedirs(output_folder, exist_ok=True)

    num_frames = 25
    num_aug_per_frame = 3

    for i in range(num_frames):
        frame_path = os.path.join(input_folder, f"frame_{i:04d}.png")
        skippy_Gen = AugmentationGenerator(path_to_image=frame_path)

        for j in range(num_aug_per_frame):
            disp_x = random.randint(-500, 500)
            disp_y = random.randint(-500, 500)
            zoom_factor = random.uniform(0.35, 1.35)

            augmented_image = skippy_Gen.apply_augmentation(disp_x, disp_y, zoom_factor)

            output_path = os.path.join(output_folder, f"frame_{i:03d}_aug_{j}.png")
            cv2.imwrite(output_path, augmented_image)

    print("Augmented images saved successfully!")