import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Load model and preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load image
image_path = "SS1_depth.png"
image = Image.open(image_path)
draw = ImageDraw.Draw(image)

# Define classification labels
labels = ["ocean", "sky", "rock", "hills", "sunset", "sand"]
text_inputs = clip.tokenize(labels).to(device)

# Convert image to numpy for patching
img_np = np.array(image)
height, width, _ = img_np.shape

# Set patch size
patch_size = 100  # Adjust as needed
step = patch_size // 2

# Iterate over patches
for y in range(0, height - patch_size, step):
    for x in range(0, width - patch_size, step):
        patch = image.crop((x, y, x + patch_size, y + patch_size))
        patch_tensor = preprocess(patch).unsqueeze(0).to(device)
        
        with torch.no_grad():
            patch_features = model.encode_image(patch_tensor)
            text_features = model.encode_text(text_inputs)
            similarities = (patch_features @ text_features.T).softmax(dim=-1)

        # Assign highest confidence label
        best_label_idx = similarities.argmax().item()
        best_label = labels[best_label_idx]

        # Draw label on image
        draw.text((x + 5, y + 5), best_label, fill="red")  

# Show image with overlaid labels
plt.imshow(image)
plt.axis("off")
plt.show()