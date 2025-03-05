from ultralytics import SAM

# Load a model
model = SAM("sam2.1_b.pt")

# Display model information (optional)
model.info()

# Run inference
model(source="cliffs4.mp4", show=True, save=True)