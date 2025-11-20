from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("./runs/detect/train24/weights/best.pt")

# Define path to the image file
source = "./dataset/images/train/00054_145.jpg"

# Run inference on the source
results = model(source)  # list of Results objects
results[0].show()