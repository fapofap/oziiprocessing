from ultralytics import FastSAM

# Create a FastSAM model
model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt

# Validate the model
results = model.val(data="coco8-seg.yaml")