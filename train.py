from ultralytics import YOLO

model = YOLO("yolo11n.pt")


if __name__ == '__main__':
    model.train(
        data="dataset_split/dataset.yaml",
        epochs=150,
        patience=100,
        device=0,
        lrf=0.001,
        imgsz=256,
        hsv_s=0.8,
        degrees=45,
        translate=0.1,
        shear=0.1,
        flipud=0.5,
        fliplr=0.5,
    )
