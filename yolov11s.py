from ultralytics import YOLO

MODEL = r"yolo11s.pt"
DATA_YAML = r"C:\Users\doong\Desktop\mengyuan\yolo\yolov8\swimmingpool_data.yaml"
EPOCHS = 300
IMGSZ = 1280
BATCH = 4

if __name__ == "__main__":
    
    model = YOLO(MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        workers=4,
        amp=True,
        project=r"C:\Users\doong\Desktop\mengyuan\yolo\runs\detect",
        name="swimmingpoolnew6",
        exist_ok=True,
        pretrained=True,

        fliplr=0.5,
        flipud=0.0,
        degrees=10,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3

    )

    print("最佳權重路徑：", results.save_dir)