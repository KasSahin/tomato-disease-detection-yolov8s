from ultralytics import YOLO

def convert_to_onnx():
    # En iyi modelimizin tam yolu
    model_path = r"C:/Users/memoc/OneDrive/Masaüstü/runs/detect/tarim_kurumsal_mlops/domates_hastalik_v1/weights/best.pt"
    
    print(f"Model yükleniyor: {model_path}")
    model = YOLO(model_path)

    print("Model ONNX formatına dönüştürülüyor (simplify=True)...")
    # ONNX export işlemi (Raspberry Pi için optimize edilmiş sadeleştirme ile)
    export_path = model.export(
        format='onnx', 
        imgsz=640, 
        simplify=True,    
        dynamic=False     
    )
    
    print(f"\n[BAŞARILI] Üretime hazır ONNX modeli şu konuma kaydedildi:\n{export_path}")

if __name__ == "__main__":
    convert_to_onnx()