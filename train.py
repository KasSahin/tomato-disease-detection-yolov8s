import os  # iştetim sistemi dosya yollarını birleştirmek ve klasör yönetimini yapmak için kullanılır windows linux,macos uyumluluğu    
import torch # ekran kartı cuda kontrolü
from ultralytics import YOLO 

# DONANIM KONTROLÜ 
def validate_enviroment()-> int | str:
    print("="*50)
    print("[1/4] DONANIM VE ORTAM KONTROLÜ")
    print("="*50)

    pytorch_version=torch.__version__
    cuda_available = torch.cuda.is_available()

    print(f"Pytorch Sürümü:{pytorch_version}")
    print(f"CUDA AKTİF:{cuda_available}")

    if cuda_available:
        gpu_name=torch.cuda.get_device_name(0)
        print(f"GPU Aygıtı:{gpu_name}")
        device=0
    else:
        print("UYARI: GPU TESPİT EDİLEMEDİ , EĞİTİM CPU ÜZERİNDEN YÜRÜTÜLECEK")
        device='cpu'
    print("-"*50)
    return device

# TRAİN MODEL EĞİTİMİ 
def train_model(device:int|str,project_name:str,experiment_name:str) ->str:
    model_weights = r"C:/Users/memoc/OneDrive/Masaüstü/runs/detect/tarim_kurumsal_mlops/domates_hastalik_v1/weights/last.pt"
    model=YOLO(model_weights)

    model.train(
                  data='dataset/data.yaml',
                  epochs=100,
                  imgsz=640,
                  batch=16,
                  device=device,
                  
                  # OPTİMİZASYON
                  optimizer='AdamW',
                  lr0=0.001,
                  lrf=0.01,
                  weight_decay=0.0005,
                  warmup_epochs=3.0,
                  amp=True,
        
                  # VERİ ARTIRMA 
                  mosaic= 1.0,
                  mixup=0.15,
                  degrees=20.0,
                  translate=0.1,
                  scale=0.6,
                  shear=2.0,
                  perspective=0.0001,
                  flipud=0.5,
                  hsv_h=0.015,
                  hsv_s=0.7,
                  hsv_v=0.4,
        
                  # DURDURMA VE KAYIT 
        
                  patience=25,
                  save=True,
                  save_period=10,
                  val=True,
                  plots=True,
                  project=project_name,
                  name=experiment_name
           
    )

       
    best_model_path=os.path.join(project_name,experiment_name,'weights','best.pt')
    return best_model_path    
# MODEL DOĞRULAMA TEST
def evaluate_model(model_path:str):
    print("\n"+"="*50)
    print("[3/4] model doğrulama metrik raporlama")
    print("="*50)

    model=YOLO(model_path)
    val_metrics=model.val()

    print(f"mAP50 (Genel Başarı Oranı)       : {val_metrics.box.map50 * 100:.2f}%")
    print(f"mAP50-95 (Yüksek Hassasiyet Oranı) : {val_metrics.box.map * 100:.2f}%")
    print("-" * 50)

# ONNX DÖNÜŞÜM VE RAPORLAMA 
def export_to_onnx(model_path:str):
    print('[4/4] edge aı (RASPBERRY PI) PNNX EXPORT')
    print('='*50)

    production_model=YOLO(model_path)
    export_path=production_model.export(
        format='onnx',
        imgsz=640,
        simplify=True,
        dynamic=False
    )
    print(f"[BAŞARILI]:{export_path}")


def main():
    torch.backends.cudnn.benchmark=True

    PROJECT_NAME="tarim_kurumsal_mlops"
    EXPERIMENT_NAME="domates_hastalik_v1"

    device=validate_enviroment()
    best_weights=train_model(device,PROJECT_NAME,EXPERIMENT_NAME)
    evaluate_model(best_weights)
    export_to_onnx(best_weights)
if __name__=='__main__':
    main()        




  









        
             


             