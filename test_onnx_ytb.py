import cv2
from ultralytics import YOLO
import yt_dlp

def get_youtube_stream_url(youtube_url: str) -> str:
    """
    YouTube video linkini hata almadan en uygun formatta akış adresine çevirir.
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]/best[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict['url']

def run_onnx_youtube_inference():
    # 1. ONNX Modelimizin Tam Yolu
    model_path = r"C:\Users\memoc\OneDrive\Masaüstü\runs\detect\tarim_kurumsal_mlops\domates_hastalik_v1\weights\best.onnx"
    
    print(f"ONNX Model yükleniyor: {model_path}")
    model = YOLO(model_path)

    # 2. Güncel ve Test Edilmiş YouTube Videosu
    youtube_video_url = "https://youtu.be/yd2oUEJriic?si=5CRea2MYdYhRQuH1"
    
    print("YouTube video akış adresi alınıyor...")
    stream_url = get_youtube_stream_url(youtube_video_url)
    
    # 3. OpenCV ile Video Akışını Başlatma
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("HATA: YouTube video akışı açılamadı!")
        return

    print("\n[BİLGİ] ONNX model ile YouTube testi başladı. Çıkmak için 'q' tuşuna basın.")

    # 4. Kare Kare İşleme Döngüsü
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video bitti veya akış sonlandı.")
            break

        # 5. Model ile Anlık Tahmin (Inference)
        results = model(frame, imgsz=640, conf=0.4)

        # 6. Sonuçları Görselleştirme
        annotated_frame = results[0].plot()

        # 7. Ekranda Gösterme
        cv2.imshow("Edge AI (ONNX) - Domates Hastalik Testi", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 8. Kaynakları Serbest Bırakma
    cap.release()
    cv2.destroyAllWindows()
    print("[BİLGİ] Test güvenli bir şekilde sonlandırıldı.")

if __name__ == "__main__":
    run_onnx_youtube_inference()
         

