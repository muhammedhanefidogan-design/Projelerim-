import cv2
import winsound # Windows'un ses çıkarma aracı
from ultralytics import YOLO

# 1. Modeli yükle (Dosya yolunu gerekirse düzelt)
model = YOLO("best.pt")

print("Sistem devrede... Kaza aranıyor... (Çıkmak için ekrana tıkla ve 'q'ya bas)")

# 2. Canlı Tespiti Başlat
# stream=True: Videoyu kare kare işlememizi sağlar (önemli!)
# conf=0.50: Yüzde 50'den emin değilse ötmesin (hatayı azaltır)
results = model.predict(source="0", show=True, stream=True, conf=0.50)

for result in results:
    # O an ekranda görünen kutucukların isimlerini alalım
    # result.boxes.cls -> Tespit edilenlerin ID numaraları
    # result.names -> ID'lerin isim karşılığı (0: moderate, 1: severe gibi)
    
    detected_classes = result.boxes.cls.tolist() # Ekranda ne var? Listeye çevir.
    names = result.names

    kaza_var_mi = False
    
    # Ekranda tespit edilen her şeye tek tek bak
    for class_id in detected_classes:
        class_name = names[int(class_id)]
        
        # Eğer tespit edilen şey 'severe' veya 'moderate' ise alarmı tetikle
        if class_name == "severe" or class_name == "moderate":
            kaza_var_mi = True
            break # Bir tane bulsak yeter, döngüden çık

    if kaza_var_mi:
        print("🚨 DİKKAT! KAZA TESPİT EDİLDİ! 🚨")
        
        # BİİP SESİ ÇIKAR
        # İlk sayı: Frekans (Sesin inceliği, 1000 iyidir)
        # İkinci sayı: Süre (Milisaniye, 200ms kısa bip sesi)
        # Süreyi çok uzatırsan video donar, kısa tutmak iyidir.
        winsound.Beep(2500, 100) 

    # Çıkış işlemi (Video penceresindeyken 'q'ya basınca durur)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break