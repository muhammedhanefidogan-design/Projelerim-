import cv2
import time
import os
import streamlit as st
from ultralytics import YOLO
from datetime import datetime
import winsound
import smtplib
import ssl
from email.message import EmailMessage
import threading
import requests 
from collections import deque

# --- TEMEL AYARLAR ---
MODEL_YOLU = "best.pt"
KAYIT_KLASORU = "kaza_kayitlari"
GECMIS_SANIYE = 5  # Kazadan kaç saniye öncesi hafızada tutulsun?
FPS_TAHMINI = 20   # Kameranın ortalama FPS değeri
BUFFER_BOYUTU = GECMIS_SANIYE * FPS_TAHMINI
ONAY_SAYISI = 2    # Kaza tespitinin kararlı olması için gereken kare sayısı

# Klasör yoksa oluştur
if not os.path.exists(KAYIT_KLASORU):
    os.makedirs(KAYIT_KLASORU)

st.set_page_config(page_title="Trafik Kaza Tespit & Kara Kutu", page_icon="🚨", layout="wide")

# --- HAFIZA SİSTEMİ (Kara Kutu) ---
if 'buffer' not in st.session_state:
    st.session_state.buffer = deque(maxlen=BUFFER_BOYUTU)

# --- ARKA PLAN MAİL İŞLEMCİSİ ---
def mail_islemci(gonderen, sifre, alici, foto_yolu):
    try:
        # Şifredeki boşlukları temizle (Örn: "abcd efgh" -> "abcdefgh")
        sifre = sifre.replace(" ", "")
        
        # Konumu çek
        try:
            ip = requests.get('https://api.ipify.org', timeout=5).text 
            loc = requests.get(f'http://ip-api.com/json/{ip}', timeout=5).json()
            konum_str = f"{loc.get('city')}, {loc.get('country')}"
        except:
            konum_str = "Konum bilgisi alınamadı."

        msg = EmailMessage()
        msg.set_content(f"🚨 ACİL DURUM: Kaza Tespit Edildi!\n\n📍 KONUM: {konum_str}\n⏰ ZAMAN: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\nLütfen ekteki kaza anı fotoğrafını ve kayıtları kontrol edin.")
        msg['Subject'] = '🚨 TRAFİK KAZA BİLDİRİMİ'
        msg['From'] = gonderen
        msg['To'] = alici

        if foto_yolu and os.path.exists(foto_yolu):
            with open(foto_yolu, 'rb') as f:
                msg.add_attachment(f.read(), maintype='image', subtype='jpeg', filename="kaza_ani.jpg")

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(gonderen, sifre)
            smtp.send_message(msg)
        print("✅ Mail başarıyla gönderildi!")
    except Exception as e:
        print(f"❌ Mail gönderim hatası: {e}")

# --- ARAYÜZ ---
st.title("🚦 Akıllı Trafik Kaza Tespit Sistemi")

st.sidebar.header("Sistem Kontrolü")
sistem_acik = st.sidebar.checkbox("Sistemi Başlat", value=False)
conf_threshold = st.sidebar.slider("Yapay Zeka Hassasiyeti (Conf)", 0.20, 0.95, 0.40)

st.sidebar.markdown("---")
st.sidebar.header("Kamera Ayarları")
# Varsayılan olarak boş bıraktım, sen video ismini veya IP'yi buraya yazacaksın
ip_kamera_url = st.sidebar.text_input("IP Webcam Adresi / Video Adı (örn: test.mp4):", "test.mp4")

mail_alici = st.sidebar.text_input("Bildirim Maili", "alici_mail@gmail.com")

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_model():
    return YOLO(MODEL_YOLU)

model = load_model()

# --- ANA DÖNGÜ ---
if sistem_acik:
    # Video Kaynağını Belirle
    try:
        if ip_kamera_url.isdigit(): # Eğer sadece sayı girildiyse (0, 1 gibi)
            video_kaynagi = int(ip_kamera_url)
        else:
            video_kaynagi = ip_kamera_url # URL veya dosya adıysa string kalır
    except:
        video_kaynagi = 0
        
    cap = cv2.VideoCapture(video_kaynagi)
    
    # Değişkenler
    kaza_sayisi = 0
    kayit_modu = False
    video_writer = None
    consecutive_frames = 0
    kayit_bitis_zamani = 0 # Soğuma zamanlayıcısı
    
    col1, col2 = st.columns([4, 1])
    with col1:
        frame_placeholder = st.empty()
    with col2:
        durum_text = st.empty()
        kaza_metric = st.metric("Toplam Kaza", 0)

    while cap.isOpened() and sistem_acik:
        ret, frame = cap.read()
        if not ret:
            st.warning("Video bitti veya görüntü alınamıyor.")
            break

        # 1. Her kareyi hafızaya (Buffer) ekle
        st.session_state.buffer.append(frame.copy())

        # 2. Yapay Zeka Tespiti (Resmi küçülterek hızlandırıyoruz: imgsz=480)
        results = model.predict(frame, conf=conf_threshold, verbose=False, imgsz=480)
        
        kaza_tespit_edildi = False
        tespit_edilen_siniflar = [] # Ekranda ne görüyor merak edersen diye

        for box in results[0].boxes:
            cls_name = results[0].names[int(box.cls[0])]
            tespit_edilen_siniflar.append(cls_name)

            # --- DİKKAT: TEST İÇİN BURAYA 'car' EKLENEBİLİR ---
            # Gerçek kaza için: ["severe", "moderate", "accident", "crash"]
            # Test için (Araba görünce ötsün): ["car", "truck", "bus", "severe", "moderate"]
            if cls_name in ["severe", "moderate", "accident", "crash"]: 
                kaza_tespit_edildi = True
                break

        # 3. Kaza Algılama Mantığı (SPAM KORUMALI)
        simdiki_zaman = time.time()

        # Eğer şu an kayıt yapmıyorsak VE son kaydın üzerinden 5 saniye geçtiyse (Cool-down)
        if not kayit_modu and (simdiki_zaman > kayit_bitis_zamani):
            if kaza_tespit_edildi:
                consecutive_frames += 1
            else:
                consecutive_frames = 0
            
            # Kaza Kesinleştiğinde (Start Action)
            if consecutive_frames >= ONAY_SAYISI:
                durum_text.error("🚨 KAZA ALGILANDI! (Kayıt Başladı)")
                
                kayit_modu = True
                kaza_sayisi += 1
                kaza_metric.metric("Toplam Kaza", kaza_sayisi)
                
                zaman_damgasi = datetime.now().strftime('%Y%m%d_%H%M%S')
                foto_yolu = f"{KAYIT_KLASORU}/kaza_{zaman_damgasi}.jpg"
                video_yolu = f"{KAYIT_KLASORU}/olay_ani_{zaman_damgasi}.avi"
                
                # Fotoğrafı kaydet
                cv2.imwrite(foto_yolu, frame)
                
                # Mail gönder (Senin bilgilerinle güncelledim)
                threading.Thread(target=mail_islemci, args=("muhammedhanefidogan493@gmail.com", "wqyxbvzdxpzctvnl", mail_alici, foto_yolu)).start()
                
                # Video kayıtçısını başlat
                h, w, _ = frame.shape
                video_writer = cv2.VideoWriter(video_yolu, cv2.VideoWriter_fourcc(*'XVID'), 20, (w, h))
                
                # Geçmişi (Buffer) videoya yaz
                for past_frame in st.session_state.buffer:
                    video_writer.write(past_frame)
                
                # Sesli uyarı
                try: winsound.Beep(1000, 500)
                except: pass

                # Kayıt ne zaman bitecek? (Şu an + 5 saniye sonra)
                kayit_bitis_zamani = simdiki_zaman + 5 

        # 4. Kayıt İşlemi (Devam Eden Kayıt)
        if kayit_modu:
            if video_writer:
                video_writer.write(frame)
            
            # Süre dolduysa kaydı bitir
            if simdiki_zaman > kayit_bitis_zamani:
                kayit_modu = False
                consecutive_frames = 0 # Sayacı sıfırla
                if video_writer:
                    video_writer.release()
                    video_writer = None
                
                st.toast("✅ Olay kaydedildi ve mail gönderildi. Sistem beklemede...")
                # Bir sonraki kayıt için sisteme 3 saniye dinlenme süresi ver (Spam engelleme)
                kayit_bitis_zamani = simdiki_zaman + 3 

        # 5. Ekrana Basma
        if not kaza_tespit_edildi and not kayit_modu: 
            durum_text.success("Yol Güvenli ✅")
            
        ann_frame = results[0].plot()
        frame_rgb = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
        # use_container_width=True güncel Streamlit komutudur
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    if video_writer: video_writer.release()
else:
    st.warning("Sistemi başlatmak için soldaki kutucuğu işaretleyin.")