# Akbank_DerinOgrenme_BootCamp
Beyin MR Görüntülerinden Tümör Tespiti
Bu depo, TensorFlow ve EfficientNetB0 kullanarak MR görüntülerinden beyin tümörü tespiti yapan bir derin öğrenme projesi içermektedir. Model, MR taramalarını tümör içeren ("yes") veya içermeyen ("no") şeklinde sınıflandırır.
📋 Proje Genel Bakış
Bu proje, Global AI Hub işbirliğiyle Akbank Bootcamp kapsamında geliştirilmiştir. Amaç, MR taramalarından beyin tümörlerinin erken teşhisine yardımcı olacak doğru ve güvenilir bir derin öğrenme modeli oluşturmaktır.

Temel Özellikler:
Sınıf dengesizliğini handle etmek için veri ön işleme ve artırma

EfficientNetB0 mimarisi kullanılarak transfer öğrenme

Çoklu metriklerle kapsamlı model değerlendirme

Yeni görüntüler üzerinde tahmin yapmak için kullanıcı dostu arayüz

📊 Veri Seti
Bu projede kullanılan veri seti Kaggle'dan alınan "Brain MRI Images for Brain Tumor Detection" veri setidir:

155 adet beyin tümörlü MR görüntüsü ("yes" sınıfı)

98 adet beyin tümörsüz MR görüntüsü ("no" sınıfı)

Veri Seti Kaynağı: Kaggle - Beyin MR Görüntüleri için Tümör Tespiti

🏗️ Metodoloji
1. Veri Ön İşleme
Veri sızıntısı sorunları düzeltildi

Veri seti uygun şekilde bölündü (Eğitim/Doğrulama/Test: %70/%15/%15)

Dengesiz veri için sınıf ağırlıkları uygulandı

2. Veri Artırma (Data Augmentation)
Döndürme, genişlik/yükseklik kaydırma, yakınlaştırma

Yatay çevirme ve parlaklık ayarlama

EfficientNet ön işleme fonksiyonu kullanıldı

3. Model Mimarisi
Temel Model: EfficientNetB0 (ImageNet üzerinde ön eğitimli)

Özel Katmanlar: GlobalAveragePooling2D, BatchNormalization, Dropout, Dense

Çıktı: İkili sınıflandırma için Sigmoid aktivasyonu

4. Eğitim Stratejisi
Optimizer: Adam (öğrenme oranı 1e-4)

Kayıp Fonksiyonu: Binary Crossentropy

Callback'ler: EarlyStopping ve ReduceLROnPlateau

Epoch: 20 (erken durdurma ile)

📈 Sonuçlar
Model doğrulama setinde aşağıdaki performans metriklerini elde etmiştir:
Sınıflandırma Raporu (Precision, Recall, F1-Score):
              precision    recall  f1-score   support

          no       0.81      0.87      0.84        15
         yes       0.91      0.87      0.89        23

    accuracy                           0.87        38
   macro avg       0.86      0.87      0.86        38
weighted avg       0.87      0.87      0.87        38

Projeyi Çalıştırma
Depoyu klonlayın

Kaggle API anahtarınızı yükleyin (kaggle.json)

Veri setini indirmek ve modeli eğitmek için notebook'u çalıştırın

Eğitilmiş modeli yeni MR görüntüleri üzerinde tahmin yapmak için kullanın

Tahmin Yapma
Proje, yeni MR görüntüleri yükleyip tahmin almak için fonksiyonellik içerir:

python
uploaded = files.upload()

# Model otomatik olarak görüntüyü ön işler ve sınıflandırır
🔧 Teknik Uygulama
Veri Generator'ları
Veri sızıntısını önlemek için uygun şekilde yapılandırıldı

Batch boyutu: 32

Görüntü boyutu: 224×224 piksel

Eğitim, doğrulama ve test için ayrı generator'lar

Model Konfigürasyonu
python
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)
🎯 Gelecek Geliştirmeler
Kısa Vadeli Hedefler:
Daha sofistike veri artırma teknikleri uygulama

Farklı mimarilerle denemeler (ResNet, DenseNet)

Daha sağlam değerlendirme için çapraz doğrulama ekleme

Uzun Vadeli Vizyon:
Kolay erişim için web uygulaması geliştirme

Gerçek zamanlı tahmin yetenekleri uygulama

Çok sınıflı sınıflandırmaya genişletme (farklı tümör tipleri)

Tıbbi görüntüleme sistemleriyle entegrasyon

📚 Referanslar ve Bağlantılar
Proje Bağlantıları:
Kaggle Veri Seti: Beyin MR Görüntüleri için Tümör Tespiti

Kaggle Notebook'um: Brain MR - Akbank Bootcamp

Teknik Referanslar:
EfficientNet: Konvolüsyonel Sinir Ağları için Model Ölçeklendirmeyi Yeniden Düşünmek (ICML 2019)

TensorFlow Dokümantasyonu

Keras Ön İşleme Dokümantasyonu

🤝 Katkıda Bulunma
Bu proje iyileştirmeler ve işbirlikleri için açıktır. Her türlü geliştirme için depoyu fork'layıp pull request göndermekten çekinmeyin.

📄 Lisans
Bu proje, Global AI Hub ile Akbank Bootcamp kapsamında eğitim amaçlı olarak oluşturulmuştur.

