# Kalp Hastalığı Tahmini Projesi

# PROJEYİ GELİŞTİRENLER

`Muhammed Uğur Emre` 
`Selen Sarı` 
`Eren Çetin` 
`Doruk Alp Bucak` 

Bu proje, bir veri kümesi kullanarak kalp hastalığı tahminini gerçekleştirmek için çeşitli makine öğrenimi algoritmalarını (KNN, MLP ve Naive Bayes) uygular.
Veri kümesi hem kategorik hem de sayısal öznitelikler içerir ve bu özniteliklerin etkilerini analiz ederek algoritmaların başarımını kıyaslamayı amaçlar.

## Proje Adımları

### 1. Veri Kümesinin Seçimi

Proje, içinde hem sayısal hem de kategorik öznitelikler barındıran `heart_disease.csv` adlı veri kümesi üzerinde gerçekleştirilmiştir. Veri kümesi, bireylerin demografik, sağlık ve yaşam tarzı özelliklerini içerir ve kalp krizi geçirip geçirmediğine dair bir sınıf etiketi (`Heart_ stroke`) barındırır.

---

### 2. Özniteliklerin Tanıtımı

| **Öznitelik**       | **Açıklama**                                                                      |
| ------------------- | --------------------------------------------------------------------------------- |
| **Gender**          | Cinsiyet (0: Kadın, 1: Erkek).                                                    |
| **age**             | Yaş.                                                                              |
| **currentSmoker**   | Şu an sigara içip içmediği (0: Hayır, 1: Evet).                                   |
| **cigsPerDay**      | Günde içilen sigara sayısı.                                                       |
| **BPMeds**          | Tansiyon ilacı kullanımı (0: Hayır, 1: Evet).                                     |
| **prevalentStroke** | Daha önce felç geçirip geçirmediği (0: Hayır, 1: Evet).                           |
| **prevalentHyp**    | Daha önce hipertansiyon geçirip geçirmediği (0: Hayır, 1: Evet).                  |
| **diabetes**        | Diyabet hastası olup olmadığı (0: Hayır, 1: Evet).                                |
| **totChol**         | Toplam kolesterol seviyesi.                                                       |
| **sysBP**           | Sistolik tansiyon seviyesi.                                                       |
| **diaBP**           | Diyastolik tansiyon seviyesi.                                                     |
| **BMI**             | Vücut kitle indeksi.                                                              |
| **heartRate**       | Kalp atış hızı (bpm).                                                             |
| **glucose**         | Glikoz seviyesi.                                                                  |
| **Heart\_ stroke**  | Kalp krizi geçirip geçirmediği (0: Hayır, 1: Evet).                               |
| **education**       | Eğitim seviyesi (0: İlkokul, 1: Ortaokul, 2: Lise, 3: Üniversite, 4: Lisansüstü). |

---

### 3. Kategorik Özniteliklerin Dönüştürülmesi

Veri kümesindeki kategorik öznitelikler `LabelEncoder` ve `OneHotEncoder` kullanılarak sayısal değerlere dönüştürülmüştür:

- **Gender**, **prevalentStroke**, **Heart\_ stroke** öznitelikleri `LabelEncoder` ile dönüştürüldü.
- **education** özniteliği `OneHotEncoder` kullanılarak yeni sütunlara ayrıldı.

---

### 4. Özniteliklerin Normalize Edilmesi

Tüm sayısal öznitelikler, farklı ölçekleri dengelemek için `StandardScaler` ile normalize edildi. Bu adım, makine öğrenimi algoritmalarının daha iyi performans göstermesini sağlamıştır.

---

### 5. Makine Öğrenimi Algoritmaları

#### a. KNN

- KNN algoritması K=3, K=7 ve K=11 komşuluk değerleri için çalıştırılmıştır.
- Doğruluk (accuracy), kesinlik (precision), geri çağırma (recall) ve F1 skor değerleri hesaplanmıştır.

#### b. MLP

- Çok katmanlı algılayıcı (MLP) algoritması şu yapılandırmalarla çalıştırılmıştır:
  - 1 gizli katman (32 nöron),
  - 2 gizli katman (32’şer nöron),
  - 3 gizli katman (32’şer nöron).
- SMOTE yöntemi kullanılarak veri dengesizliği giderilmiş ve model performansı artırılmıştır.

#### c. Naive Bayes

- Naive Bayes algoritması varsayılan parametrelerle çalıştırılmıştır.

---

### 6. Sonuçların Analizi

- Algoritmaların doğruluk (accuracy), kesinlik (precision), geri çağırma (recall) ve F1 skor değerleri karşılaştırılmıştır.
- Elde edilen sonuçlar bir tablo halinde CSV dosyasına (`algorithm_results.csv`) kaydedilmiştir.

---

## Nasıl Çalıştırılır?

1. Gerekli bağımlılıkları yüklemek için:
   ```bash
   pip install -r requirements.txt
   ```
2. Ana Python dosyasını çalıştırın:
   ```bash
   python main.py
   ```
3. Algoritma sonuçları `algorithm_results.csv` dosyasına kaydedilecektir.

---

Teşekkürler. 😊
