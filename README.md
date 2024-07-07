# PENJELASAN PADA CODE
## 1. Mendeteksi Tepi dan Garis
### a). Mendeteksi Tepi
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import skimage
```
Pada bagian adalah sebuah library nya yang dimana `importcv2` adalah library yang digunakan untuk bisa membaca gambar dari file, melakukan operasi seperti deteksi objek, atau mengubah warna gambar, kemudian pada bagian `import numpy as np` ini untuk mengolah data gambar dalam bentuk array, kemudian `matplotlib.pyplot as plt` Digunakan untuk membuat grafik dan plot yang menampilkan data secara visual, seperti histogram atau gambar, kemudian `%matplotlib inline` ini memastikan bahwa semua plot yang dibuat dengan matplotlib akan ditampilkan di output notebook secara langsung setelah kode dieksekusi, dan `import skimage` ini untuk memproses dan menganalisis gambar. Dibangun di atas numpy, scipy, dan matplotlib untuk melakukan operasi seperti pemrosesan gambar, deteksi fitur, dan lainnya.
&nbsp;
```
image = cv2.imread('2.jpg')
```
pada bagian Perintah `image = cv2.imread('2.jpg')` digunakan untuk membaca gambar dengan nama file '2.jpg' menggunakan OpenCV yang hasilnya disimpan dalam variabel image sebagai representasi gambar dalam bentuk matriks NumPy, siap untuk digunakan dalam analisis atau manipulasi selanjutnya.
&nbsp;
```
cv2.imshow("Gambar Parking", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
Dibagian ini, pada bagian `cv2.imshow("Gambar Parking", image)` digunakan untuk menampilkan gambar yang telah dibaca (image) menggunakan OpenCV dengan judul "Gambar Parking", kemudian `cv2.waitKey(0)` menunggu tombol keyboard ditekan untuk menutup jendela gambar yang dimana angka 0 menunjukkan bahwa jendela akan tetap terbuka sampai tombol keyboard ditekan, dan `cv2.destroyAllWindows()` digunakan untuk menutup semua jendela OpenCV yang terbuka setelah pengguna menekan tombol keyboard.
&nbsp;
```
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(image, 100, 150)
```
Pada bagian `gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)` ini mengubah gambar image dari format warna BGR (Blue-Green-Red) menjadi skala abu-abu (grayscale). Hasilnya disimpan dalam variabel gray, dan `edges = cv2.Canny(image, 100, 150)` ini endeteksi tepi dalam gambar image menggunakan metode Canny edge detection. Angka 100 dan 150 adalah threshold yang digunakan untuk menentukan tepi yang hasilnya disimpan dalam variabel edges.
&nbsp;
```
cv2.imshow("Gambar Parkir", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
Pada bagian `cv2.imshow("Gambar Parkir", edges)` Menampilkan gambar hasil deteksi tepi yang disimpan dalam variabel edges yang jendelanya memiliki judul "Gambar Parkir",  kemudian `cv2.waitKey(0)` menunggu tombol keyboard ditekan untuk menutup jendela gambar yang dimana angka 0 menunjukkan bahwa jendela akan tetap terbuka sampai tombol keyboard ditekan, dan `cv2.destroyAllWindows()` digunakan untuk menutup semua jendela OpenCV yang terbuka setelah pengguna menekan tombol keyboard.
&nbsp;
```
fig, axs = plt.subplots(1,2, figsize =(10,10))
ax = axs.ravel()

ax[0].imshow(gray, cmap = "gray")
ax[0].set_title("Gambar Asli")

ax[1].imshow(edges, cmap = "gray")
ax[1].set_title("Gambar Setelah di Olah")
```
Dibagian ini, pada bagian `fig, axs = plt.subplots(1, 2, figsize=(10, 10))` membuat sebuah figure dengan 1 baris dan 2 kolom, yang berarti akan ada dua subplot dalam satu baris dan figsize=(10, 10) mengatur ukuran keseluruhan figure menjadi 10x10 inci, kemudian `ax = axs.ravel()` mengubah array 2D dari subplot menjadi array 1D yang memungkinkan kita untuk mengakses setiap subplot dengan indeks tunggal, kemudian `ax[0].imshow(gray, cmap="gray")` itu menampilkan gambar grayscale gray di subplot pertama (ax[0]) dengan colormap "gray", kemudian `ax[0].set_title("Gambar Asli")` menetapkan judul "Gambar Asli" untuk subplot pertama, `ax[1].imshow(edges, cmap="gray")` itu menampilkan gambar hasil deteksi tepi edges di subplot kedua (ax[1]) dengan colormap "gray", dan `ax[1].set_title("Gambar Setelah di Olah")` itu mnetapkan judul "Gambar Setelah di Olah" untuk subplot kedua. dan selanjutnya akan muncul deteksi tepi gambar tersebut yang sebelum dan sesudah diolah 
&nbsp;
### b). Mendeteksi Garis
```
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap = 20)
image_line = image.copy()
```
Pada bagian ini menggunakan transformasi Hough probabilistik (`HoughLinesP`) dari OpenCV untuk mendeteksi garis-garis pada gambar. edges adalah gambar tepi (`edge`) yang sudah dihasilkan sebelumnya. Parameters seperti `1, np.pi/180, 30, dan maxLineGap = 20` mengontrol deteksi garis dengan cara tertentu, seperti resolusi garis dan jarak antar titik.
&nbsp;
```
for line in lines :
    x1, y1, x2, y2 = line[0]
    cv2.line(image_line,(x1,y1),(x2,y2),(100,8, 255),1)
```
Pada bagian ini melakukan iterasi pada setiap garis yang dideteksi dalam variabel lines. Untuk setiap garis, nilai koordinat titik awal `(x1, y1)` dan titik akhir `(x2, y2)` diekstraksi dari line. Kemudian, fungsi `cv2.line()` digunakan untuk menggambar garis tersebut pada gambar `image_line` yang dimana `cv2.line(image_line, (x1, y1), (x2, y2), (100, 8, 255), 1)` ini menggambar garis dari titik `(x1, y1)` ke `(x2, y2)` pada gambar image_line, dan `(100, 8, 255)` itu Warna garis dalam format BGR (biru, hijau, merah). Misalnya, `(100, 8, 255)` mewakili warna ungu (biru + merah), dan `1` yakni Ketebalan garis dalam piksel.
&nbsp;
```
fig,axs = plt.subplots(1,2,figsize = (10,10))
ax = axs.ravel()

ax[0].imshow(gray, cmap = "gray")
ax[0].set_title("Gambar Asli")

ax[1].imshow(image_line, cmap = "gray")
ax[1].set_title("Gambar Setelah di Olah")
```
Pada bagian ini  membuat subplot dengan dua gambar (`ax[0] dan ax[1]`) dalam satu figur (`fig`) dengan ukuran 10x10 inci, yang dimana `ax[0].imshow(gray, cmap="gray")` ini menampilkan gambar gray dalam skala abu-abu, kemudian `ax[0].set_title("Gambar Asli")` memberikan judul "Gambar Asli" pada subplot pertama (`ax[0]`), kemudian `ax[1].imshow(image_line, cmap="gray")` ini menampilkan gambar `image_line` setelah dilakukan pemrosesan. dalam hal ini, deteksi garis dalam skala abu-abu, serta `ax[1].set_title("Gambar Setelah di Olah")` memberikan judul "Gambar Setelah di Olah" pada subplot kedua (`ax[1]`). dan selanjutnya akan muncul deteksi garis gambar tersebut yang sebelum dan sesudah diolah.
&nbsp;
## 2. Ekstraksi fitur menggunakan skimage (scikit-image) RGB to HSV
```
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops
```
Pada bagian ini adalah sebuah library mengimpor `matplotlib.pyplot` untuk menampilkan gambar, `skimage.io,color`,`graycomatrix`,`graycoprops` untuk membaca gambar dan mengkonversi warna dalam ekstraksi fitur.
&nbsp;
```
image_path = ('2.jpg')
image_rgb = io.imread(image_path)
```
Pada bagian ini Kode `image_path = ('2.jpg')` dan `image_rgb = io.imread(image_path)` digunakan untuk memuat gambar dengan nama file `2.jpg` menggunakan pustaka `skimage.io`.
&nbsp;
```
image_hsv = color.rgb2hsv(image_rgb)
```
Pada bagian ini Kode `image_hsv = color.rgb2hsv(image_rgb)` digunakan untuk mengonversi gambar dari representasi warna RGB (Red-Green-Blue) ke representasi warna HSV (Hue-Saturation-Value) menggunakan pustaka `skimage.color`.
&nbsp;
```
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_rgb) 
axes[0].set_title('Original RGB Image') 
axes[0].axis('off')

axes[1].imshow(image_hsv) 
axes[1].set_title('HSV Image')
axes[1].axis('off')

plt.show()
```
Pada bagian ini Kode `fig, axes = plt.subplots(1, 2, figsize=(12, 6))` digunakan untuk membuat jendela gambar dengan dua subplot yang menampilkan gambar asli dalam format RGB dan gambar yang telah dikonversi ke format HSV. Pada subplot pertama `(axes[0])`, gambar dalam format RGB `(image_rgb)` ditampilkan menggunakan `axes[0].imshow(image_rgb)`. Subplot ini diberi judul "Original RGB Image" dengan menggunakan `axes[0].set_title('Original RGB Image')` dan sumbu koordinatnya dihilangkan dengan `axes[0].axis('off')`, kemudian pada subplot kedua `(axes[1])`, gambar yang telah dikonversi ke format HSV (`image_hsv`) ditampilkan menggunakan `axes[1].imshow(image_hsv)`. Subplot ini diberi judul "HSV Image" dengan menggunakan `axes[1].set_title('HSV Image')` dan sumbu koordinatnya dihilangkan dengan `axes[1].axis('off')`, serta pada bagian `plt.show()` digunakan untuk menampilkan jendela gambar lengkap dengan kedua subplot yang menampilkan perbandingan antara gambar asli dalam format RGB dan gambar yang telah dikonversi ke format HSV.
&nbsp;
```
mean_h = np.mean(image_hsv[:, :, 0])
mean_s = np.mean(image_hsv[:, :, 1])
mean_v = np.mean(image_hsv[:, :, 2])
```
Pada bagian  `mean_h = np.mean(image_hsv[:, :, 0])`, `mean_s = np.mean(image_hsv[:, :, 1])`, dan `mean_v = np.mean(image_hsv[:, :, 2])` digunakan untuk menghitung nilai rata-rata dari setiap kanal dalam citra HSV. Dalam citra HSV, terdapat tiga kanal yang masing-masing mewakili Hue (H), Saturation (S), dan Value (V). Dengan menggunakan np.mean, kita dapat menghitung rata-rata nilai dari setiap kanal tersebut, di mana image_hsv[:, :, 0] merujuk pada kanal H, image_hsv[:, :, 1] merujuk pada kanal S, dan image_hsv[:, :, 2] merujuk pada kanal V. Nilai rata-rata ini kemudian disimpan dalam variabel mean_h, mean_s, dan mean_v untuk masing-masing kanal.
&nbsp;
```
print("Mean H:", mean_h)
print("Mean S:", mean_s)
print("Mean V:", mean_v)
```
Pada bagian print("Mean H:", mean_h), print("Mean S:", mean_s), dan print("Mean V:", mean_v) digunakan untuk mencetak nilai rata-rata dari setiap kanal dalam citra HSV. Setelah menghitung rata-rata nilai untuk kanal Hue (H), Saturation (S), dan Value (V), nilai-nilai tersebut disimpan dalam variabel mean_h, mean_s, dan mean_v. Baris kode ini akan menampilkan hasil perhitungan tersebut di konsol atau output, sehingga kita dapat melihat nilai rata-rata untuk setiap kanal.
&nbsp;
```
image_gray = (image_hsv[:, :, 2] * 255).astype(np.uint8)
```
Pada bagian ` image_gray = (image_hsv[:, :, 2] * 255).astype(np.uint8) ` digunakan untuk mengubah kanal V (Value) dari citra HSV menjadi citra grayscale dalam format 8-bit.
&nbsp;
```
glcm = graycomatrix(image_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
```
Pada bagian ` glcm = greycomatrix(image_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)` digunakan untuk menghitung Grey Level Co-occurrence Matrix (GLCM) dari citra grayscale.
&nbsp;
```
contrast = graycoprops(glcm, 'contrast')[0, 0]
dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
energy = graycoprops(glcm, 'energy')[0, 0]
correlation = graycoprops(glcm, 'correlation')[0, 0]
```
Pada bagian ini digunakan untuk menghitung beberapa properti dari matriks Grey Level Co-occurrence Matrix (GLCM) yang sudah dihitung sebelumnya, yang dimana:
- contrast = greycoprops(glcm, 'contrast')[0, 0]: Menghitung kontras dari GLCM. Kontras mengukur intensitas lokal dan variasi dalam citra, dengan nilai tinggi menunjukkan perbedaan intensitas yang besar di antara piksel yang bersebelahan.
- dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]: Menghitung dissimilaritas dari GLCM. Dissimilarity mengukur ketidaksamaan antara pasangan piksel, dengan nilai tinggi menunjukkan variasi yang besar.
- homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]: Menghitung homogenitas dari GLCM. Homogenitas mengukur kedekatan distribusi elemen dalam GLCM ke diagonal GLCM, dengan nilai tinggi menunjukkan distribusi yang lebih seragam.
- energy = greycoprops(glcm, 'energy')[0, 0]: Menghitung energi dari GLCM. Energi, atau uniformity, adalah jumlah kuadrat elemen GLCM dan mengukur kehalusan dalam citra, dengan nilai tinggi menunjukkan citra yang lebih halus.
- correlation = greycoprops(glcm, 'correlation')[0, 0]: Menghitung korelasi dari GLCM. Korelasi mengukur bagaimana piksel dalam citra berkorelasi satu sama lain, dengan nilai tinggi menunjukkan korelasi yang kuat.
&nbsp;
```
print("GLCM Contrast:", contrast)
print("GLCM Dissimilarity:", dissimilarity)
print("GLCM Homogeneity:", homogeneity)
print("GLCM Energy:", energy)
print("GLCM Correlation:", correlation)
```
Pada bagian ini digunakan untuk mencetak hasil perhitungan properti GLCM ke konsol atau output yang kemudian menampilkan nilai-nilai tersebut sehingga kita bisa melihat hasil dari ekstraksi fitur tekstur pada citra. 




