import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib notebook

image = cv2.imread('IMG_3.jpg', 0)

image = cv2.resize(image, (240, 180))

clickarr = np.zeros(image.shape)

dft = cv2.dft(np.float32(image), flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

dft_ishift = np.fft.ifftshift(dft_shift)
img_back = cv2.idft(dft_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

def Addclick(x, y, click):
    a = click
    a[y, x] = 1
    return a

def Wave(x, y):
    b = np.zeros(image.shape)
    b[y, x] = 1
    wave = cv2.idft(b)
    return wave

def Synthesis(x, y, a):
    c = np.zeros(dft_shift.shape)
    c[:,:,0] = a
    c[:,:,1] = a
    copy = dft_shift*c
    copy = np.fft.ifftshift(copy)
    copy = cv2.idft(copy)
    d = cv2.magnitude(copy[:,:,0], copy[:,:,1])
    return d

def Clickprocess(event):
    global clickarr, X, Y
    if event.button == 1:
        X = int(round(event.xdata))
        Y = int(round(event.ydata))
        ax7.plot(X, Y, marker = '.', markersize = '1')
        clickarr = Addclick(X, Y, clickarr)

def Calculate(event):
    global clickarr, X, Y
    wave = Wave(X, Y)
    ax5.imshow(wave, cmap = 'gray')
    reimage = Synthesis(X, Y, clickarr)
    reimage = np.float32(reimage)
    ax4.imshow(reimage, cmap = 'gray')
    plt.draw()
    
def Clickq(event):
    if event.key == 'q':
        sys.exit()

figure = plt.figure(figsize = (9, 4))
ax1 = figure.add_subplot(2, 3, 1)
ax1.imshow(image, cmap = 'gray')
ax1.set_title('Input Image')
ax1.set_xticks([]), ax1.set_yticks([])

ax2 = figure.add_subplot(2, 3, 2)
ax2.imshow(magnitude_spectrum, cmap = 'gray')
ax2.set_title('Magnitude')
ax2.set_xticks([]), ax2.set_yticks([])

ax3 = figure.add_subplot(2, 3, 3)
ax3.imshow(img_back, cmap = 'gray')
ax3.set_title('Ifft')
ax3.set_xticks([]), ax3.set_yticks([])

ax4 = figure.add_subplot(2, 3, 4)
ax5 = figure.add_subplot(2, 3, 5)
figure2 = plt.figure(figsize = (8, 4))
ax7 = figure2.add_subplot(1, 1, 1)
ax7.imshow(magnitude_spectrum, cmap = 'gray')
plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.5, hspace = 0)

figure2.canvas.mpl_connect('button_press_event', Clickprocess)
figure2.canvas.mpl_connect('motion_notify_event', Clickprocess)
figure2.canvas.mpl_connect('button_release_event', Calculate)
figure2.canvas.mpl_connect('key_press_event', Clickq)


[プログラム説明]
1-4 環境設定

6 画像を読み込む
8 画像のサイズを変更
10 配列clickarrの初期化
12-14 フーリエ変換
16-18 逆フーリエ変換
20-23 クリックした座標を1とする関数
25-29 sin波を抽出する関数
31-39 sin波を合成する関数
41-47 左クリックもしくはドラッグが確認された時の実行を示す関数
49-56 計算を実行する関数
58-60 実行の終了を行う関数
62-83 表示図に関する設定
85-88 マウスの動きに対応する関数

[実装方法]
Jupyter notebook　で上記のコードを実装する。その際、Macでは、　control+Enter　が実行コマンドとなる。

[依存ライブラリとver.]
numpyはver.3.0.3で、matplotlibはver.1.16.2である。

[参考サイト]
https://stats.biopapyrus.jp/python/subplot.html
http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
https://qiita.com/HajimeKawahara/items/abc24fa2216009523656
http://yura2.hateblo.jp/entry/2017/09/12/matplotlib_%E3%81%A7%E3%83%97%E3%83%AD%E3%83%83%E3%83%88%E4%B8%8A%E3%81%AE%E7%82%B9%E3%82%92%E3%83%89%E3%83%A9%E3%83%83%E3%82%B0%E3%81%99%E3%82%8B%E4%BE%8B

