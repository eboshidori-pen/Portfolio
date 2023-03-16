# coding:utf-8

import dlib
from imutils import face_utils
import cv2
import imutils #OpenCVの補助
import numpy as np
from playsound import playsound
from IPython.display import Image, display
import math
import serial, time
#ser = serial.Serial("COM3", 115200, timeout=0.1) #timeout追加した
#not_used = ser.readline()


n = 0
m = 0
judge1 = 0
judge2 = 0
count = 0

# --------------------------------
# 1.顔ランドマーク検出の前準備
# --------------------------------
# 顔ランドマーク検出ツールの呼び出し
face_detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_predictor = dlib.shape_predictor(predictor_path)

# --------------------------------
# 2.画像から顔のランドマーク検出する関数
# --------------------------------
def face_landmark_find(img):
    # 顔検出
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(img_gry, 1)

    # 検出した全顔に対して処理
    for face in faces:
        # 顔のランドマーク検出
        landmark = face_predictor(img_gry, face)
        # 処理高速化のためランドマーク群をNumPy配列に変換(必須)
        landmark = face_utils.shape_to_np(landmark)

        

    return img

def ripcut(rip_points):
    img2 = img[rip_points[2][1]-10 : rip_points[9][1]+10, rip_points[0][0]-10 : rip_points[6][0]+10]
    cv2.imwrite("landrip.jpg",img2)

    return img2

def rip_hsv(img2, rip_points):
    if n == 0:
        black = np.zeros((750, 1000, 3))
        white_rip = cv2.fillConvexPoly(black, rip_points, (255, 255, 255))
        cv2.drawContours(white_rip, [rip_points], -1, color=(255,255,255), thickness=-1)
        bmask = white_rip[rip_points[2][1]-10: rip_points[9][1]+10, rip_points[0][0]-10 : rip_points[6][0]+10]
        white = cv2.imread("white.jpg")
        wcut = white[rip_points[1][1]-10 : rip_points[8][1]+10, rip_points[11][0]-10 : rip_points[5][0]+10]
        cv2.imwrite("bmask.jpg",bmask)
        cv2.imwrite("wcut.jpg",wcut)


    imgg = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    img3 = cv2.inRange(imgg, (7, 0, 0), (178, 255, 255))

    cv2.imwrite("hsvrip.jpg",img3)

def ripdst():
    # 画像を読み込む。
    fg_img = cv2.imread("wcut.jpg")

    # 背景画像を読み込む。
    bg_img = cv2.imread("hsvrip.jpg")
    mask = cv2.imread("bmask.jpg")

    # HSV に変換する。
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    # 2値化する。
    bin_img = ~cv2.inRange(hsv, (0, 0, 0), (0, 0, 0))

    # 輪郭抽出する。
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 面積が最大の輪郭を取得する
    contour = max(contours, key=lambda x: cv2.contourArea(x))

    # マスク画像を作成する。
    mask = np.zeros_like(bin_img)
    cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
    cv2.imwrite("mask.jpg",mask)

    # 幅、高さは前景画像と背景画像の共通部分をとる
    w = min(fg_img.shape[1], bg_img.shape[1])
    h = min(fg_img.shape[0], bg_img.shape[0])

    # 合成する領域
    fg_roi = fg_img[:h, :w]
    bg_roi = bg_img[:h, :w]

    # 合成する。
    dst = np.where(mask[:h, :w, np.newaxis] == 0, bg_roi, fg_roi)

    return dst

def ripcutend_ue():
    hikaku_0 = cv2.imread("hikaku_0.jpg")
    yRipHalf = rip_points[11][1] - (rip_points[1][1]-10)
    ripcutend_0 = hikaku_0[0 : yRipHalf, 0 : width]#上下左右
    cv2.imwrite("ripcutend_0.jpg",ripcutend_0)
    hikaku_1 = cv2.imread("rehikaku_1.jpg")
    ripcutend_1 = hikaku_1[0 : yRipHalf, 0 : width]#上下左右
    cv2.imwrite("ripcutend_1.jpg",ripcutend_1)
    du = different_Area()
    print(du)
    if du >= 25:
        judge1 = 1
    else:
        judge1 = 0
    return judge1

def ripcutend_sita():
    hikaku_0 = cv2.imread("hikaku_0.jpg")
    yRipHalf = rip_points[11][1] - (rip_points[1][1]-10)
    ripcutend_0 = hikaku_0[yRipHalf : height, 0 : width]#上下左右
    cv2.imwrite("ripcutend_0.jpg",ripcutend_0)
    hikaku_1 = cv2.imread("rehikaku_1.jpg")
    ripcutend_1 = hikaku_1[yRipHalf : height, 0 : width]#上下左右
    cv2.imwrite("ripcutend_1.jpg",ripcutend_1)
    ds = different_Area()
    if ds >= 25:
        judge2 = 1
    else:
        judge2 = 0
    return judge2


def different_Area():
    hikaku_0 = cv2.imread("ripcutend_0.jpg")
    hikaku_1 = cv2.imread("ripcutend_1.jpg")
    image_size = hikaku_0.size
    samePixels = np.count_nonzero(hikaku_0 == hikaku_1)
    differentPixels = image_size - samePixels
 
    sameAreaRatio = (samePixels/image_size)*100#[%]
    differentAreaRatio = (differentPixels/image_size)*100#[%]
 
    print("Same Area [%] : ", sameAreaRatio)
    print("Different Area [%] : ", differentAreaRatio)

    return differentAreaRatio

def judge(j1, j2):
    if j1 == 1:
        playsound("uekuchi.mp3")

    if j2 == 1:
        playsound("sitakuchi.mp3")

    if j1 == 1 or j2 == 1:
        playsound("hamidashiari.mp3")
    else:
        playsound("hamidashinashi.mp3")

def mayucutl(mul, msl, mhx2, mml):
    lmcut = img[mul : msl, mhx2 : mml]#右　上下左右
    cv2.imwrite("mayucutl.jpg",lmcut)

    return lmcut

def mayucutr(mur, msr, mmr, mhx2):
    face = cv2.imread("face.jpg")
    rmcut = face[mur : msr, mmr : mhx2]#右　上下左右
    cv2.imwrite("mayucutr.jpg",rmcut)

    return rmcut

#傾きを求める
def tyokusenn(x1, y1, x2, y2, point_x, point_y):
    kx = x2 - x1
    ky = y2 - y1
    k = ky / kx
    c = y1 - (k * x1)
    if mayu_distance(k, c, point_x, point_y) >= 30:
        return mayu_ryouiki(k, c, point_x, point_y)
    return 0


#点と直線の距離
def mayu_distance(a, c, point_x, point_y): #直線ax+by+c=0 点(x0,y0)
    numer = abs(a*point_x + -1*point_y + c) #分子
    denom = math.sqrt(pow(a,2)+pow(-1,2)) #分母
    return numer/denom #計算結果

def mayu_ryouiki(k, c, point_x, point_y):
    tyokusenn_y = k * point_x + c
    ryouiki = 0
    if (tyokusenn_y < point_y):
        ryouiki = 1
    else:
        ryouki = -1
    return ryouiki
    

# --------------------------------
# 3.カメラ画像の取得
# --------------------------------
# カメラの指定(適切な引数を渡す)
cap = cv2.VideoCapture(0)

# カメラ画像の表示
while(True):
    ret, img = cap.read()

     # 顔のランドマーク検出(2.の関数呼び出し)
    img = face_landmark_find(img)


    img = imutils.resize(img, width=1000) #frameの画像の表示サイズを整える
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gray scaleに変換する
    rects = face_detector(gray, 0) #grayから顔を検出
    image_points = None

    #val_decoded = 0
    if (count%100 == 0):
        #ser.write(chr('N'))
        #val_arduino = ser.readline()
        # val_decoded = float(repr(val_arduino.decode())[1:-5])
        #val_decoded = val_arduino
        #print(val_decoded)

    for rect in rects:
        shape = face_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

          # ランドマーク描画
        # for (i, (x, y)) in enumerate(shape):
        #     cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

        image_points = np.array([
                tuple(shape[30]),#鼻頭
                tuple(shape[21]),
                tuple(shape[22]),
                tuple(shape[39]),
                tuple(shape[42]),
                tuple(shape[31]),
                tuple(shape[35]),
                tuple(shape[48]),
                tuple(shape[54]),
                tuple(shape[57]),
                tuple(shape[8]),
                ],dtype='double')

        if len(rects) > 0:
            cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 2
            model_points = np.array([
                    (0.0,0.0,0.0), # 30
                    (-30.0,-125.0,-30.0), # 21
                    (30.0,-125.0,-30.0), # 22
                    (-60.0,-70.0,-60.0), # 39
                    (60.0,-70.0,-60.0), # 42
                    (-40.0,40.0,-50.0), # 31
                    (40.0,40.0,-50.0), # 35
                    (-70.0,130.0,-100.0), # 48
                    (70.0,130.0,-100.0), # 54
                    (0.0,158.0,-10.0), # 57
                    (0.0,250.0,-50.0) # 8
                ])

        size = img.shape

        focal_length = size[1]
        center = (size[1] // 2, size[0] // 2) #顔の中心座標

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype='double')

        dist_coeffs = np.zeros((4, 1))

        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        #回転行列とヤコビアン
        (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
        mat = np.hstack((rotation_matrix, translation_vector))

        #yaw,pitch,rollの取り出し
        (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)
        yaw = eulerAngles[1]
        roll = eulerAngles[2]

        cv2.putText(img, 'yaw : ' + str(int(yaw)), (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(img, 'roll : ' + str(int(roll)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        rip_points = np.array([
                    tuple(shape[49]),#口
                    tuple(shape[50]),
                    tuple(shape[51]),
                    tuple(shape[52]),
                    tuple(shape[53]),
                    tuple(shape[54]),
                    tuple(shape[55]),
                    tuple(shape[56]),
                    tuple(shape[57]),
                    tuple(shape[58]),
                    tuple(shape[59]),
                    tuple(shape[60]),
                ],dtype='int')

        eyebrow_points = np.array([
                    tuple(shape[35]),#右小鼻0
                    tuple(shape[31]),#左小鼻1
                    tuple(shape[45]),#右目尻2
                    tuple(shape[36]),#左目尻3
                    tuple(shape[16]),#右輪郭側面4
                    tuple(shape[0]),#左輪郭側面5
                    tuple(shape[26]),#右眉尻6
                    tuple(shape[17]),#左眉尻7
                    tuple(shape[24]),#右眉山8
                    tuple(shape[19]),#左眉山9
                    tuple(shape[30]),#鼻頭10
                ],dtype='int')

    # 結果の表示
    cv2.imshow('img', img)
    count = count + 1

    if cv2.waitKey(1) & 0xFF == ord('c'):#val_decoded == b'1\r\n':
        if yaw >= -5 and yaw <= 5 and roll >= -5 and roll <= 5:
            playsound("syatta-.mp3")
            img2 = ripcut(rip_points)
            rip_hsv(img2, rip_points)
            if n == 0:
                cv2.imwrite('hikaku_0.jpg',ripdst())
                playsound("takepicture.mp3")
            else:
                cv2.imwrite('hikaku_1.jpg',ripdst())
                ripdst_0 = cv2.imread("hikaku_0.jpg")
                ripdst_1 = cv2.imread("hikaku_1.jpg")
                height, width = ripdst_0.shape[:2]
                size = (width, height)
                reimg = cv2.resize(ripdst_1, size)
                cv2.imwrite('rehikaku_1.jpg',reimg)
                du = ripcutend_ue()
                ds = ripcutend_sita()
                judge(du, ds)
            n = n + 1
            playsound("yomikomi.mp3")
            #if n == 1:


    if cv2.waitKey(1) & 0xFF == ord('m'):#val_decoded == b'2\r\n':#
        if yaw >= -5 and yaw <= 5 and roll >= -5 and roll <= 5:
            playsound("syatta-.mp3")
            cv2.imwrite('face.jpg',img)
            if m == 0:
                mul = eyebrow_points[8][1]-30#左上
                msl = eyebrow_points[6][1]+5#左下
                mhx = eyebrow_points[10][0]#鼻x
                mhy = eyebrow_points[10][1]#鼻y
                mml = eyebrow_points[4][0]-20#左輪郭(眉尻側)

                mur = eyebrow_points[9][1]-30#右上
                msr = eyebrow_points[7][1]+5#右下
                mmr = eyebrow_points[5][0]+20#右輪郭(眉尻側)
                m = m + 1
            else:
                mhx2 = eyebrow_points[10][0]
                mhy2 = eyebrow_points[10][1]
                sx = mhx - mhx2
                sy = mhy - mhy2
                mul = mul - sy
                msl = msl - sy
                mml = mml - sx

                mur = mur - sy
                msr = msr - sy
                mmr = mmr - sx
                mcutl = mayucutl(mul, msl, mhx2, mml)
                mcutr = mayucutr(mur, msr, mmr, mhx2)

                count=0
                counth=1000

                # 元画像のグレースケール化
                mayugrayl = cv2.cvtColor(mcutl, cv2.COLOR_BGR2GRAY)
                mayugrayr = cv2.cvtColor(mcutr, cv2.COLOR_BGR2GRAY)
                
                # コーナー検出
                cornersl = cv2.goodFeaturesToTrack(mayugrayl, 10, 0.05, 10)
                cornersl = np.int0(cornersl)
                cornersr = cv2.goodFeaturesToTrack(mayugrayr, 10, 0.05, 10)
                cornersr = np.int0(cornersr)

                # 特徴点を元画像に反映
                for i in cornersl:
                    a,b= i.ravel()
                    if count < a:
                        count = a
                        msxl, msyl= i.ravel()
                for j in cornersr:
                    c,d= j.ravel()
                    if counth > c:
                        counth = c
                        msxr, msyr= j.ravel()

                print((msxl,msyl))
                print((msxr,msyr))

                msxl = mhx2 + msxl
                msyl = mul + msyl
                cv2.circle(img, (msxl, msyl), 3, 255, -1)
                
                msxr = mmr + msxr
                msyr = mur + msyr
                cv2.circle(img, (msxr, msyr), 3, 255, -1)
                
                # 特徴点を元画像に反映させた画像を作成
                cv2.imwrite("mayuout.png", img)
                kyoril = tyokusenn(eyebrow_points[0][0], eyebrow_points[0][1], eyebrow_points[2][0], eyebrow_points[2][1], msxl,msyl)
                kyorir = tyokusenn(eyebrow_points[1][0], eyebrow_points[1][1], eyebrow_points[3][0], eyebrow_points[3][1], msxr,msyr)

                if (kyoril == 0):
                    playsound("hamidashinashi.mp3")
                elif (kyoril == 1):
                    playsound("yomikomi.mp3")
                elif (kyoril == -1):
                    playsound("yomikomi.mp3")
                
                if (kyorir == 0):
                    playsound("hamidashinashi.mp3")
                elif (kyorir == 1):
                    playsound("yomikomi.mp3")
                elif (kyorir == -1):
                    playsound("yomikomi.mp3")

            playsound("yomikomi.mp3")
        
                
    # 'q'が入力されるまでループ
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# 後処理
#ser.close()
cap.release()
cv2.destroyAllWindows()

