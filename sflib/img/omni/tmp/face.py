# coding: utf-8
"""
顔処理関係
"""
__author__ = "Shinya FUJIE <shinya.fujie@p.chibakoudai.jp>"
__version__ = "0.0.0"
__date__ = "6 Sept 2018"

import cv2
import dlib
import numpy as np

if __name__ == "__main__":
    from bgs import BGSTracker
    from conv import OmniImageConverter
    import util
    import sys

    model_path = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    # 第1引数を入力動画のファイル名とする
    if len(sys.argv) < 2:
        print("{} <filename>".format(sys.argv[0]))
        sys.exit(-1)
    filename = sys.argv[1]

    # ビデオキャプチャ
    if filename == '0':
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    else:
        cap = cv2.VideoCapture(filename)
    

    # 入力画像（元のサイズだと大きすぎるのでこのサイズに縮小する）
    W, H = 640, 640
    # 変換後の画像サイズ
    CW, CH = 160, 160
    # ズームの割合（単位は不明だがこれくらいが妥当？小さければ引いた画像になり，
    # 大きければよった画像になる）
    NEAR = 200

    # 表示用の大きな画像
    img2show = np.zeros((H, W + CW * 2, 3), np.uint8)

    # 画面表示用の変換後画像の原点．
    # 変換後画像は最大4(8)枚表示する
    convert_image_pos = (
        (W, 0),
        (W, CH),
        (W + CW, 0),
        (W + CW, CH),
    )

    # 背景差分〜おおまかな顔の位置までを推定するオブジェクト
    bgst = BGSTracker(dist_offset=50)

    # 全方位画像から特定の位置を中心とした平面画像を生成するオブジェクト
    cvtr = OmniImageConverter(R=W / 2, w=CW, h=CH, near=NEAR)

    # 何枚のフレームを処理したか
    count = 0

    rt = util.ResultTracker()

    timer = util.Timer()
    fpsc = util.FpsCalculator()

    cv2.namedWindow("image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    
    # 1枚ずつ処理
    while cap.isOpened():
        # キャプチャする
        timer.start()
        ret, frame = cap.read()
        if ret is False:
            continue
        # import ipdb; ipdb.set_trace()        
        timer.report("cap")

        # 縮小する
        frame = cv2.resize(frame, (W, H))
        
        # 表示用にコピーする
        frame2show = frame.copy()

        # 背景差分〜おおまかな顔の位置までを推定
        timer.start()
        bgst.apply(frame)
        timer.report("bgs")

        # 処理枚数をインクリメント
        count += 1
        # 処理枚数が100枚を超えて入れば画像変換処理をする
        # （ある程度の枚数を処理しないと背景差分がまともに動作しないため）
        if count > 0:
            results = rt.track(bgst.results)

            # おおまかな顔の位置ごとに，表示用フレーム画像にマーカーを描画する
            for angle, x, y, dist in results:
                cv2.drawMarker(frame2show, (x, y), (0, 0, 255),
                               cv2.MARKER_CROSS, 20, 3)
            img2show[:, :, :] = 0

            # おおまかな顔の位置ごとに変換画像を生成する
            timer.start()
            i = 0
            for angle, x, y, dist in results:
                img_o, table = cvtr.convert(frame, x, y)
                if img_o is None:
                    continue

                dets = detector(img_o, 1)
                for d in dets:
                    r = predictor(img_o, d)
                    xs, ys = [], []
                    for p in r.parts():
                        cv2.circle(img_o, (p.x, p.y), 1, (255, 0, 0),
                                   cv2.FILLED)
                        xs.append(p.x)
                        ys.append(p.y)
                    if len(xs) > 0:
                        xs = np.array(xs, np.int32)
                        ys = np.array(ys, np.int32)
                        cond = (xs > 0) & (xs < CW) & (ys > 0) & (ys < CH)
                        xs = xs[cond]
                        ys = ys[cond]
                        xyis = table[ys, xs, :]
                        for px, py in xyis:
                            cv2.circle(frame2show, (int(px), int(py)), 1, (255, 0, 0),
                                       cv2.FILLED)
                    cv2.rectangle(img_o, (d.left(), d.top()),
                                  (d.right(), d.bottom()), (0, 255, 0), 1)

                # 表示用画像にコピーする
                img2show[convert_image_pos[i][1]:convert_image_pos[i][1] + CH,
                         convert_image_pos[i][0]:convert_image_pos[i][0] + CW, :] \
                         = img_o
                i += 1
                if i >= len(convert_image_pos):
                    break
            timer.report("cnv")
            img2show[:H, :W, :] = frame2show
        else:
            img2show[:H, :W, :] = frame2show

        cv2.putText(img2show, "%5.2ffps" % fpsc.tick(), (0, H),
                    cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 1)

        cv2.imshow("image", img2show)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
