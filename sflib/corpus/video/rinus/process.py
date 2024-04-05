from os import path
import numpy as np
import pandas as pd
from . import Rinus
from ....img.face.feature.ppes2.extraction import PPESv2, PPESv2Extractor
from ....video.reader import VideoReader


class PPESv2Data:
    """
    Rinusのデータ100人から，それぞれ100枚ずつ抜き出し，
    PPESv2を抽出してまとめたもの
    """
    DEFAULT_DF_PATH = path.join(Rinus.DEFAULT_PATH, 'rinus_ppes_v2.df.pkl')

    def __init__(self, filepath=None):
        if filepath is None:
            filepath = PPESv2Data.DEFAULT_DF_PATH
        self.dataframe = pd.read_pickle(filepath)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get_item_by_index(key)
        elif isinstance(key, slice):
            r = [
                self.get_item_by_index(i)
                for i in range(*(key.indices(len(self))))
            ]
            return tuple(r)
        elif isinstance(key, str):
            return self.get_items_by_name(key)

    def get_item_by_index(self, i) -> PPESv2:
        r = self.dataframe.iloc[i, :]
        id = r['id']
        s = 2
        e = s + 64 * 64
        image = np.array(r[s:e], 'uint8')
        image = image.reshape((64, 64))
        s = e
        e = s + 68 * 2
        landmarks = np.array(r[s:e], 'float32').reshape(-1, 2)
        return PPESv2(id, image, landmarks)

    @staticmethod
    def generate(filepath=None, refresh=False):
        if filepath is None:
            filepath = PPESv2Data.DEFAULT_DF_PATH
            
        # カラム名を生成
        # (1) IDと通し番号
        column_names = ['id', 'image_no']
        dtypes = ['object', 'int64']
        # (2) 画像（64x64）
        for y in range(64):
            for x in range(64):
                column_names.append('i%03d/%03d' % (x, y))
                dtypes.append('uint8')
        # (3) ランドマーク
        for i in range(68):
            column_names.extend([('l%02dx' % i), ('l%02dy' % i)])
            dtypes.extend(['float32', 'float32'])

        if path.exists(filepath) and refresh is False:
            df_out = pd.read_pickle(filepath)
        else:
            df_out = None

        extractor = PPESv2Extractor()

        rinus = Rinus()
        id_list = rinus.get_id_list()
        count = 0
        for id in id_list:
            # 書き出すID．通し番号だけだとどこかで重複する可能性が
            # あるので先頭に"Rinus"をつける．
            eid = "Rinus{}".format(id)
            mp4path = rinus.get_mp4_path(id)
            video = VideoReader(mp4path)
            num_frames = video.framecount
            num_to_extract = 100
            interval = num_frames / num_to_extract
            indices = np.int64(np.floor(
                np.arange(num_to_extract) * interval)).tolist()
            for idx in indices:
                print("{} {} ".format(eid, idx), end='', flush=True)
                if df_out is not None and \
                   np.any((df_out['id'] == eid) &
                          (df_out['image_no'] == int(idx))):
                    print("already processed")
                    continue
                img = video.get_frame(idx)
                ppes_list = extractor.extract(img, id=eid)
                while len(ppes_list) == 0:
                    print("any face is not found")
                    idx += 1
                    if idx >= num_frames:
                        break
                    print("{} {} ".format(eid, idx), end='', flush=True)
                    img = video.get_frame(idx)
                    ppes_list = extractor.extract(img, id=eid)
                ppes = ppes_list[0]
                data = [eid, int(idx)]
                data.extend(ppes.image_orig.ravel().tolist())
                data.extend(ppes.landmarks.ravel().tolist())
                ldata = [[d] for d in data]
                # import ipdb; ipdb.set_trace()
                df = pd.DataFrame(dict(zip(column_names, ldata)))
                # import ipdb; ipdb.set_trace()
                # df = df.astype(dict(zip(column_names, dtypes)))
                if df_out is None:
                    df_out = df
                else:
                    df_out = df_out.append(df)
                print("OK")
                # 念のため100回に1回は保存しておく
                count = count + 1
                if count % 100 == 0:
                    df_out = df_out.astype(dict(zip(column_names, dtypes)))
                    df_out.to_pickle(filepath)
        df_out.to_pickle(filepath)
                
