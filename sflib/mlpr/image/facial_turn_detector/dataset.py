from os import path
import os
import shutil
import numpy as np
import pandas as pd
import torch
import time
from torch.utils.data import Dataset
from .... import config
from ....video.reader import VideoReader
from ....data.elan.io import read_from_eaf

from .base import FacialTurnDetectorFeatureExtractor


def extract_label_eaf_kobayashi(duration_info: list, framerate: float,
                                output_num_frames: int):
    """
    Args:
      duration_info (list): (開始時刻, 終了時刻, タグ) のタプル．
        時刻の単位はミリ秒．
        タグは今回は空なので無視．
        開始時刻から終了時刻までを1，それ以外を0にする．
      framerate (list): 1秒あたりのフレーム数

    Returns:
      フレームごとに0（非発話状態）か，1（発話状態）に直したリスト．
    """
    values = []
    for ts, te, _ in duration_info:
        prev_te = len(values) / framerate * 1000.0
        num_frames = int(np.ceil((ts - prev_te) / 1000.0 * framerate))
        values.extend([0] * num_frames)
        prev_te = len(values) / framerate * 1000.0
        num_frames = int(np.ceil((te - prev_te) / 1000.0 * framerate))
        values.extend([1] * num_frames)
    if len(values) > output_num_frames:
        values = values[:output_num_frames]
    elif len(values) < output_num_frames:
        values += [0] * (output_num_frames - len(values))
    return values


def make_cache_filename(fullpath: str) -> str:
    """動画ファイルのフルパス名から，
    キャッシュファイル名(*.pkl)を作成する．
    出力ディレクトリは特徴抽出器によって決まる．
    この関数で生成されるのはファイル名のみ"""
    r = fullpath
    # 拡張子を外す
    r, _ = path.splitext(r)
    # データディレクトリのトップディレクトリを外す
    r = r.replace(config.TOPDIR, '')
    # '/' を '_' に変更する
    r = r.replace(path.sep, '_')
    # ファイル名の先頭に '_' がある場合は取り除く
    while r[0] == '_':
        r = r[1:]
    # 拡張子を付与
    r = r + '.pkl'
    return r


def make_cache_filename_fullpath(
        feature_extractor: FacialTurnDetectorFeatureExtractor,
        movie_fullpath: str) -> str:
    """特徴抽出器，動画ファイルのフルパスから，
    キャッシュファイルのフルパスを生成する
    """
    cache_filename = make_cache_filename(movie_fullpath)
    cache_dirname = path.join(config.get_package_data_dir(__package__),
                              'dataset', feature_extractor.get_filename_base())
    r = path.join(cache_dirname, cache_filename)
    return r


def generate_features(video_path, eaf_path,
                      feature_extractor: FacialTurnDetectorFeatureExtractor):
    fname = path.basename(video_path)
    reader = VideoReader(video_path)
    num_frames = int(reader.framecount)
    # num_frames = 321  # for test
    framerate = reader.framerate

    # 1000フレーム毎に特徴ベクトルを抽出
    sub_num_frames = 1000
    feat_seq = []
    feature_extractor.reset()
    for st in range(0, num_frames, sub_num_frames):
        en = st + sub_num_frames
        if en > num_frames:
            en = num_frames
        t = time.time()
        print("\rgenerate features for {}: {:4d}/{:4d}".format(
            fname, en, num_frames),
              end='',
              flush=True)
        # img_seq = [reader.get_frame(i) for i in range(st, en)]
        img_seq = [reader.get_next_frame() for i in range(st, en)]
        print(" ... {:.3f}".format(time.time() - t), end='', flush=True)
        feat_seq.append(feature_extractor.calc([img_seq])[0])
        fps = (en - st) / (time.time() - t)
        print(" ... {:.3f} fps".format(fps), end='', flush=True)
    print("")
    feat = torch.cat(feat_seq, dim=0)

    # EAFからターゲットのリストを作成
    ei = read_from_eaf(eaf_path)
    target = extract_label_eaf_kobayashi(ei.tiers[0]['annotations'],
                                         framerate=framerate,
                                         output_num_frames=num_frames)
    target = np.int64(target)
    target = torch.tensor(target)

    return feat, target


def save_cache(cache_path, x, t):
    dirname = path.dirname(cache_path)
    if not path.exists(dirname):
        os.makedirs(dirname, mode=0o755, exist_ok=True)
    num_frames, dim = x.shape
    column_names = ['x{:03d}'.format(i) for i in range(dim)] + ['target']
    data = np.concatenate(
        [x.cpu().numpy(), t.cpu().numpy().reshape(-1, 1)], axis=1)
    df = pd.DataFrame(data, columns=column_names)
    df.to_pickle(cache_path)


def read_cache(cache_path):
    df = pd.read_pickle(cache_path)
    x = np.float32(df.iloc[:, :-1])
    t = np.int64(df.iloc[:, -1])

    return torch.tensor(x), torch.tensor(t)


class CachedDataset(Dataset):
    """指定のデータセットで，指定のビデオから特徴抽出済であれば，
    それを返し，そうでなければ抽出を行ってキャッシュしてから
    返すデータセット"""

    def __init__(self, feature_extractor: FacialTurnDetectorFeatureExtractor,
                 video_eaf_paths: list):
        self._feature_extractor = feature_extractor
        self._video_eaf_paths = video_eaf_paths

        # キャッシュファイルのパスリストを作成
        cache_paths = []
        for video_path, _ in self._video_eaf_paths:
            cache_paths.append(
                make_cache_filename_fullpath(self._feature_extractor,
                                             video_path))
        self._cache_paths = cache_paths

    def clear_cache(self):
        dirname = path.dirname(self._cache_paths[0])
        if path.exists(dirname):
            shutil.rmtree(dirname)

    def __len__(self):
        return len(self._video_eaf_paths)

    def __getitem__(self, i):
        cache_path = self._cache_paths[i]
        if path.exists(cache_path):
            x, t = read_cache(cache_path)
        else:
            video_path, eaf_path = self._video_eaf_paths[i]
            x, t = generate_features(video_path, eaf_path,
                                     self._feature_extractor)
            save_cache(cache_path, x, t)
        # offset = torch.randint(0, 150, (1,))[0]
        # x = x[offset:, :]
        # t = t[offset:]
        return x, t
