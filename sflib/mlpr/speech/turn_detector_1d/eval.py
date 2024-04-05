# 評価用データの生成
from typing import List
import numpy as np
from os import path
import os
import torch
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from .dataset import CachedLabelFeaturesDataset
from .base import TurnDetector1d, construct_turn_detector
from ....corpus.speech.duration_v2 import DurationInfoV2, DurationInfoV2Manager
import pandas as pd


def predict_with_feature(turn_detector: TurnDetector1d,
                         batch: List[torch.Tensor],
                         sub_batch_len: int = 500):
    """CachedLabelFeaturesDatasetから得られるデータのリストを
    用いて，評価のための予測を行う

    Args:
      turn_detector: 予測器
      batch: CachedLabelFeaturesDatasetから得られるデータのリスト
      sub_batch_len: 何フレーム毎にdetachしながら予測するか．

    Returns:
      batchと同じ長さのリスト．
      i番目の要素は，batch[i]と，batch[i]を入力とした予測結果を並べたもの．
      ただし特徴量は含まない．
      具体的には，(L, 2, D)のテンソルで，
      Dは順に，u(t)，s(t)，z(t)，f(t)，y(t)，a(t)(locked)，a(t)，u(t)(predicted)を並べたもの．
    """
    # 各データの長さのリスト
    lengths = [d.shape[0] for d in batch]
    # 最長の長さ
    max_length = max(lengths)
    # u(t)の次元数
    ut_dim = turn_detector.input_calculator.ut_dim
    
    dummy_zero = torch.zeros(
        1, 2, batch[0].shape[2],
        device=turn_detector.input_calculator.device)

    y_padded_list = []
    turn_detector.torch_model.eval()
    turn_detector.reset()
    offset_frame = 0
    while offset_frame < max_length:
        sub_batch = [d[offset_frame:(offset_frame + sub_batch_len)]
                     for d in batch]
        ut_list = []
        feat_list = []
        st_list = []
        real_lengths = []
        for sb in sub_batch:
            for ch in (0, 1):
                real_lengths.append(sb.shape[0])
                if sb.shape[0] == 0:
                    sb = dummy_zero
                ut_list.append(sb[:, ch, 4:(4 + ut_dim)])
                feat_list.append(sb[:, ch, (4 + ut_dim):])
                st_list.append(sb[:, ch, 1:2])
        ut_packed = pack_sequence(ut_list, enforce_sorted=False)
        feat_packed = pack_sequence(feat_list, enforce_sorted=False)
        st_packed = pack_sequence(st_list, enforce_sorted=False)

        y_packed = turn_detector.forward_core(
            ut_packed, feat_packed, st_packed)
        y_padded, _ = pad_packed_sequence(y_packed)
        y_padded_list.append(y_padded)

        turn_detector.detach()
        offset_frame += sub_batch_len
    y_padded = torch.cat(y_padded_list, dim=0)
    ret = []
    for i, d in enumerate(batch):
        length = lengths[i]
        data = torch.cat([d[:, :, :4],
                          y_padded[:length, (i*2):(i*2+2), :],
                          d[:, :, 4:(4 + ut_dim)]],
                         dim=2)
        ret.append(data)
    return ret


def get_filename(turn_detector: TurnDetector1d,
                 dataset_name: str, id):
    """predict_with_featureの結果を書き出すファイル名（フルパス）を生成する
    """
    from .... import config
    package_dir = config.get_package_data_dir(__package__)
    filename = path.join(package_dir,
                         'eval',
                         turn_detector.filename_base,
                         dataset_name,
                         '{}.npy'.format(id))
    return filename


def predict_and_save(turn_detector: TurnDetector1d,
                     duration_info_manager: DurationInfoV2Manager,
                     dataset_name: str,
                     id_list: List[str], overwrite=False):
    dataset = CachedLabelFeaturesDataset(
        turn_detector.input_calculator, duration_info_manager,
        'waseda_soma', id_list)
    batch_size = 10
    count = 0
    while count < len(dataset):
        sub_id_list = []
        sub_data_list = []
        while count < len(dataset) and len(sub_id_list) < batch_size:
            id = id_list[count]
            if overwrite is False:
                filename = get_filename(turn_detector, dataset_name, id)
                if path.exists(filename):
                    count += 1
                    continue
            sub_id_list.append(id)
            sub_data_list.append(dataset[count])
            count += 1
        if len(sub_id_list) > 0:
            result_list = predict_with_feature(turn_detector, sub_data_list)
            for id, result in zip(sub_id_list, result_list):
                filename = get_filename(turn_detector, dataset_name, id)
                dirname = path.dirname(filename)
                if not path.exists(dirname):
                    os.makedirs(dirname, mode=0o755, exist_ok=True)
                print(filename)
                np.save(filename, result.cpu().detach().numpy())


def read_result(turn_detector: TurnDetector1d,
                duration_info_manager: DurationInfoV2Manager,
                dataset_name: str, id, ch):
    filename = get_filename(
        turn_detector, dataset_name, id)
    r = np.load(filename)
    r = r[:, ch, :]

    # r = predict_and_pack_for_each_ch(
    #     turn_detector, duration_info_manager.get_duration_info(id), ch)
    
    ut = r[:, 0]
    st = r[:, 1]
    zt = r[:, 2]
    ft = r[:, 3]
    yt = r[:, 4]
    at_locked = r[:, 5]
    at = r[:, 6]
    ut_pred = r[:, 7:]
    return ut, st, zt, ft, yt, at_locked, at, ut_pred
    

def bin_to_interval(x, framerate):
    x = np.concatenate([x, [0]])
    x_pre = np.concatenate([[0], x[:-1]])
    x_diff = x - x_pre
    starts = np.where(x_diff > 0.1)
    ends = np.where(x_diff < -0.1)

    starts = starts[0].tolist()
    ends = ends[0].tolist()
    
    intervals = [(s/framerate, e/framerate) for s, e in zip(starts, ends)]

    return intervals


def is_intervals_crossed(int1, int2):
    return int1[0] <= int2[1] and int2[0] <= int1[1]


def extract_crossed_intervals(ints_src, ints_ref):
    extracted = []
    for int_src in ints_src:
        for int_ref in ints_ref:
            if is_intervals_crossed(int_src, int_ref):
                extracted.append(int_src)
                break
    return extracted


def bin_to_timings(x, framerate):
    t = np.where(x > 0.1)
    t = t[0] / framerate
    t = t.tolist()
    return t


def extract_withing_timings(timings, ints_ref):
    extracted = []
    for t in timings:
        for int_ref in ints_ref:
            if t >= int_ref[0] and t <= int_ref[1]:
                extracted.append(t)
    return extracted


def make_zt_pred(yt):
    yt1 = np.concatenate([[0.0], yt[:-1]])
    zt_pred = np.float32((yt1 < 0.8) & (yt >= 0.8))
    return zt_pred


def evaluate_timings(zt_pred, zt_true, e_ints):
    """評価値を計算する

    zt_pred: zt（時間）のリスト（予測値）
    zt_true: zt（時間）のリスト（真値）
    e_ints: 有効なutの区間情報（開始時間と終了時間のリスト）
    """
    count_true_positive = 0
    count_false_positive = 0
    count_true_negative = 0
    count_false_negative = 0
    errors_in_true_positive = []
    errors_in_false_positive = []

    for st, en in e_ints:
        pred_positive = False
        pred_timing = None
        for tim in zt_pred:
            if st <= tim and tim <= en:
                pred_positive = True
                pred_timing = tim
                break
        target_positive = False
        target_timing = None
        for tim in zt_true:
            if st <= tim and tim <= en:
                target_positive = True
                target_timing = tim
                break
        if pred_positive and target_positive:
            count_true_positive += 1
            errors_in_true_positive.append(pred_timing - target_timing)
        elif pred_positive and not target_positive:
            count_false_positive += 1
            errors_in_false_positive.append(
                (pred_timing - st, en - pred_timing))
        elif not pred_positive and target_positive:
            count_false_negative += 1
        else:
            count_true_negative += 1
    return \
        count_true_positive, count_false_positive, \
        count_true_negative, count_false_negative, \
        errors_in_true_positive, errors_in_false_positive


def evaluate_timings_from_data_core(ut, zt, ft, yt, framerate):
    ut_ints = bin_to_interval(ut, framerate)
    ft_ints = bin_to_interval(ft, framerate)
    ut_ints = extract_crossed_intervals(ut_ints, ft_ints)
    zt_tim = bin_to_timings(zt, framerate)
    zt_tim = extract_withing_timings(zt_tim, ut_ints)
    zt_pred = make_zt_pred(yt)
    zt_pred_tim = bin_to_timings(zt_pred, framerate)
    # import ipdb; ipdb.set_trace()
    zt_pred_tim = extract_withing_timings(zt_pred_tim, ut_ints)
    return evaluate_timings(zt_pred_tim, zt_tim, ut_ints)


def evaluate_timings_from_data(turn_detector: TurnDetector1d,
                               duration_info_manager: DurationInfoV2Manager,
                               dataset_name: str, id, ch):
    ut, _, zt, ft, yt, _, _, ut_pred = read_result(
        turn_detector, duration_info_manager, dataset_name, id, ch)
    return evaluate_timings_from_data_core(
        ut, zt, ft, yt, turn_detector.input_calculator.feature_rate)


# ---------------------------------------------------------------------
def evaluate_timing_data_and_write(turn_detector: TurnDetector1d,
                                   duration_info_manager: DurationInfoV2Manager,
                                   dataset_name: str, id_list: List[str]):
    predict_and_save(turn_detector, duration_info_manager,
                     dataset_name, id_list)
    
    results_id = []
    results_ch = []
    results_tp = []
    results_fp = []
    results_tn = []
    results_fn = []
    results_er_tp = []
    results_er_fp = []
    
    for id in id_list:
        for ch in (0, 1):
            tp, fp, tn, fn, er_tp, er_fp = \
                evaluate_timings_from_data(
                    turn_detector, duration_info_manager,
                    'waseda_soma', id, ch)
            results_id.append(id)
            results_ch.append(ch)
            results_tp.append(tp)
            results_fp.append(fp)
            results_tn.append(tn)
            results_fn.append(fn)
            results_er_tp.append(er_tp)
            results_er_fp.append(er_fp)
    df = pd.DataFrame(dict(
        id=results_id,
        channel=results_ch,
        true_positive=results_tp,
        false_positive=results_fp,
        true_negative=results_tn,
        false_negative=results_fn,
        errors_in_true_positive=results_er_tp,
        errors_in_false_positive=results_er_fp))
    from .... import config
    package_dir = config.get_package_data_dir(__package__)
    filename = path.join(package_dir,
                         'eval',
                         '{}_{}.df.pkl'.format(
                             turn_detector.filename_base,
                             dataset_name))
    df.to_pickle(filename)


def _construct_and_load():
    vad_feature_extractor_construct_args = dict(
        autoencoder_number=12,
        autoencoder_trainer_number=13,
        autoencoder_model_version=2)
    input_calculator_construct_args = dict(
        voice_activity_detector_number=1,
        voice_activity_detector_trainer_number=1,
        voice_activity_detector_feature_extractor_number=3,
        voice_activity_detector_feature_extractor_construct_args=vad_feature_extractor_construct_args,
    )
    turn_detector = construct_turn_detector(
        turn_detector_number=11,
        trainer_number=16,
        input_calculator_number=1,
        input_calculator_construct_args=input_calculator_construct_args)
    turn_detector.load()
    turn_detector.to('cuda:0')
    return turn_detector


def _test():
    turn_detector = _construct_and_load()
    
    from ....corpus.speech import waseda_soma as ws
    wss = ws.WASEDA_SOMA()
    dim = ws.DurationInfoV2ManagerWasedaSoma(wss)
    id_list = wss.get_id_list()

    predict_and_save(turn_detector, dim, 'waseda_soma', id_list[310:])

    results_id = []
    results_ch = []
    results_tp = []
    results_fp = []
    results_tn = []
    results_fn = []
    results_er_tp = []
    results_er_fp = []
    
    for id in id_list[310:]:
        for ch in (0, 1):
            tp, fp, tn, fn, er_tp, er_fp = \
                evaluate_timings_from_data(
                    turn_detector, dim, 'waseda_soma', id, ch)
            results_id.append(id)
            results_ch.append(ch)
            results_tp.append(tp)
            results_fp.append(fp)
            results_tn.append(tn)
            results_fn.append(fn)
            results_er_tp.append(er_tp)
            results_er_fp.append(er_fp)
    import pandas as pd
    df = pd.DataFrame(dict(
        id=results_id,
        channel=results_ch,
        true_positive=results_tp,
        false_positive=results_fp,
        true_negative=results_tn,
        false_negative=results_fn,
        errors_in_true_positive=results_er_tp,
        errors_in_false_positive=results_er_fp))
    return df
            
            
