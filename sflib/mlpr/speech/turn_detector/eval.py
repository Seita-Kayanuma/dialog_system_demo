from ....corpus.speech.duration import DurationInfo
import numpy as np
from .base import TurnDetector
from ....eval.interval import eval_interval, summarize_eval
import tqdm


def extract_turn_from_duration_info(duration_info: DurationInfo, ch: int):
    """DurationInfoから評価用のターン情報を展開する．
    duration_info.turn の中から，'T'とラベル付けされた区間の
    リストを返す．

    Args:
      duration_info (DurationInfo): 区間情報
      ch (int): チャネル． 0 か 1．

    Returns:
      list: ターンの区間情報 (開始時間, 終了時間) のタプルのリスト．
        開始時間順（早い順）にソートされており，区間のオーバーラップは無い．
    """
    result = [(s, e) for (s, e, t) in duration_info.turn[ch] if t == 'T']
    return result


def apply_median_filter(x: np.array, kernel_size: int = 5):
    """メディアンフィルタを適用する（実際の問題に即すために遅れ時間を考慮する）
    """
    from scipy.signal import medfilt
    rx = medfilt(x, kernel_size)
    rx = np.concatenate([[0] * (kernel_size - 1), rx[:-(kernel_size - 1)]])
    return rx


def convert_labels_to_interval_list(x: np.array, rate: float):
    """
    1である区間を抽出する
    """
    x = np.concatenate([[0], x, [0]])
    dx = x[1:] - x[:-1]
    st = np.where(dx > 0)[0] / rate * 1000.0
    en = np.where(dx < 0)[0] / rate * 1000.0
    return list(zip(st, en))


def evaluate_turn_detector(turn_detector: TurnDetector,
                           duration_info: DurationInfo,
                           ch: int,
                           median_kernel_size: int = 5):
    """TurnDetectorの評価を，一つの音声情報に対して行う．
    
    Args:
      turn_detector (TurnDetector): 評価対象のTurnDetector
      duration_info (DurtionInfo): 評価を行うデータ
      ch (int): DurationInfoはステレオ音声なので，どちらのチャネルを評価するか 0 か 1
      median_kernel_size (int): 出力平滑化のフィルタのカーネルサイズ．大きくすると出力が安定するが
        遅延が大きくなる     
    """
    # 音声データ
    wav = duration_info.wav[ch, :].reshape(1, -1)
    # 予測を行う
    turn_detector.reset()
    y = turn_detector.predict(wav, out_type='softmax')
    duration_info.clear_cache()
    y_raw = y[0][0].clone().detach().cpu().numpy()
    # 出力が1のところだけに1を立てる
    y = np.array(np.argmax(y_raw, axis=1) == 1, dtype=np.int64)
    # 平滑化する
    if median_kernel_size > 0:
        y = apply_median_filter(y, median_kernel_size)
    # 出力のフレームレート
    rate = turn_detector.feature_extractor.feature_rate
    # 区間情報に直す
    y_intervals = convert_labels_to_interval_list(y, rate)

    # 正解区間情報
    t_intervals = extract_turn_from_duration_info(duration_info, ch)

    eval_info = eval_interval(y_intervals, t_intervals)
    eval_info_summary = summarize_eval(eval_info)

    return eval_info, eval_info_summary, y_intervals, t_intervals, y_raw


def combine_eval_info(eval_info_list):
    """評価情報を統合する．
    eval_info自体は各項目を足せばOKなようにできている
    """
    import copy
    if len(eval_info_list) == 0:
        raise RuntimeError('empty list')
    names = eval_info_list[0].keys()
    result = copy.deepcopy(eval_info_list[0])
    for e in eval_info_list[1:]:
        for n in names:
            result[n] += e[n]
    return result


def evaluate_turn_detector_duration_infos(turn_detector: TurnDetector,
                                          duration_info_list: list,
                                          ch_list: list = [0, 1],
                                          median_kernel_size: int = 5):
    ce = None
    for di in tqdm.tqdm(duration_info_list):
        for ch in ch_list:
            e, es, y, t, _ = evaluate_turn_detector(turn_detector, di, ch,
                                                    median_kernel_size)
            if ce is None:
                ce = e
            else:
                ce = combine_eval_info([ce, e])
    eval_info_summary = summarize_eval(ce)
    return ce, eval_info_summary
