# coding: utf-8
# 雑多な実験スクリプトの集合
# 各関数の数字には特に意味は無く，単なる通し番号


def train0001(device='cuda'):
    from .feature_extractor_0001 \
        import TurnDetectorFeatureExtractor0001 as FeatureExtractor
    from .turn_detector_0001 \
        import TurnDetector0001 as TurnDetector
    from .trainer_waseda_0001 \
        import TurnDetectorTrainerWaseda0001 as TurnDetectorTrainer
    from . import base

    feature_extractor = FeatureExtractor(12,
                                         'csj_0006',
                                         'CSJ0006',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    trainer.train()

    base.save_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()


def train0002(device='cuda'):
    from .feature_extractor_0002 \
        import TurnDetectorFeatureExtractor0002 as FeatureExtractor
    from .turn_detector_0001 \
        import TurnDetector0001 as TurnDetector
    from .trainer_waseda_0001 \
        import TurnDetectorTrainerWaseda0001 as TurnDetectorTrainer
    from . import base
    feature_extractor = FeatureExtractor(12,
                                         'csj_0006',
                                         'CSJ0006',
                                         2,
                                         5,
                                         'csj_rwcp_0003',
                                         'CSJRWCP0003',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    trainer.train()

    base.save_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()


def train0003(device='cuda'):
    from .feature_extractor_0001 \
        import TurnDetectorFeatureExtractor0001 as FeatureExtractor
    from .turn_detector_0001 \
        import TurnDetector0001 as TurnDetector
    from .trainer_waseda_0001 \
        import TurnDetectorTrainerWaseda0001 as TurnDetectorTrainer
    from . import base

    feature_extractor = FeatureExtractor(12,
                                         'csj_0008',
                                         'CSJ0008',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    trainer.train()

    base.save_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()


def train0004(device='cuda'):
    # --- 2019/07/09 作ってみたけど Trainer0003 を試したいので未完
    from .feature_extractor_0001 \
        import TurnDetectorFeatureExtractor0001 as FeatureExtractor
    from .turn_detector_0001 \
        import TurnDetector0001 as TurnDetector
    from .trainer_waseda_0002 \
        import TurnDetectorTrainerWaseda0002 as TurnDetectorTrainer
    from . import base

    feature_extractor = FeatureExtractor(12,
                                         'csj_0008',
                                         'CSJ0008',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    trainer.train()

    base.save_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()


def train0005(device='cuda'):
    from .feature_extractor_0001 \
        import TurnDetectorFeatureExtractor0001 as FeatureExtractor
    from .turn_detector_0001 \
        import TurnDetector0001 as TurnDetector
    from .trainer_waseda_0003 \
        import TurnDetectorTrainerWaseda0003 as TurnDetectorTrainer
    from . import base

    feature_extractor = FeatureExtractor(12,
                                         'csj_0008',
                                         'CSJ0008',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    trainer.train()

    base.save_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()


def train0005TR0005(device='cuda'):
    from .feature_extractor_0001 \
        import TurnDetectorFeatureExtractor0001 as FeatureExtractor
    from .turn_detector_0001 \
        import TurnDetector0001 as TurnDetector
    from .trainer_waseda_0005 \
        import TurnDetectorTrainerWaseda0005 as TurnDetectorTrainer
    from . import base

    feature_extractor = FeatureExtractor(12,
                                         'csj_0008',
                                         'CSJ0008',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    trainer.train()

    base.save_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()


def train0005TD0002(device='cuda'):
    from .feature_extractor_0001 \
        import TurnDetectorFeatureExtractor0001 as FeatureExtractor
    from .turn_detector_0002 \
        import TurnDetector0002 as TurnDetector
    from .trainer_waseda_0003 \
        import TurnDetectorTrainerWaseda0003 as TurnDetectorTrainer
    from . import base

    feature_extractor = FeatureExtractor(12,
                                         'csj_0008',
                                         'CSJ0008',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    trainer.train()

    base.save_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()

    
def train0005TR0005TD0002(device='cuda'):
    from .feature_extractor_0001 \
        import TurnDetectorFeatureExtractor0001 as FeatureExtractor
    from .turn_detector_0002 \
        import TurnDetector0002 as TurnDetector
    from .trainer_waseda_0005 \
        import TurnDetectorTrainerWaseda0005 as TurnDetectorTrainer
    from . import base

    feature_extractor = FeatureExtractor(12,
                                         'csj_0008',
                                         'CSJ0008',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    trainer.train()

    base.save_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()


def train0005TD0003(device='cuda'):
    from .feature_extractor_0001 \
        import TurnDetectorFeatureExtractor0001 as FeatureExtractor
    from .turn_detector_0003 \
        import TurnDetector0003 as TurnDetector
    from .trainer_waseda_0003 \
        import TurnDetectorTrainerWaseda0003 as TurnDetectorTrainer
    from . import base

    feature_extractor = FeatureExtractor(12,
                                         'csj_0008',
                                         'CSJ0008',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    trainer.train()

    base.save_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()


def train0006(device='cuda'):
    from .feature_extractor_0002 \
        import TurnDetectorFeatureExtractor0002 as FeatureExtractor
    from .turn_detector_0001 \
        import TurnDetector0001 as TurnDetector
    from .trainer_waseda_0003 \
        import TurnDetectorTrainerWaseda0003 as TurnDetectorTrainer
    from . import base

    feature_extractor = FeatureExtractor(12,
                                         'csj_0008',
                                         'CSJ0008',
                                         2,
                                         5,
                                         'csj_rwcp_0003',
                                         'CSJRWCP0003',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    trainer.train()

    base.save_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()


def train0006TD0002(device='cuda'):
    from .feature_extractor_0002 \
        import TurnDetectorFeatureExtractor0002 as FeatureExtractor
    from .turn_detector_0002 \
        import TurnDetector0002 as TurnDetector
    from .trainer_waseda_0003 \
        import TurnDetectorTrainerWaseda0003 as TurnDetectorTrainer
    from . import base

    feature_extractor = FeatureExtractor(12,
                                         'csj_0008',
                                         'CSJ0008',
                                         2,
                                         5,
                                         'csj_rwcp_0003',
                                         'CSJRWCP0003',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    trainer.train()

    base.save_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()

    
def train0007(device='cuda'):
    from .feature_extractor_0001 \
        import TurnDetectorFeatureExtractor0001 as FeatureExtractor
    from .turn_detector_0001 \
        import TurnDetector0001 as TurnDetector
    from .trainer_waseda_0003 \
        import TurnDetectorTrainerWaseda0003 as TurnDetectorTrainer
    from . import base

    feature_extractor = FeatureExtractor(18,
                                         'csj_0008',
                                         'CSJ0008',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    trainer.train()

    base.save_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()


def train0008(device='cuda'):
    from .feature_extractor_0001 \
        import TurnDetectorFeatureExtractor0001 as FeatureExtractor
    from .turn_detector_0001 \
        import TurnDetector0001 as TurnDetector
    from .trainer_waseda_0003 \
        import TurnDetectorTrainerWaseda0003 as TurnDetectorTrainer
    from . import base

    feature_extractor = FeatureExtractor(12,
                                         'csj_0009',
                                         'CSJ0009',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    trainer.train()

    base.save_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()

    
def train0008TD0002(device='cuda'):
    from .feature_extractor_0001 \
        import TurnDetectorFeatureExtractor0001 as FeatureExtractor
    from .turn_detector_0002 \
        import TurnDetector0002 as TurnDetector
    from .trainer_waseda_0003 \
        import TurnDetectorTrainerWaseda0003 as TurnDetectorTrainer
    from . import base

    feature_extractor = FeatureExtractor(12,
                                         'csj_0009',
                                         'CSJ0009',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    trainer.train()

    base.save_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()
    
    
def train0009(device='cuda'):
    # 0005 とほぼ同じ．Trainer が 0003 -> 0004 に変更
    from .feature_extractor_0001 \
        import TurnDetectorFeatureExtractor0001 as FeatureExtractor
    from .turn_detector_0001 \
        import TurnDetector0001 as TurnDetector
    from .trainer_waseda_0004 \
        import TurnDetectorTrainerWaseda0004 as TurnDetectorTrainer
    from . import base

    feature_extractor = FeatureExtractor(12,
                                         'csj_0008',
                                         'CSJ0008',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    trainer.train()

    base.save_turn_detector(trainer, upload=True)
    trainer.upload_csv_log()
    

def setup_eval0001(device='cuda'):
    from .feature_extractor_0001 \
        import TurnDetectorFeatureExtractor0001 as FeatureExtractor
    from .turn_detector_0001 \
        import TurnDetector0001 as TurnDetector
    from .trainer_waseda_0001 \
        import TurnDetectorTrainerWaseda0001 as TurnDetectorTrainer
    from . import base
    from ....corpus.speech.waseda_soma \
        import WASEDA_SOMA, DurationInfoManagerWasedaSoma

    feature_extractor = FeatureExtractor(12,
                                         'csj_0006',
                                         'CSJ0006',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    base.load_turn_detector(trainer, map_location=device)

    waseda_soma = WASEDA_SOMA()
    id_list_eval = waseda_soma.get_id_list()[310:]
    dim = DurationInfoManagerWasedaSoma(waseda_soma)
    duration_info_list = [dim.get_duration_info(id) for id in id_list_eval]
    return turn_detector, duration_info_list


# -------------------
def update_evaluation_result(cond_name, eval_summary):
    from ....eval.interval import flatten_eval_summary
    from datetime import datetime
    fes = flatten_eval_summary(eval_summary)
    d = dict(name=cond_name, datetime=datetime.now())
    d.update(fes)

    import pandas as pd
    df = pd.DataFrame(d, index=[0])

    from os import path
    from .... import config
    basename = 'turn_detector_evaluation'
    pkl_path = path.join(config.get_package_data_dir(__package__),
                         basename + '.df.pkl')
    xls_path = path.join(config.get_package_data_dir(__package__),
                         basename + '.xlsx')
    if path.exists(pkl_path):
        old_df = pd.read_pickle(pkl_path)
        df = pd.concat([old_df, df], ignore_index=True)
    df.to_pickle(pkl_path)
    df.to_excel(xls_path)

    from ....cloud.google import GoogleDriveInterface
    g = GoogleDriveInterface(read_only=False)
    g.upload(pkl_path,
             path.basename(pkl_path),
             mediaType='application/octet-stream')
    g.upload(xls_path,
             path.basename(xls_path),
             mediaType='application/vnd.ms-excel')


def eval0001(device='cuda'):
    turn_detector, duration_info_list = setup_eval0001(device)

    from .eval import evaluate_turn_detector_duration_infos
    ce, es = evaluate_turn_detector_duration_infos(turn_detector,
                                                   duration_info_list)
    update_evaluation_result('0001', es)
    return ce, es


def setup_eval0002(device='cuda'):
    from .feature_extractor_0002 \
        import TurnDetectorFeatureExtractor0002 as FeatureExtractor
    from .turn_detector_0001 \
        import TurnDetector0001 as TurnDetector
    from .trainer_waseda_0001 \
        import TurnDetectorTrainerWaseda0001 as TurnDetectorTrainer
    from . import base
    from ....corpus.speech.waseda_soma \
        import WASEDA_SOMA, DurationInfoManagerWasedaSoma

    feature_extractor = FeatureExtractor(12,
                                         'csj_0006',
                                         'CSJ0006',
                                         2,
                                         5,
                                         'csj_rwcp_0003',
                                         'CSJRWCP0003',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    base.load_turn_detector(trainer, map_location=device)

    waseda_soma = WASEDA_SOMA()
    id_list_eval = waseda_soma.get_id_list()[310:]
    dim = DurationInfoManagerWasedaSoma(waseda_soma)
    duration_info_list = [dim.get_duration_info(id) for id in id_list_eval]
    return turn_detector, duration_info_list


def eval0002(device='cuda'):
    turn_detector, duration_info_list = setup_eval0002(device)

    from .eval import evaluate_turn_detector_duration_infos
    ce, es = evaluate_turn_detector_duration_infos(turn_detector,
                                                   duration_info_list)

    update_evaluation_result('0002', es)
    return ce, es


def eval0003(device='cuda'):
    turn_detector, _ = setup_eval0001(device)

    from ....corpus.speech.rwcp_spxx \
        import RWCP_SPXX, DurationInfoManagerRWCP
    rwcp = RWCP_SPXX()
    id_list_eval = rwcp.get_id_list()
    dim = DurationInfoManagerRWCP(rwcp)
    duration_info_list = [dim.get_duration_info(id) for id in id_list_eval]

    from .eval import evaluate_turn_detector_duration_infos
    ce, es = evaluate_turn_detector_duration_infos(turn_detector,
                                                   duration_info_list)

    update_evaluation_result('0003', es)
    return ce, es


def eval0004(device='cuda'):
    turn_detector, _ = setup_eval0002(device)

    from ....corpus.speech.rwcp_spxx \
        import RWCP_SPXX, DurationInfoManagerRWCP
    rwcp = RWCP_SPXX()
    id_list_eval = rwcp.get_id_list()
    dim = DurationInfoManagerRWCP(rwcp)
    duration_info_list = [dim.get_duration_info(id) for id in id_list_eval]

    from .eval import evaluate_turn_detector_duration_infos
    ce, es = evaluate_turn_detector_duration_infos(turn_detector,
                                                   duration_info_list)

    update_evaluation_result('0004', es)
    return ce, es


def setup_eval0005(device='cuda'):
    from .feature_extractor_0001 \
        import TurnDetectorFeatureExtractor0001 as FeatureExtractor
    from .turn_detector_0001 \
        import TurnDetector0001 as TurnDetector
    from .trainer_waseda_0001 \
        import TurnDetectorTrainerWaseda0001 as TurnDetectorTrainer
    from . import base
    from ....corpus.speech.waseda_soma \
        import WASEDA_SOMA, DurationInfoManagerWasedaSoma

    feature_extractor = FeatureExtractor(12,
                                         'csj_0008',
                                         'CSJ0008',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    base.load_turn_detector(trainer, map_location=device)

    waseda_soma = WASEDA_SOMA()
    id_list_eval = waseda_soma.get_id_list()[310:]
    dim = DurationInfoManagerWasedaSoma(waseda_soma)
    duration_info_list = [dim.get_duration_info(id) for id in id_list_eval]
    return turn_detector, duration_info_list


def eval0005(device='cuda'):
    turn_detector, duration_info_list = setup_eval0005(device)

    from .eval import evaluate_turn_detector_duration_infos
    ce, es = evaluate_turn_detector_duration_infos(turn_detector,
                                                   duration_info_list)

    update_evaluation_result('0005', es)
    return ce, es


# train0005に対応
def setup_eval0006(device='cuda'):
    from .feature_extractor_0001 \
        import TurnDetectorFeatureExtractor0001 as FeatureExtractor
    from .turn_detector_0001 \
        import TurnDetector0001 as TurnDetector
    from .trainer_waseda_0003 \
        import TurnDetectorTrainerWaseda0003 as TurnDetectorTrainer
    from . import base
    from ....corpus.speech.waseda_soma \
        import WASEDA_SOMA, DurationInfoManagerWasedaSoma

    feature_extractor = FeatureExtractor(12,
                                         'csj_0008',
                                         'CSJ0008',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    base.load_turn_detector(trainer, map_location=device)

    waseda_soma = WASEDA_SOMA()
    id_list_eval = waseda_soma.get_id_list()[310:]
    dim = DurationInfoManagerWasedaSoma(waseda_soma)
    duration_info_list = [dim.get_duration_info(id) for id in id_list_eval]
    return turn_detector, duration_info_list


def eval0006(device='cuda'):
    turn_detector, duration_info_list = setup_eval0006(device)

    from .eval import evaluate_turn_detector_duration_infos
    ce, es = evaluate_turn_detector_duration_infos(turn_detector,
                                                   duration_info_list)

    update_evaluation_result('0006', es)
    return ce, es


# train0006に対応
def setup_eval0007(device='cuda'):
    from .feature_extractor_0002 \
        import TurnDetectorFeatureExtractor0002 as FeatureExtractor
    from .turn_detector_0001 \
        import TurnDetector0001 as TurnDetector
    from .trainer_waseda_0003 \
        import TurnDetectorTrainerWaseda0003 as TurnDetectorTrainer
    from . import base
    from ....corpus.speech.waseda_soma \
        import WASEDA_SOMA, DurationInfoManagerWasedaSoma

    feature_extractor = FeatureExtractor(12,
                                         'csj_0008',
                                         'CSJ0008',
                                         2,
                                         5,
                                         'csj_rwcp_0003',
                                         'CSJRWCP0003',
                                         device=device)
    turn_detector = TurnDetector(feature_extractor, device=device)
    trainer = TurnDetectorTrainer(turn_detector)

    base.load_turn_detector(trainer, map_location=device)

    waseda_soma = WASEDA_SOMA()
    id_list_eval = waseda_soma.get_id_list()[310:]
    dim = DurationInfoManagerWasedaSoma(waseda_soma)
    duration_info_list = [dim.get_duration_info(id) for id in id_list_eval]
    return turn_detector, duration_info_list


def eval0007(device='cuda'):
    turn_detector, duration_info_list = setup_eval0007(device)

    from .eval import evaluate_turn_detector_duration_infos
    ce, es = evaluate_turn_detector_duration_infos(turn_detector,
                                                   duration_info_list)

    update_evaluation_result('0007', es)
    return ce, es





from .base import TurnDetector


def demo(turn_detector: TurnDetector):
    import pyaudio
    import numpy as np

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    is_turn = False
    turn_detector.reset()
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        x = np.fromstring(data, np.int16)
        turn_detector.detach()
        turn = turn_detector.predict([x], out_type='softmax')
        if turn is None:
            continue
        r = turn[0][0].clone().detach().cpu().numpy()
        vad = turn[0][2].clone().detach().cpu().numpy()
        r = np.concatenate([r, vad], axis=1)
        # print(r)
        for n, v, s, nvad, pvad in r[:, :]:
            if not is_turn and v > 0.9:
                is_turn = True
            elif is_turn and v < 0.6:
                is_turn = False

            if is_turn:
                print("!!!! ", end='')
            else:
                print("     ", end='')

            print("TURN {:.3f}".format(v) + ": " + '*' * int(30 * v) +
                  ' ' * int(30 * (1.0 - v)),
                  end='')
            # print("  VAD {:.3f}".format(pvad) + ": " + '*' * int(30 * pvad))
            print("  SHORT {:.3f}".format(s) + ": " + '*' * int(30 * s))


def demo_wav(turn_detector: TurnDetector, wav):
    import numpy as np
    CHUNK = 1600  # 0.1秒ずつくらいで...

    turn_detector.reset()
    is_turn = False
    while wav is not None:
        if len(wav) < CHUNK:
            data = wav
            wav = None
        else:
            data = wav[:CHUNK]
            wav = wav[CHUNK:]
        x = np.fromstring(data, np.int16)
        turn_detector.detach()
        turn = turn_detector.predict([x], out_type='softmax')
        if turn is None:
            continue
        r = turn[0][0].clone().detach().cpu().numpy()
        vad = turn[0][2].clone().detach().cpu().numpy()
        r = np.concatenate([r, vad], axis=1)
        # print(r)
        for n, v, s, nvad, pvad in r[:, :]:
            if not is_turn and v > 0.9:
                is_turn = True
            elif is_turn and v < 0.6:
                is_turn = False

            if is_turn:
                print("!!!! ", end='')
            else:
                print("     ", end='')

            print("TURN {:.3f}".format(v) + ": " + '*' * int(30 * v) +
                  ' ' * int(30 * (1.0 - v)),
                  end='')
            print("  SHORT {:.3f}".format(s) + ": " + '*' * int(30 * s))
