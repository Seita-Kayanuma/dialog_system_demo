from .base import construct_turn_detector
import datetime


def train(turn_detector, device, train_args=None):
    turn_detector.to(device)
    if turn_detector.get_latest_model_version() is not None:
        turn_detector.load()
    t_args, t_kwargs = [], {}
    if isinstance(train_args, tuple):
        t_args, t_kwargs = train_args
    elif isinstance(train_args, list):
        t_args = train_args
    elif isinstance(train_args, dict):
        t_kwargs = train_args
    turn_detector.train_turn_detector(*t_args, **t_kwargs)
    turn_detector.save(upload=True)
    turn_detector.upload_csv_log()

# --- 2020.2.29 002 ---
# 14系，18系，それぞれで0.6ターゲットにして動かす
def run20200229002_base(device, trainer_numbers):
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
    for trainer_number in trainer_numbers:
        print("*** TRAINER NUMBER = {} ***".format(trainer_number))
        turn_detector = construct_turn_detector(
            turn_detector_number=11,
            trainer_number=trainer_number,
            input_calculator_number=1,
            input_calculator_construct_args=input_calculator_construct_args)
        train(turn_detector, device)
    
def run20200229002_001():
    run20200229002_base('cuda:1', [31, 32])

def run20200229002_002():
    run20200229002_base('cuda:2', [33, 34])

def run20200229002_003():
    run20200229002_base('cuda:3', [35, 36])

# --- 2020.2.29 001 ---
def trainTD1D11T18IC01VAD01T01F03SIAE12T13V02():
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
        trainer_number=18,
        input_calculator_number=1,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:0')

    
def trainTD1D11T19IC01VAD01T01F03SIAE12T13V02():
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
        trainer_number=19,
        input_calculator_number=1,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:0')
    
# --- 2020.2.28 001 ---
def trainTD1D11T17IC01VAD01T01F03SIAE12T13V02():
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
        trainer_number=17,
        input_calculator_number=1,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:0')
    

# --- 仮です ---    
def trainTD1D21T98IC21VAD11T12F03SIAE12T13V02():
    vad_feature_extractor_construct_args = dict(
        autoencoder_number=12,
        autoencoder_trainer_number=13,
        autoencoder_model_version=2)
    input_calculator_construct_args = dict(
        voice_activity_detector_number=11,
        voice_activity_detector_trainer_number=12,
        voice_activity_detector_feature_extractor_number=3,
        voice_activity_detector_feature_extractor_construct_args=vad_feature_extractor_construct_args,
    )
    turn_detector = construct_turn_detector(
        turn_detector_number=21,
        trainer_number=98,
        input_calculator_number=21,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:1')

    
# --- 2020.2.24 001 ---
def trainTD1D11T16IC01VAD01T01F03SIAE12T13V02():
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
    train(turn_detector, 'cuda:0')

    
def trainTD1D21T13IC21VAD11T12F03SIAE12T13V02():
    vad_feature_extractor_construct_args = dict(
        autoencoder_number=12,
        autoencoder_trainer_number=13,
        autoencoder_model_version=2)
    input_calculator_construct_args = dict(
        voice_activity_detector_number=11,
        voice_activity_detector_trainer_number=12,
        voice_activity_detector_feature_extractor_number=3,
        voice_activity_detector_feature_extractor_construct_args=vad_feature_extractor_construct_args,
    )
    turn_detector = construct_turn_detector(
        turn_detector_number=21,
        trainer_number=13,
        input_calculator_number=21,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:1')

    
def trainTD1D21T14IC21VAD11T12F03SIAE12T13V02():
    vad_feature_extractor_construct_args = dict(
        autoencoder_number=12,
        autoencoder_trainer_number=13,
        autoencoder_model_version=2)
    input_calculator_construct_args = dict(
        voice_activity_detector_number=11,
        voice_activity_detector_trainer_number=12,
        voice_activity_detector_feature_extractor_number=3,
        voice_activity_detector_feature_extractor_construct_args=vad_feature_extractor_construct_args,
    )
    turn_detector = construct_turn_detector(
        turn_detector_number=21,
        trainer_number=14,
        input_calculator_number=21,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:1')

    
def trainTD1D21T16IC21VAD11T12F03SIAE12T13V02():
    vad_feature_extractor_construct_args = dict(
        autoencoder_number=12,
        autoencoder_trainer_number=13,
        autoencoder_model_version=2)
    input_calculator_construct_args = dict(
        voice_activity_detector_number=11,
        voice_activity_detector_trainer_number=12,
        voice_activity_detector_feature_extractor_number=3,
        voice_activity_detector_feature_extractor_construct_args=vad_feature_extractor_construct_args,
    )
    turn_detector = construct_turn_detector(
        turn_detector_number=21,
        trainer_number=16,
        input_calculator_number=21,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:3')

    
# --- 2020.2.23 001 ---
def trainTD1D11T14IC01VAD01T01F03SIAE12T13V02():
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
        trainer_number=14,
        input_calculator_number=1,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:1')

    
def trainTD1D11T15IC01VAD01T01F03SIAE12T13V02():
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
        trainer_number=15,
        input_calculator_number=1,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:2')

    
# --- 2020.2.22 001 ---
def trainTD1D11T13IC01VAD01T01F03SIAE12T13V02():
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
        trainer_number=13,
        input_calculator_number=1,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:1')


# --- 2020.2.20 001 ---
def trainTD1D01T03IC01VAD01T01F03SIAE12T13V02():
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
        turn_detector_number=1,
        trainer_number=3,
        input_calculator_number=1,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:2')

    
def trainTD1D01T04IC01VAD01T01F03SIAE12T13V02():
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
        turn_detector_number=1,
        trainer_number=4,
        input_calculator_number=1,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:3')

    
def trainTD1D02T03IC01VAD01T01F03SIAE12T13V02():
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
        turn_detector_number=2,
        trainer_number=3,
        input_calculator_number=1,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:0')

    
def trainTD1D11T03IC01VAD01T01F03SIAE12T13V02():
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
        trainer_number=3,
        input_calculator_number=1,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:1')

    
def trainTD1D11T04IC01VAD01T01F03SIAE12T13V02():
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
        trainer_number=4,
        input_calculator_number=1,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:1')

    
# --- 2020.2.19 002 ---
def trainTD1D01T02IC01VAD01T01F03SIAE12T13V02():
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
        turn_detector_number=1,
        trainer_number=2,
        input_calculator_number=1,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:1')

    
# --- 2020.2.19 001 ---
def trainTD1D01T99IC01VAD01T01F03SIAE12T13V02():
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
        turn_detector_number=1,
        trainer_number=99,
        input_calculator_number=1,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:1')

    
# --- 2020.2.18 001 ---
def trainTD1D01T01IC01VAD01T01F03SIAE12T13V02():
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
        turn_detector_number=1,
        trainer_number=1,
        input_calculator_number=1,
        input_calculator_construct_args=input_calculator_construct_args)
    train(turn_detector, 'cuda:0')
          

