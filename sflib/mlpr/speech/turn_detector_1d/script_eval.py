from .base import construct_turn_detector
from .eval import evaluate_timing_data_and_write
from ....corpus.speech import waseda_soma


def run_0001(device='cuda:0'):
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

    from ....corpus.speech import waseda_soma as ws
    wss = ws.WASEDA_SOMA()
    duration_info_manager = ws.DurationInfoV2ManagerWasedaSoma(wss)
    id_list = wss.get_id_list()
    dataset_name = 'waseda_soma'
    
    # for trainer_number in (13, 14, 15, 16):
    # for trainer_number in (17,):
    # for trainer_number in (14, 18,):
    # for trainer_number in (19,):
    # for trainer_number in (31, 33, 35):
    for trainer_number in (32, 34, 36):
        print("TRAINER NUMBER = {}".format(trainer_number))
        turn_detector = construct_turn_detector(
            turn_detector_number=11,
            trainer_number=trainer_number,
            input_calculator_number=1,
            input_calculator_construct_args=input_calculator_construct_args)
        turn_detector.load()
        turn_detector.to(device)
    
        evaluate_timing_data_and_write(
            turn_detector, duration_info_manager, dataset_name, id_list)
                                   
    

