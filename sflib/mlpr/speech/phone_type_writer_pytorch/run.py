from .base import load_phone_type_writer
from .base import PhoneTypeWriter


def setup_0205ae1206(device='cpu'):
    from .feature_autoencoder_0002 \
        import PhoneTypeWriterFeatureExtractorAutoEncoder0002 as FeatureExtractor
    from .phone_type_writer_0005 \
        import PhoneTypeWriter0005PyTorch as PhoneTypeWriter
    from .trainer_csj_rwcp_0003 \
        import PhoneTypeWriterTrainerCSJRWCP0003 as PhoneTypeWriterTrainer
    feature_extractor = FeatureExtractor(12, 'csj_0006', 'CSJ0006')
    phone_type_writer = PhoneTypeWriter(feature_extractor, device=device)
    trainer = PhoneTypeWriterTrainer(phone_type_writer)

    load_phone_type_writer(trainer, map_location=device)

    return phone_type_writer

def setup_0205ae1208(device='cpu'):
    from .feature_autoencoder_0002 \
        import PhoneTypeWriterFeatureExtractorAutoEncoder0002 as FeatureExtractor
    from .phone_type_writer_0005 \
        import PhoneTypeWriter0005PyTorch as PhoneTypeWriter
    from .trainer_csj_rwcp_0003 \
        import PhoneTypeWriterTrainerCSJRWCP0003 as PhoneTypeWriterTrainer
    feature_extractor = FeatureExtractor(12, 'csj_0008', 'CSJ0008')
    phone_type_writer = PhoneTypeWriter(feature_extractor, device=device)
    trainer = PhoneTypeWriterTrainer(phone_type_writer)

    load_phone_type_writer(trainer, map_location=device)

    return phone_type_writer


def demo(phone_type_writer: PhoneTypeWriter):
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
    
    phone_type_writer.reset()
    is_sp = True
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        x = np.fromstring(data, np.int16)
        phone_list = phone_type_writer.predict([x])
        for p in phone_list[0]:
            if p == 'nil':
                # print('_', end='', flush=True)
                continue
            if p == 'sp':
                if not is_sp:
                    print('')
                    is_sp = True
                continue
            print(p + ' ', end='', flush=True)
            is_sp = False
        phone_type_writer.detach()
