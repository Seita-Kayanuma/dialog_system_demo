from .base import PhoneTypeWriter


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
        if phone_list is None:
            continue
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
