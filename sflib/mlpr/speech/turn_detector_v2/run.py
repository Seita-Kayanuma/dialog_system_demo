from .base import TurnDetector


def demo(turn_detector: TurnDetector):
    import pyaudio
    import numpy as np
    import time

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
    start_time = time.time()
    total_frame = 0
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
        total_frame += r.shape[0]
        total_time = time.time() - start_time
        fps = total_frame / total_time
        # print(r)
        for n, v, s, nvad, pvad in r[:, :]:
            v = v
            if not is_turn and v > 0.9:
                is_turn = True
            elif is_turn and v < 0.6:
                is_turn = False

            print("{:6.2f}fps ".format(fps), end='')
                
            if is_turn:
                print("!!!! ", end='')
            else:
                print("     ", end='')
                
            print("T+S {:.3f}".format(v) + ": " +
                  'T' * int(30 * v) +
                  'S' * int(30 * s) +
                  ' ' * (30 - int(30 * v) - int(30 * s)),
                  end='')
            print("  VAD {:.3f}".format(pvad) + ": " + '*' * int(30 * pvad))
            # print("  SHORT {:.3f}".format(s) + ": " + '*' * int(30 * s))


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


def demo2X(turn_detector: TurnDetector):
    import pyaudio
    import numpy as np
    import time

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
    start_time = time.time()
    total_frame = 0
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        x = np.fromstring(data, np.int16)
        turn_detector.detach()
        result = turn_detector.predict([x], out_type='softmax')
        if result is None:
            continue
        td = result[0][0].clone().detach().cpu().numpy()
        alpha = result[0][1].clone().detach().cpu().numpy()
        vad = result[0][2].clone().detach().cpu().numpy()
        r = np.concatenate([td, alpha, vad], axis=1)
        total_frame += r.shape[0]
        total_time = time.time() - start_time
        fps = total_frame / total_time
        # print(r)
        for td, alpha, vad in r[:, :]:
            print("{:6.2f}fps ".format(fps), end='')
                
            print("{: 6.2f}[s] ".format(td), end='')
            if td < 0:
                width = int(min(2.0,  -td) / 2.0 * 15)
                print(' ' * int(15 - width) + '*' * width + '|', end='')
                print(' ' * 15, end='')
            else:
                width = int(min(2.0,  td) / 2.0 * 15)
                print(' ' * 15 + '|', end='')
                print('*' * int(width) + ' ' * (15-width), end='')
            print("{:5.2f} ".format(alpha), end='')
            print("{:4.2f} ".format(vad), end='')
            width = int(vad * 30)
            print('*' * width, end='')
            print("")
            

def demo22(turn_detector: TurnDetector):
    import pyaudio
    import numpy as np
    import time
    from torch.nn.utils.rnn import pad_packed_sequence
    import torch
    
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
    start_time = time.time()
    total_frame = 0
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        x = np.fromstring(data, np.int16)
        turn_detector.detach()
        result = turn_detector.predict([x], out_type='raw')
        if result is None:
            continue
        padded_result, result_len = pad_packed_sequence(result)
        result = []
        for b in range(padded_result.shape[1]):
            batch_result = []
            batch_result_len = result_len[b]

            y_dt = padded_result[:batch_result_len, b, 0:1]
            y_alpha = padded_result[:batch_result_len, b, 1:2]
            y_vad = padded_result[:batch_result_len, b, 2:3]

            batch_result.append(y_dt)
            batch_result.append(1.0 / torch.exp(y_alpha))
            batch_result.append(torch.sigmoid(y_vad))
            
            result.append(batch_result)
        
        y_dt = result[0][0].clone().detach().cpu().numpy()
        y_alpha = result[0][1].clone().detach().cpu().numpy()
        y_vad = result[0][2].clone().detach().cpu().numpy()
        r = np.concatenate([y_dt, y_alpha, y_vad], axis=1)
        total_frame += r.shape[0]
        total_time = time.time() - start_time
        fps = total_frame / total_time
        # print(r)
        for td, alpha, vad in r[:, :]:
            print("{:6.2f}fps ".format(fps), end='')
                
            print("{: 6.2f}[s] ".format(td), end='')
            if td < 0:
                width = int(min(2.0,  -td) / 2.0 * 15)
                print(' ' * int(15 - width) + '*' * width + '|', end='')
                print(' ' * 15, end='')
            else:
                width = int(min(2.0,  td) / 2.0 * 15)
                print(' ' * 15 + '|', end='')
                print('*' * int(width) + ' ' * (15-width), end='')
            print("{:5.2f} ".format(alpha), end='')
            print("{:4.2f} ".format(vad), end='')
            width = int(vad * 30)
            print('*' * width, end='')
            print("")
            
            
def demo24(turn_detector: TurnDetector):
    import pyaudio
    import numpy as np
    import time
    from torch.nn.utils.rnn import pad_packed_sequence
    import torch
    
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
    start_time = time.time()
    total_frame = 0
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        x = np.fromstring(data, np.int16)
        turn_detector.detach()
        result = turn_detector.predict([x], out_type='raw')
        if result is None:
            continue
        padded_result, result_len = pad_packed_sequence(result)
        result = []
        for b in range(padded_result.shape[1]):
            batch_result = []
            batch_result_len = result_len[b]

            y_dt = padded_result[:batch_result_len, b, 0:1]
            y_alpha = padded_result[:batch_result_len, b, 1:2]
            y_vad = padded_result[:batch_result_len, b, 2:3]

            batch_result.append(y_dt /(torch.exp(-y_alpha) + 0.1))
            batch_result.append(torch.exp(-y_alpha))
            batch_result.append(torch.sigmoid(y_vad))
            
            result.append(batch_result)
        
        y_dt = result[0][0].clone().detach().cpu().numpy()
        y_alpha = result[0][1].clone().detach().cpu().numpy()
        y_vad = result[0][2].clone().detach().cpu().numpy()
        r = np.concatenate([y_dt, y_alpha, y_vad], axis=1)
        total_frame += r.shape[0]
        total_time = time.time() - start_time
        fps = total_frame / total_time
        # print(r)
        for td, alpha, vad in r[:, :]:
            print("{:6.2f}fps ".format(fps), end='')
                
            print("{: 6.2f}[s] ".format(td), end='')
            if td < 0:
                width = int(min(10.0,  -td) / 10.0 * 15)
                print(' ' * int(15 - width) + '*' * width + '|', end='')
                print(' ' * 15, end='')
            else:
                width = int(min(10.0,  td) / 10.0 * 15)
                print(' ' * 15 + '|', end='')
                print('*' * int(width) + ' ' * (15-width), end='')
            print("{:5.2f} ".format(alpha), end='')
            print("{:4.2f} ".format(vad), end='')
            width = int(vad * 30)
            print('*' * width, end='')
            print("")
            
            
