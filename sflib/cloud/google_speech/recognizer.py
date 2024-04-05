import threading
import time
from .audio_stream import AudioStream
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types


class StreamSpeechRecognizer:
    def __init__(self, audio_stream: AudioStream = None):
        if audio_stream is None:
            self.__audio_stream = AudioStream()
        else:
            self.__audio_stream = audio_stream
            
        # 結果のバッファ
        self.__results = []
        # 音声認識実行中かどうか
        self.__recognizing = False
        # 音声認識が開始された時間
        self.__reco_start_time = None
        # 音声区間が開始された時間（10秒問題のリセットを行った時間）
        self.__reco_seg_start_time = None
        
        self.__cond = threading.Condition()
        self.__thread_running = True
        self.__thread = threading.Thread(target=self.__run)
        self.__thread.start()

    @property
    def recognizing(self):
        return self.__recognizing

    def start_recognition(self):
        with self.__cond:
            # while self.__recognizing:
            #     # print("Recognizer: start_recognition wait for previous recognition finished {}".format(time.time()))
            #     self.__cond.wait()
            self.__reco_start_time = time.time()
            self.__reco_seg_start_time = self.__reco_start_time
            self.__audio_stream.start()
            # print("Recognizer: start_recognition {}".format(time.time()))
            
    def finish_recognition(self):
        with self.__cond:
            self.__audio_stream.finish()
            # while self.__recognizing:
            #     self.__cond.wait()
            # print("Recognizer: finish_recognition {}".format(time.time()))

    def put_audio_data(self, data):
        with self.__cond:
            self.__audio_stream.append(data)

    def get_results(self, flush=False):
        with self.__cond:
            # ガンガン認識回っている際にこれを実行すると
            # ストールするかも...
            if flush:
                while self.__recognizing:
                    self.__cond.wait()
            r = self.__results
            self.__results = []
            return r

    def dispose(self):
        self.__thread_running = False
        # self.__audio_stream.start()
        self.__audio_stream.finish()
        # print("BEFORE JOIN")
        self.__thread.join()
        # print("AFTER JOIN")

    def __run(self):
        import traceback
        while True:
            try:
                client = speech.SpeechClient()
                self.__run_recognition_loop(client)
                return
            except Exception as e:
                print("Exception caught.")
                print(type(e))
                traceback.print_exc()
                break
            # MEMO: google.auth.exceptions.DefaultCredentialsErrorの場合は終了すべき
        
    def __run_recognition_loop(self, client):
        while self.__thread_running:
            # 認識開始 or 終了を待つ
            self.__audio_stream.wait()

            # 認識開始されてなければループを繰り返す．
            # このタイミングでスレッド終了が設定されていれば
            # 終了する
            if not self.__audio_stream.is_started:
                continue

            with self.__cond:
                self.__recognizing = True
                self.__cond.notify_all()

            config = types.RecognitionConfig(
                encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code='ja-JP',
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
                model='default',
                max_alternatives=1,
            )
            streaming_config = types.StreamingRecognitionConfig(
                config=config, interim_results=True)

            # 実際の認識作業を開始する．
            current_audio_id = self.__audio_stream.next_id
            while True:
                requests = (types.StreamingRecognizeRequest(audio_content=chunk)
                            for chunk in self.__audio_stream)
                responses = client.streaming_recognize(
                    streaming_config, requests)
                # print("Recognizer: Result loop in {} {}".format(current_audio_id, time.time()))
                for response in responses:
                    with self.__cond:
                        # print(response.results)
                        # self.__results.extend(list(response.results))
                        for r in response.results:
                            self.__results.append((current_audio_id, r))
                        self.__cond.notify_all()
                # print("Recognizer: Result loop out {} {}".format(current_audio_id, time.time()))
                if self.__audio_stream.next_id > current_audio_id:
                    break
                else:
                    # print("NEW SEGMENT")
                    pass

            # 認識終了を宣言．
            with self.__cond:
                self.__recognizing = False
                self.__cond.notify_all()
                

def print_result(result: types.StreamingRecognitionResult):
    print('Finished: {}'.format(result.is_final))
    print('Stability: {}'.format(result.stability))
    print('End Time: {}'.format(result.result_end_time.seconds +
                                result.result_end_time.nanos * 1e-9))
    alternatives = result.alternatives
    # The alternatives are ordered from most likely to least.
    for alternative in alternatives:
        print('Confidence: {}'.format(alternative.confidence))
        print(u'Transcript: {}'.format(alternative.transcript))
        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time
            print('Word: {}, start_time: {}, end_time: {}'.format(
                word,
                start_time.seconds + start_time.nanos * 1e-9,
                end_time.seconds + end_time.nanos * 1e-9))


def test():
    import pyaudio
    import numpy as np
    import pprint
    
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
    recognizer = StreamSpeechRecognizer()

    count_recognition = 0
    while count_recognition < 2:
        recognizer.start_recognition()
        count = 0
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            # x = np.fromstring(data, np.int16)
            # recognizer.put_audio_data(x)
            recognizer.put_audio_data(data)
            count += 1
            # print(count)
            results = recognizer.get_results()
            if len(results) > 0:
                # pprint.pprint(results)
                for result in results:
                    print_result(result)
            if count > 100:
                break
        recognizer.finish_recognition()
        results = recognizer.get_results(flush=True)
        if len(results) > 0:
            # pprint.pprint(results)
            for result in results:
                print_result(result)
                    
        count_recognition += 1
    recognizer.dispose()
