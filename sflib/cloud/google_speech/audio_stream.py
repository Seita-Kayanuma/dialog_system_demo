import threading
import time
import wave


class AudioStream(object):
    """音声認識用に音声区間のストリームをiterateするクラス．
    """
    def __init__(self):
        self.__cond = threading.Condition()
        self.__started = False

        # self._buff は，(id, data) のリスト
        # id は int, data はバイト列
        self._buff = []
        
        # 終了済区間のカウンタ（次の区間のID）
        self.__count = 0

        # 次に読み出す区間のID
        self.__next_id = 1

        # セグメント開始時の時間
        self.__segment_start_time = None

        # デバッグのための書き出し
        self._wf = None

    # ------------------
    @property
    def is_started(self):
        """開始しているかどうか"""
        return self.__started

    @property
    def count(self):
        """送信済の区間数"""
        return self.__count

    @property
    def next_id(self):
        """次に読み出す区間のID"""
        return self.__next_id

    # ------------------
    def start(self):
        """区間開始を報告"""
        with self.__cond:
            # --- デバグのための書き出し準備
            w = wave.Wave_write("{:04d}.wav".format(self.count))
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            self._wf = w
            # ---
            self.__started = True
            self.__segment_start_time = time.time()
            self.__cond.notify_all()

    def finish(self):
        """区間終了を報告"""
        with self.__cond:
            if not self.__started:
                # print("WARNING: not started")
                self.__cond.notify_all()
                return

            # ---
            self._wf.close()
            # ---

            # print("AudioStream: finish {} {} ({})".format(
            #     self.count, time.time(), len(self._buff)))
            self._buff.append((self.__count, None))
            self.__started = False
            self.__count += 1
            self.__cond.notify_all()
                
    def append(self, data):
        """音声データを追加"""
        with self.__cond:
            if not self.__started:
                raise RuntimeError('stream is not started')
            # ---
            if self._wf is not None:
                self._wf.writeframes(data)
            # ---
            self._buff.append((self.__count, data))
            self.__cond.notify_all()

    # ------------------
    def wait(self):
        """状態変化を待つ．
        区間開始，区間終了，音声追加が発生すると終了する．
        """
        with self.__cond:
            self.__cond.wait()
    
    # ------------------
    def __iter__(self):
        return self

    def __next__(self):
        with self.__cond:
            # データがあればそれを読み出す
            if len(self._buff) > 0:
                id, data = self._buff[0]
                self._buff = self._buff[1:]
                if data is None:
                    self.__next_id += 1
                    raise StopIteration()
                else:
                    # print(data)
                    return data

            # セグメント開始後の経過時間
            segment_elapsed_time = time.time() - self.__segment_start_time
            # データが読み込まれるか，タイムアウトまで待つ
            while len(self._buff) == 0:
                timeout = 9.0 - segment_elapsed_time
                self.__cond.wait(timeout)
                segment_elapsed_time = time.time() - self.__segment_start_time
                if segment_elapsed_time > 9.0:
                    break
                
            # セグメント開始から9秒経っていたら，一旦 StopIteration を発生させる
            if segment_elapsed_time > 9.0:
                self.__segment_start_time = time.time()
                raise StopIteration()

            id, data = self._buff[0]
            self._buff = self._buff[1:]
            if data is None:
                self.__next_id += 1
                raise StopIteration()
            else:
                # print(data)
                return data
