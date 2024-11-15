import curses
import threading
import numpy as np
import time
import psutil
from queue import Queue
from enum import Enum
from dataclasses import dataclass

from .asr import ASR
from .audio import AbsAudio, AudioData
from .vad_silero import VAD, VADState

# PyTorchのスレッド数を制限
import torch
torch.set_num_threads(1)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def make_progress_bar(step, total_steps, bar_width=60):
    """
    プログレスバーを作成する

    Args:
        step (int): 現在のステップ数
        total_steps (int): 総ステップ数
        bar_width (int, optional): プログレスバーの幅（デフォルトは60）

    Returns:
        str: プログレスバーの文字列
    """
    # UTF-8の左側のブロック: 1, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8
    utf_8s = ["█", "▏", "▎", "▍", "▌", "▋", "▊", "█"]
    perc = 100 * float(step) / float(total_steps)
    max_ticks = bar_width * 8
    num_ticks = int(round(perc / 100 * max_ticks))
    full_ticks = num_ticks // 8      # フルブロックの数
    part_ticks = num_ticks % 8      # 部分ブロックのサイズ（配列のインデックス）

    bar = ""                 # 変数を初期化
    bar += utf_8s[0] * int(full_ticks)  # プログレスバーにフルブロックを追加

    # 部分ブロックが0でない場合、部分文字を追加
    if part_ticks > 0:
        bar += utf_8s[part_ticks]

    # プログレスバーを埋めるためにfill文字を追加
    bar += "▒" * int((max_ticks/8 - float(num_ticks)/8.0))
    return bar

class RotateBar:
    def __init__(self, step=5):
        self._bars = ['|', '/', '-', '\\']
        self._count = 0
        self.step = step

    def get(self):
        return self._bars[(self._count // self.step) % 4]

    def next(self):
        self._count += 1
        # overflow guard
        self._count = self._count % (4 * self.step)
        return self.get()

    def reset(self):
        self._count = 0


class TimeSeriesVirticalBar:
    def __init__(self, max_len, max_value=100, min_value=0):
        self.bars = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        self.max_len = max_len
        self.max_value = max_value
        self.min_value = min_value
        self.values = []

    def append(self, value):
        self.values.append(value)
        if len(self.values) > self.max_len:
            self.values.pop(0)

    def get_string(self):
        if len(self.values) < self.max_len:
            values = self.values + [self.min_value] * (self.max_len - len(self.values))
        else:
            values = self.values
        result = ''
        for value in values:
            if value > self.max_value:
                value = self.max_value
            if value < self.min_value:
                value = self.min_value
            index = int((value - self.min_value) / (self.max_value - self.min_value) * 8)
            result += self.bars[index]
        return result


class StatusWindow:
    def __init__(self, window):
        self.window = window
        # ASRの状態
        self.asr_is_active = True
        # 音声データの状態
        self.audio_power_db_max = -10   # 描画上の最大値
        self.audio_power_db_min = -100  # 描画上の最小値
        self.audio_power_db = self.audio_power_db_min # 現在の音声データのパワー
        # VADの状態（音声区間中は回る）
        self.vad_rotate_bar = RotateBar()

        # 入力デバイス名
        self.device_name = ""
        # CPU使用率表示用のバー
        self.cpu_bar = TimeSeriesVirticalBar(10, max_value=100, min_value=0)
        # メモリ使用量
        self.memory_in_gb = 0.0

    def set_asr_is_active(self, is_active: bool):
        self.set_asr_is_active = is_active

    def update_audio_power(self, audio_data: AudioData):
        p = np.power(audio_data.data_np, 2).mean()
        if p > 0:
            p_db = 10 * np.log10(p)
        else:
            p_db = -100
        self.audio_power = p_db

    def step_vad_rotate_bar(self):
        self.vad_rotate_bar.next()

    def set_device_name(self, device_name):
        self.device_name = device_name

    def update_cpu_percent(self, cpu_percent):
        self.cpu_bar.append(cpu_percent)

    def update_memory_in_gb(self, memory_in_gb):
        self.memory_in_gb = memory_in_gb

    def draw(self):
        self.window.erase()

        self.window.attron(curses.A_REVERSE)
        if self.asr_is_active:
            self.window.addstr("[ON]  ")
        else:
            self.window.addstr("[OFF] ")

        p_db_percent = (self.audio_power_db - self.audio_power_db_min) / (self.audio_power_db_max - self.audio_power_db_min)
        p_db_percent = max(0, min(1, p_db_percent))
        total_steps = 100
        bar = make_progress_bar(p_db_percent * total_steps, total_steps, bar_width=20)
        self.window.addstr(f"{bar} ")

        self.window.addstr(self.vad_rotate_bar.next() + " ")

        self.window.addstr("Device: ")
        self.window.addstr(self.device_name + " ")

        self.window.addstr(f"CPU: {self.cpu_bar.get_string()} ")

        self.window.addstr(f"Memory: {self.memory_in_gb:4.2f}GB ")

        self.window.addstr(" " * (self.window.getmaxyx()[1] - self.window.getyx()[1] - 1))

        self.window.refresh()

def draw_kana(window: curses.window, kana: str):
    text = kana
    text = text.replace('<mask>', '▒' * 3)
    while text.startswith('<sp>'):
        text = text[5:]
    text = text.replace('<sp>', '、')
    mode = 0
    current_text = ''
    for token in text.split():
        para = 'N'
        prev_mode = mode
        if '+' in token:
            token, para = token.split('+')
        if para == 'N':
            mode = 0
        elif para == 'F':
            mode = 1
        elif para == 'D':
            mode = 2
        if mode != prev_mode:
            if prev_mode > 0:
                window.addstr(current_text, curses.color_pair(prev_mode))
            else:
                window.addstr(current_text)
            current_text = ''
        current_text += token
        prev_mode = mode
    if mode > 0:
        window.addstr(current_text, curses.color_pair(mode))
    else:
        window.addstr(current_text)

def draw_kanji(window, kanji: str):
    text = kanji
    text = text.replace('<mask>', '▒' * 3)
    text = text.replace(' ', '')
    text = text.replace('<sp>', '、')
    current_text = ''
    mode = 0
    while len(text) > 0:
        prev_mode = mode
        if text[:3] == '<F>':
            mode = 1
            text = text[3:]
        elif text[:4] == '</F>':
            mode = 0
            text = text[4:]
        if mode != prev_mode:
            if prev_mode > 0:
                window.addstr(current_text, curses.color_pair(prev_mode) | curses.A_BOLD)
            else:
                window.addstr(current_text, curses.A_BOLD)
            current_text = ''
            prev_mode = mode
            continue
        if len(text) > 0:
            current_text += text[0]
            text = text[1:]
    if mode > 0:
        window.addstr(current_text, curses.color_pair(mode) | curses.A_BOLD)
    else:
        window.addstr(current_text, curses.A_BOLD)


class ASRCurrentResultWindow:
    def __init__(self, parent_window,
                 nlines_kana, nlines_kanji, ncols, y, x):
        assert ncols > 20, "ncols must be greater than 20"
        self.window = parent_window.derwin(nlines_kana + nlines_kanji + 3, ncols, y, x)
        self.kana_window = self.window.derwin(nlines_kana, ncols - 16, 1, 15)
        self.kana_window.scrollok(True)
        self.kanji_window = self.window.derwin(nlines_kanji, ncols - 16, nlines_kana + 2, 15)
        self.kanji_window.scrollok(True)

        self.nlines_kana = nlines_kana
        self.nlines_kanji = nlines_kanji

        self.kana = None
        self.kanji = None
        self.processed_time = 0.0
        self.asr_time = 0.0
        self.kana_kanji_time = 0.0

    def update(self, kana: str, kanji: str,
               processed_time: float,
               asr_time: float,
               kana_kanji_time: float = None):
        self.kana = kana
        self.kanji = kanji
        self.processed_time = processed_time
        self.asr_time = asr_time
        self.kana_kanji_time = kana_kanji_time

    def draw(self):
        self._draw()
        self.window.noutrefresh()

    def _draw_frame(self):
        self.window.border()
        maxy, maxx = self.window.getmaxyx()
        self.window.addch(0, 7,  curses.ACS_TTEE)
        self.window.addch(0, 14, curses.ACS_TTEE)
        self.window.addch(self.nlines_kana + 1, 0,  curses.ACS_LTEE)
        self.window.addch(self.nlines_kana + 1, 7,  curses.ACS_PLUS)
        self.window.addch(self.nlines_kana + 1, 14, curses.ACS_PLUS)
        self.window.addch(self.nlines_kana + 1, maxx - 1, curses.ACS_RTEE)
        self.window.addch(maxy - 1, 7,  curses.ACS_BTEE)
        self.window.addch(maxy - 1, 14, curses.ACS_BTEE)
        for x in range(1, 7):
            self.window.addch(self.nlines_kana + 1, x, curses.ACS_HLINE)
        for x in range(8, 14):
            self.window.addch(self.nlines_kana + 1, x, curses.ACS_HLINE)
        for x in range(15, maxx-1):
            self.window.addch(self.nlines_kana + 1, x, curses.ACS_HLINE)
        for y in range(1, self.nlines_kana + 1):
            self.window.addch(y, 7, curses.ACS_VLINE)
            self.window.addch(y, 14, curses.ACS_VLINE)
        for y in range(self.nlines_kana + 2, maxy - 1):
            self.window.addch(y, 7, curses.ACS_VLINE)
            self.window.addch(y, 14, curses.ACS_VLINE)

    def _draw(self):
        self.window.erase()
        self._draw_frame()
        if self.kana is not None:
            self.window.addstr(1, 1, f"{self.processed_time:6.3f}")
            self.window.addstr(1, 8, f"{self.asr_time:6.3f}")
            self._draw_kana()
        if self.kanji is not None:
            self.window.addstr(self.nlines_kana + 2, 8, f"{self.kana_kanji_time:6.3f}")
            self._draw_kanji()

    def _draw_kana(self):
        self.kana_window.erase()
        self.kana_window.move(0, 0)
        draw_kana(self.kana_window, self.kana)

    def _draw_kanji(self):
        self.kanji_window.erase()
        self.kanji_window.move(0, 0)
        draw_kanji(self.kanji_window, self.kanji)

class ASRResultHistoryWindow:
    def __init__(self, parent_window, nlines, ncols, y, x):
        self.window = parent_window.derwin(nlines, ncols, y, x)
        self.kanakan_window = self.window.derwin(nlines, ncols - 16, 0, 15)
        self.nlines = nlines
        self.ncols = ncols
        self.history = []

    def append(self, kana: str, kanji: str, processed_time: float, asr_time: float, kana_kanji_time: float):
        self.history.insert(0, (kana, kanji, processed_time, asr_time, kana_kanji_time))

    def draw(self):
        self.window.erase()
        self.window.move(0, 0)

        y = 0
        final_index = None
        for index in range(len(self.history)):
            kana, kanji, processed_time, asr_time, kana_kanji_time = self.history[index]
            try:
                self.window.addstr(y, 1, f"{processed_time:6.3f}")
                self.window.addstr(y, 8, f"{asr_time:6.3f}")
                self.kanakan_window.move(y, 0)
                draw_kana(self.kanakan_window, kana)
                y = self.kanakan_window.getyx()[0] + 1
                self.window.addstr(y, 8, f"{kana_kanji_time:6.3f}")
                self.kanakan_window.move(y, 0)
                draw_kanji(self.kanakan_window, kanji)
                y = self.kanakan_window.getyx()[0] + 1
            except curses.error:
                final_index = index
                break
        if final_index is not None and final_index < len(self.history) - 1:
            self.history = self.history[:final_index + 1]
        self.window.noutrefresh()

class MainASRState(Enum):
    Idle = 0
    Started = 1




@dataclass
class VADData:
    audio_data: AudioData
    vad_state: VADState

class ASRWorker:
    def __init__(self,
                 asr: ASR,
                 current_result_window: ASRCurrentResultWindow,
                 result_history_window: ASRResultHistoryWindow,
                 audio: AbsAudio,
                 lock_scr: threading.Condition,):
        self.audio = audio
        self.asr = asr
        self.current_result_window = current_result_window
        self.result_history_window = result_history_window
        self.lock_scr = lock_scr

        self.is_active = True

        self.asr_state = MainASRState.Idle
        self.asr_start_time = None

        self.queue = Queue()

    def _get_current_time(self):
        return self.audio.current_time()

    def setActive(self, active: bool):
        self.is_active = active

    def put(self, data: VADData):
        self.queue.put(data)

    def run(self):
        while True:
            data = self.queue.get()

            audio_data = data.audio_data
            vad_state = data.vad_state
            if self.asr_state == MainASRState.Idle:
                processed_time = 0.0
            else:
                processed_time = audio_data.time - self.asr_start_time + len(audio_data.data_np) / 16000

            # アクティブ状態でVADがStartされた場合は音声認識を開始．
            # 非アクティブ状態の場合は，開始はしない．
            if self.is_active and vad_state == VADState.Started:
                self.asr_start_time = audio_data.time
                self.asr_state = MainASRState.Started

                with self.lock_scr:
                    self.current_result_window.update(None, None, 0.0, 0.0)
                    self.current_result_window.draw()
                    self.result_history_window.draw()

            if self.asr_state == MainASRState.Started:
                # 認識中であれば，とりあえず音声認識は行う

                # 終了判定は，VADがEnded，もしくは非アクティブ状態になった場合
                is_final = vad_state == VADState.Ended or not self.is_active
                kana = self.asr.recognize(audio_data.data_np, is_final=is_final)
                asr_time = self._get_current_time() - self.asr_start_time

                if kana is not None:
                    if not is_final:
                        with self.lock_scr:
                            self.current_result_window.update(
                                kana, None, processed_time, asr_time)
                            self.current_result_window.draw()
                    else:
                        kana_kanji_start_time = self._get_current_time()
                        kanji = self.asr.convert_kana_to_text(kana)
                        kana_kanji_time = self._get_current_time() - kana_kanji_start_time

                        with self.lock_scr:
                            self.current_result_window.update(
                                kana, kanji, processed_time, asr_time, kana_kanji_time)
                            self.current_result_window.draw()

                        # 発話時間が1秒未満で，認識結果が3文字未満のものは無視
                        if not (processed_time < 1.0 and len(kana.split(' ')) < 3):
                            self.result_history_window.append(
                                kana, kanji, processed_time, asr_time, kana_kanji_time)

                        self.asr_state = MainASRState.Idle

            with self.lock_scr:
                curses.doupdate()


class Main:
    def __init__(self,
                 audio: AbsAudio,
                 vad: VAD,
                 asr: ASR):
        self.asr = asr
        self.audio = audio
        self.vad = vad

        self.cond = threading.Condition()
        self.stdscr = None

        self.asr_result_window = None
        self.asr_result_windows = []
        self.asr_start_time = None

        # self.rotate_bar = RotateBar()

        self.asr_worker = None

    def _get_current_time(self):
        return self.audio.current_time()

    def _vad_callback(self, data: AudioData, state: VADState):
        assert self.asr_worker is not None
        self.asr_worker.put(VADData(data, state))
        self.vad_state = state

    def _run_monitor_cpu_percent(self):
        while True:
            cpu_percent = psutil.Process().cpu_percent(interval=1)
            with self.cond:
                # self.cpu_percent = cpu_percent
                self.status_window.update_cpu_percent(cpu_percent)
                # self.cpu_bar.append(cpu_percent)
            time.sleep(1)

    def run(self, stdscr):
        # self.cpu_percent = 0.0
        # self.cpu_bar = TimeSeriesVirticalBar(10, max_value=100, min_value=0)

        self._monitor_cpu_thread = threading.Thread(target=self._run_monitor_cpu_percent)
        self._monitor_cpu_thread.daemon = True
        self._monitor_cpu_thread.start()

        self.asr_state = MainASRState.Idle
        self.vad_state = VADState.Idle

        self.audio.add_callback(self.vad.process)
        self.vad.add_callback(self._vad_callback)
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_RED)
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLUE)
        curses.start_color()
        curses.curs_set(0)

        self.stdscr = stdscr
        self.stdscr.refresh()
        h, w = self.stdscr.getmaxyx()
        self.screen_height = h
        self.screen_width = w

        ### main window
        self.main_window = self.stdscr.derwin(h-1, w, 0, 0)
        nlines_kana = 2
        nlines_kanji = 2
        self.asr_result_window = ASRCurrentResultWindow(
            self.main_window, nlines_kana, nlines_kanji, self.screen_width, 0, 0)
        self.asr_result_history_window = ASRResultHistoryWindow(
            self.main_window,
            h-1-(nlines_kana+nlines_kanji+3), self.screen_width,
            nlines_kana+nlines_kanji+3, 0)
        self.status_window = StatusWindow(
            self.stdscr.derwin(1, w, h-1, 0))
        self.status_window.set_device_name(self.audio.get_device_name())

        self.asr_result_window.draw()
        self.stdscr.refresh()

        self.asr_worker = ASRWorker(
            self.asr,
            self.asr_result_window,
            self.asr_result_history_window,
            self.audio,
            self.cond)
        self.asr_worker_thread = threading.Thread(target=self.asr_worker.run)
        self.asr_worker_thread.daemon = True
        self.asr_worker_thread.start()

        ### status window
        # self.status_window = self.stdscr.derwin(1, w, h-1, 0)
        # self.status_on = True
        def audio_callback_(data: AudioData):
            with self.cond:
                # self.status_window.set_asr_is_active(self.asr_worker.is_active)
                self.status_window.update_audio_power(data)
                if self.vad_state == VADState.Started or self.vad_state == VADState.Continue:
                    self.status_window.step_vad_rotate_bar()
                # self.status_window.update_cpu_percent(self.cpu_percent)
                self.status_window.update_memory_in_gb(
                    psutil.Process().memory_info().rss / (1024 ** 3))
                self.status_window.draw()

            # with self.cond:
            #     self.status_window.erase()
            #     self.status_window.attron(curses.A_REVERSE)
            #     if self.asr_worker.is_active:
            #         self.status_window.addstr(0, 0, "[ON]  ")
            #     else:
            #         self.status_window.addstr(0, 0, "[OFF] ")

            #     p = np.power(data.data_np, 2).mean()
            #     if p > 0:
            #         p_db = 10 * np.log10(p)
            #     else:
            #         p_db = -100
            #     p_db_min, p_db_max = -100, -10
            #     p_db_percent = (p_db - p_db_min) / (p_db_max - p_db_min)
            #     p_db_percent = max(0, min(1, p_db_percent))
            #     total_steps = 100
            #     bar = make_progress_bar(p_db_percent * total_steps, total_steps, bar_width=20)
            #     # self.status_window.addstr(f"Power: {p_db:.3f} ")
            #     # self.status_window.addstr(f"Power: {bar} ")
            #     self.status_window.addstr(f"{bar} ")

            #     # self.status_window.addstr(f"{self.vad_state} ")
            #     if self.vad_state == VADState.Started or self.vad_state == VADState.Continue:
            #         self.status_window.addstr(self.rotate_bar.next() + " ")
            #     else:
            #         self.status_window.addstr(self.rotate_bar.get() + " ")
            #         # self.rotate_bar.reset()

            #     # self.status_window.addstr(f"{self.__ch} ")
            #     self.status_window.addstr("Device: ")
            #     self.status_window.addstr(self.audio.get_device_name() + " ")
            #     # self.status_window.addstr(f"Time: {data.time:.3f} ")

            #     # CPU使用率を取得
            #     # cpu_percent = self.cpu_percent
            #     # self.status_window.addstr(f"CPU: {cpu_percent:5.1f}% ")
            #     self.status_window.addstr(f"CPU: {self.cpu_bar.get_string()} ")

            #     # 現在のメモリの使用量をGBで取得
            #     memory_in_byte = psutil.Process().memory_info().rss
            #     # memory_info = psutil.Process().memory_info()
            #     # memory_in_byte = memory_info.vms
            #     memory_in_gb = memory_in_byte / (1024 ** 3)
            #     self.status_window.addstr(f"Memory: {memory_in_gb:4.2f}GB ")


            #     # fill the rest space with space
            #     self.status_window.addstr(" " * (self.status_window.getmaxyx()[1] - self.status_window.getyx()[1] - 1))

            #     self.status_window.refresh()

        self.audio.add_callback(audio_callback_)

        ### main routine
        self.__ch = None
        self.audio.start()
        self.stdscr.nodelay(True)

        while True:
            # with self.cond:
            #     self.cond.wait()
            ch = self.stdscr.getch()
            if ch == ord('q'):
                break
            if ch == ord(' '):
                # self.status_on = not self.status_on
                self.asr_worker.setActive(not self.asr_worker.is_active)
                self.status_window.set_asr_is_active(self.asr_worker.is_active)
            self.__ch = ch
            time.sleep(0.01)


if __name__ == "__main__":
    from .audio import PyaudioAudio

    asr = ASR.from_pretrained(
        espnet2_asr_model_tag="fujie/espnet_asr_cbs_transducer_120303_hop132_cc0105",
        # espnet2_asr_model_tag="fujie/espnet_asr_cbs_transducer_120303_hop132_csj_alt",
        espnet2_asr_args=dict(
            streaming=True,
            lm_weight=0.0,
            beam_size=20,
            beam_search_config=dict(search_type="maes")
        ),
        kana_kanji_model_tag="fujie/kana_kanji_20240307")

    audio = PyaudioAudio()
    vad = VAD(webrtcvad_mode=3, end_frame_num_thresh=30)

    main = Main(audio, vad, asr)
    curses.wrapper(main.run)
