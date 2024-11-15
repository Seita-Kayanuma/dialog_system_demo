# Demo System README

このプロジェクトは、音声認識と音声再生を用いたデモシステムです。以下のファイル構成と役割に従ってシステムが動作します。
大元は藤江先生の音声認識　
https://github.com/fujielab/kana_kan_streaming_asr/blob/main/fjext/kana_kan_asr/main.py

## ファイル構成
- demo/silent/demo_system.py
  - メインのシステムファイル。AudioPlayerおよびMainクラスを中心に修正を行います。

## クラスの役割
### ASRWorker
- 音声認識を動かすクラス
- 実際の音声認識処理はasr.pyで行われています。
- 基本的にこのクラスを大きく変更する必要はありません。

### AudioPlayer (修正必要)
### シナリオを選択する
- 音声を再生するクラス。
- 事前にシナリオの音声を読み込み、順番に再生します。
- 主にこのクラスを修正して、音声再生の機能を改善します。

### Main (修正必要)
### システムをが発話する時の条件を決める
- 色々なものの初期化を行うクラス。
- 主な修正箇所はrunメソッドのwhile文内で、以下の条件に基づいて音声を再生するかどうかを判断します：
  - `self.user_utterance_flg`：システム発話後にユーザーの発話があったか。
  - `not self.system_utterance_flg`：システム発話中でないか。
  - `self.asr_worker.all_len`：ユーザーの発話長。
  - `self.vad.silent_time >= self.silent_utterance_limit`：無音区間の長さが所定の長さを超えるか。

- これらの条件を用いて、ゼミで流しているデモの発話タイミングを制御します。無音区間の長さを変更することで、発話タイミングを調整することが可能です。

## 詳細
ASRWorker: 音声認識処理
  ・関数
    ・put: 音声データ/VAE状態の取得
    ・run: 音声認識処理
  ・変数
    ・self.asr_state: 状態(音声認識をするかどうかを決める)
    ・self.history_text: (すでにfinalになってる)1ターンの発話内容(モーラ)
    ・self.history_len: self.history_textのモーラ長
    ・self.all_text: 1ターンの発話内容(モーラ)
    ・self.all_len: self.all_textのモーラ長

AudioPlayer: 発話処理
  ・事前にシナリオの音を読み込み, 順番に再生している
  ・関数
    ・play_pyaudio: (self.audio_datasの)音声再生
    ・play: どの音声を再生するか/終了条件
  ・変数
    ・self.audio_num: 音番号(シナリオに沿って変わる)
    ・self.audio_limit: 音番号の長さ(シナリオの終了確認に使用)
    ・self.flg: 処理を終了するかどうか(self.audio_limitに達した時, Mainの処理を止める)
    ・self.audio_datas: 音声データ(pyaudio)

Main: 初期化/発話処理
 ・初期化
   ・AudioPlayer
   ・ASRWorker
 ・関数
   ・_vad_callback: VADの結果から, ASRWorkerを動かす
   ・audio_play: 発話条件を満たした時(runのwhile)AudioPlayerによって再生
   ・reset: 発話(audio_play)後の処理
   ・run: threadの立ち上げ/発話(発話条件はrunのwhile内に記載)
 ・変数
   ・self.user_utterance_flg: システム発話後にユーザの発話があったか
   ・self.system_utterance_flg: システム発話中か
   ・self.asr_worker.all_len: ユーザの発話長
   ・self.vad.silent_time: 無音区間の長さ
   ・self.silent_utterance_limit: 発話する際の無音区間の長さの閾値

