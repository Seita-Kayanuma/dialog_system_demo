from dataclasses import dataclass
import time

from transformers.pipelines import pipeline

from fjext.espnet2.bin.asr_transducer_inference_cbs import Speech2Text
from .audio import AudioData
from enum import Enum

class ASR:
    def __init__(self,
                 speech2text: Speech2Text = None,
                 kana_kan_pipeline: pipeline = None):
        self.speech2text = speech2text
        self.kana_kan_pipeline = kana_kan_pipeline

    @classmethod
    def from_pretrained(cls,
                        espnet2_asr_model_tag: str = None,
                        espnet2_asr_args: dict = None,
                        kana_kanji_model_tag: str = None):
        speech2text = Speech2Text.from_pretrained(
            espnet2_asr_model_tag,
            **espnet2_asr_args
        )
        kana_kan_pipeline = pipeline(
            "translation",
            model=kana_kanji_model_tag,
        )
        return cls(speech2text, kana_kan_pipeline)

    def recognize(self, audio: AudioData, is_final: bool = False) -> str:
        assert self.speech2text is not None, "Speech2Text is not initialized"
        hyps = self.speech2text.streaming_decode(audio, is_final=is_final)
        results = self.speech2text.hypotheses_to_results(hyps)
        if len(results) > 0 and len(results[0]) > 0:
            return results[0][0]

    def convert_kana_to_text(self, kana_text: str, nbest: int = 5) -> str:
        assert self.kana_kan_pipeline is not None, "Kana-Kanji model is not initialized"
        rsults = self.kana_kan_pipeline(kana_text,
                                        max_length=1000,
                                        num_beams=5,
                                        early_stopping=True,
                                        num_return_sequences=nbest)
        return rsults[0]['translation_text']

if __name__ == "__main__":
    from .audio import PyaudioAudio
    from .vad_silero import VAD, VADState

    asr = ASR.from_pretrained(
        espnet2_asr_model_tag="fujie/espnet_asr_cbs_transducer_120303_hop132_cc0105",
        espnet2_asr_args=dict(
            streaming=True,
            lm_weight=0.0,
            beam_size=20,
            beam_search_config=dict(search_type="maes")
        ),
        kana_kanji_model_tag="fujie/kana_kanji_20240307")

    audio = PyaudioAudio()
    vad = VAD(webrtcvad_mode=3)

    audio.add_callback(vad.process)

    def callback(data, state):
        if data is not None:
            print(data.time, data.data_np.shape, state)
        if state == VADState.Started or state == VADState.Continue:
            text = asr.recognize(data.data_np, is_final=False)
            if text is not None and len(text) > 0:
                print(text)
        elif state == VADState.Ended:
            text = asr.recognize(data.data_np, is_final=True)
            print(text)
            print(asr.convert_kana_to_text(text))

    vad.add_callback(callback)

    audio.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        audio.stop()
        print("Stopped")
        pass
