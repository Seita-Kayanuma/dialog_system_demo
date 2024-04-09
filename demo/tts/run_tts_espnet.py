import os
import soundfile as sf
from IPython.display import Audio
from espnet2.utils.types import str_or_none
from espnet2.bin.tts_inference import Text2Speech
from espnet_model_zoo.downloader import ModelDownloader


# python demo/tts/run_tts_espnet.py
OUTPATH = 'demo/tts/audio/sample2'


def main():
    # source: https://qiita.com/kan-bayashi/items/0371e06202641dbfa0ad
    audio_path = os.path.join(OUTPATH, 'sample.wav')
    d = ModelDownloader()
    text2speech = Text2Speech(**d.download_and_unpack("kan-bayashi/jsut_fastspeech2"))
    text = "はじめまして。私の名前はつくよみです。"
    wav = text2speech(text)['wav']
    sf.write(audio_path, wav, text2speech.fs)


def main2():
    # source: https://zenn.dev/if001/articles/df65e5a7c35f3c
    audio_path = os.path.join(OUTPATH, 'sample.wav')
    lang = 'Japanese'
    tag = 'kan-bayashi/tsukuyomi_full_band_vits_prosody'
    vocoder_tag = 'none'
    text2speech = Text2Speech.from_pretrained(
        model_tag=str_or_none(tag),
        vocoder_tag=str_or_none(vocoder_tag),
        device='cpu',
        # Only for Tacotron 2 & Transformer
        threshold=0.5,
        # Only for Tacotron 2
        minlenratio=0.0,
        maxlenratio=10.0,
        use_att_constraint=False,
        backward_window=1,
        forward_window=3,
        # Only for FastSpeech & FastSpeech2 & VITS
        speed_control_alpha=1.0,
        # Only for VITS
        noise_scale=0.333,
        noise_scale_dur=0.333,
    )
    text = "はじめまして。私の名前はつくよみです。"
    wav = text2speech(text)['wav']
    sf.write(audio_path, wav, text2speech.fs)


if __name__ == '__main__':
    # main()
    main2()