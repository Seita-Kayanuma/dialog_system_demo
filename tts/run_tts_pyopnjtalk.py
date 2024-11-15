import os
import torch
import soundfile
import numpy as np
from scipy.io import wavfile
from speecht5_openjtalk_tokenizer import SpeechT5OpenjtalkTokenizer
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5FeatureExtractor, SpeechT5Processor


# openjtalk error
# python demo_/tts/run_tts.py
MODELNAME = "esnya/japanese_speecht5_tts"


def speecht5_tts():  
    device = torch.device('cpu')
    with torch.no_grad():
        model = SpeechT5ForTextToSpeech.from_pretrained(MODELNAME, device_map=device, torch_dtype=torch.bfloat16)
        tokenizer = SpeechT5OpenjtalkTokenizer.from_pretrained(MODELNAME)
        feature_extractor = SpeechT5FeatureExtractor.from_pretrained(MODELNAME)
        processor = SpeechT5Processor(feature_extractor, tokenizer)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", device_map=device, torch_dtype=torch.bfloat16)
        input = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"
        input_ids = processor(text=input, return_tensors="pt").input_ids.to(model.device)
        speaker_embeddings = np.random.uniform(-1, 1, (1, 16))  # (batch_size, speaker_embedding_dim = 16), first dimension means male (-1.0) / female (1.0)
        speaker_embeddings = torch.FloatTensor(speaker_embeddings).to(device=model.device, dtype=model.dtype)
        waveform = model.generate_speech(input_ids, speaker_embeddings,vocoder=vocoder,)
        waveform = waveform / waveform.abs().max()  # normalize
        waveform = waveform.reshape(-1).cpu().float().numpy()
        soundfile.write("output.wav", waveform, vocoder.config.sampling_rate)   


if __name__ == '__main__':
    speecht5_tts()