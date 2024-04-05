# coding: utf-8
# Spectrogram Auto Encoder と Phone Type Writer を用いた特徴抽出器
from .base import TurnDetectorFeatureExtractor
from ....speech.feature.autoencoder_pytorch.base \
    import load as load_autoencoder_trainer
from ..phone_type_writer_pytorch import base as ptwb
from ....sound.sigproc.spec_image import SpectrogramImageGenerator
import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


class TurnDetectorFeatureExtractor0002(TurnDetectorFeatureExtractor):
    def __init__(self,
                 autoencoder_version,
                 autoencoder_trainer_module_postfix,
                 autoencoder_trainer_class_postfix,
                 phone_type_writer_feature_extractor_version,
                 phone_type_writer_version,
                 phone_typer_writer_trainer_module_postfix,
                 phone_typer_writer_trainer_class_postfix,
                 spec_image_shift=2,
                 batch_size=1,
                 device='cpu'):
        # オートエンコーダの取得（まずCPUに詠みこむ）
        trainer = load_autoencoder_trainer(autoencoder_version,
                                           autoencoder_trainer_module_postfix,
                                           autoencoder_trainer_class_postfix,
                                           map_location='cpu')
        autoencoder = trainer.autoencoder
        self._autoencoder_filename_base = \
            autoencoder.get_filename_base() + '+' + \
            trainer.get_filename_base()

        self._autoencoder = autoencoder
        if device is not None:
            self._autoencoder.to(device)
        self._autoencoder.share_memory()

        # Phone Type Writer の取得
        phone_type_writer_feature_extractor = \
            ptwb.construct_feature_extractor_with_autoencoder(
                phone_type_writer_feature_extractor_version,
                autoencoder_version,
                autoencoder_trainer_module_postfix,
                autoencoder_trainer_class_postfix)
        phone_type_writer = ptwb.construct_phone_type_writer(
            phone_type_writer_version, phone_type_writer_feature_extractor)
        phone_type_writer_trainer = ptwb.construct_phone_type_writer_trainer(
            phone_typer_writer_trainer_module_postfix,
            phone_typer_writer_trainer_class_postfix, phone_type_writer)
        ptwb.load_phone_type_writer(phone_type_writer_trainer,
                                    download=False,
                                    map_location='cpu')
        self._phone_type_writer = phone_type_writer
        if device is not None:
            self._phone_type_writer.device = device
        self._phone_type_writer.torch_model.eval()

        self._batch_size = batch_size
        self._spec_image_generators = []
        for i in range(self._batch_size):
            self._spec_image_generators.append(
                SpectrogramImageGenerator(image_shift=spec_image_shift))
        self._spec_image_shift = spec_image_shift
        self._device = device

    def get_filename_base(self):
        s = super().get_filename_base()
        s += '+' + self._autoencoder_filename_base
        s += '+' + self._phone_type_writer.get_filename_base()
        s += '+' + "{}".format(self._spec_image_generators[0].image_shift)
        return s

    def reset(self):
        self._phone_type_writer.reset()
        for gen in self._spec_image_generators:
            gen.reset()

    def detach(self):
        self._phone_type_writer.detach()

    def calc(self, wav_list: list):
        batch = len(wav_list)

        # スペクトル画像生成器の足りない分は自動的に拡張する
        if batch > len(self._spec_image_generators):
            num = batch - len(self._spec_image_generators)
            for i in range(num):
                self._spec_image_generators.append(
                    SpectrogramImageGenerator(
                        image_shift=self._spec_image_shift))

        # -- 学習時は非常に長いデータが与えられるので，1秒ごと順番に
        # -- 特徴抽出を行う
        pos_start = 0
        dur = int(16000 * 0.5)
        feat_list = []

        self._phone_type_writer.reset()

        while True:
            spec_images = []
            spec_image_nums = []
            for i, (wav, gen) in enumerate(
                    zip(wav_list, self._spec_image_generators)):
                if len(wav) <= pos_start:
                    spec_image_nums.append(0)
                else:
                    pos_end = min(pos_start + dur, len(wav))
                    img_list = gen.input_wave(wav[pos_start:pos_end])
                    spec_images.extend(img_list)
                    spec_image_nums.append(len(img_list))
            if max(spec_image_nums) == 0:
                break
            spec_images = np.stack(spec_images)
            spec_images = torch.tensor(np.float32(spec_images))
            _, w, h = spec_images.shape
            spec_images = spec_images.reshape(-1, 1, w, h)
            if self._device is not None:
                spec_images = spec_images.to(self._device)
            x, l2 = self._autoencoder.encode(spec_images)

            # パスを切る（これをしないとメモリを大量消費する）
            x = x.clone().detach()

            sub_ae_feat_list = []
            cumidx = np.cumsum(spec_image_nums).tolist()
            fake_flag_list = []  # 仮の特徴量を入れてしまったものにTrue
            for s, e in zip([0] + cumidx[:-1], cumidx):
                if s != e:
                    sub_ae_feat_list.append(x[s:e])
                    fake_flag_list.append(False)
                else:
                    sub_ae_feat_list.append(x[0:1] * 0)
                    fake_flag_list.append(True)

            # # 音素タイプライタのベクトルを求める
            packed_sub_ae_feat = pack_sequence(sub_ae_feat_list, enforce_sorted=False)
            feat_ptw = self._phone_type_writer.calc_hidden_feature_from_extracted_feature(
                packed_sub_ae_feat)
            self._phone_type_writer.detach()
            padded_sub_ptw_feat, sub_ptw_feat_len = pad_packed_sequence(
                feat_ptw)
            padded_sub_ptw_feat = padded_sub_ptw_feat.clone().detach()
            sub_ptw_feat_len = sub_ptw_feat_len.clone().detach()
            sub_ptw_feat_list = []
            for i, l in enumerate(sub_ptw_feat_len):
                sub_ptw_feat_list.append(padded_sub_ptw_feat[:l, i, :])

            # 求めた特徴量をオートエンコーダ特徴量と連結する
            # import ipdb; ipdb.set_trace();
            sub_concat_feat_list = []
            for aef, ptwf in zip(sub_ae_feat_list, sub_ptw_feat_list):
                sub_concat_feat_list.append(torch.cat([aef, ptwf], dim=1))

            # 全体リストに加える
            if len(feat_list) == 0:
                for sub_feat, fake_flag in zip(sub_concat_feat_list,
                                               fake_flag_list):
                    if not fake_flag:
                        feat_list.append([sub_feat])
                    else:
                        feat_list.append([])
            else:
                for feat, sub_feat, fake_flag in zip(feat_list,
                                                     sub_concat_feat_list,
                                                     fake_flag_list):
                    if not fake_flag:
                        feat.append(sub_feat)
            pos_start += dur
            # import ipdb; ipdb.set_trace()
        concat_feat_list = [torch.cat(f) for f in feat_list]
        if len(concat_feat_list) > 0:
            feat = pack_sequence(concat_feat_list, enforce_sorted=False)
            # import ipdb; ipdb.set_trace()
            return feat
        else:
            return None

    @property
    def feature_dim(self):
        return self._autoencoder.bottleneck_dim + \
            self._phone_type_writer.hidden_feature_dim

    @property
    def feature_rate(self):
        gen = self._spec_image_generators[0]
        return 16000 / gen.num_samples_per_image_shift
