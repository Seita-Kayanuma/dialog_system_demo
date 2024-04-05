# .dataset の SpectrogramImageDataset, SpectrogramImageDatasetForNoise を使った学習器
from ....corpus.noise.spec_image import SpecImageDataManagerForNoise
from ....sound.sigproc.spec_image import SpectrogramImageGenerator
from ....sound.sigproc.noise import NoiseAdder
from .dataset import SpectrogramImageDataset, SpectrogramImageDatasetForNoise
from .dataset import CollateForSpectrogramImageDataset
from .base import SpectrogramImageAutoEncoderTrainer
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from ....ext.torch.callbacks.early_stopper import EarlyStopper


class SpectrogramImageAutoEncoderTrainer0009(
        SpectrogramImageAutoEncoderTrainer):
    """
    """

    def __init__(self):
        super().__init__()
        self._train_loader = None
        self._test_loader = None

    # --- 各種データセットの組み立て ---

    # Noise Adder for Speech
    def build_noise_adder(self):
        from ....corpus.noise import JEIDA, SoundffectLab, Fujie
        wav_path_list = JEIDA().get_wav_path_list() + \
            SoundffectLab().get_wav_path_list() + \
            Fujie().get_wav_path_list()
        return NoiseAdder(wav_path_list)

    # Train Speech CSJ
    def build_train_dataset_csj(self,
                                cond: str,
                                generator: SpectrogramImageGenerator,
                                noise_adder: NoiseAdder,
                                max_utterance_num=None):
        from ....corpus.speech.csj import CSJ
        csj = CSJ()
        id_list = csj.get_id_list(
            cond,
            # cond='[AS].*[^1]$',
            # cond='A01M0014',
        )
        sidm = csj.get_spec_image_data_manager(generator, noise_adder)
        return SpectrogramImageDataset(sidm, id_list, max_utterance_num)

    # Train Noise SoundeffectLab
    def build_train_dataset_SoundeffectLab(
            self,
            generator: SpectrogramImageGenerator,
            noise_adder: NoiseAdder = None):
        from ....corpus.noise import SoundffectLab
        sel = SoundffectLab()
        id_list = sel.get_id_list()
        sidm = SpecImageDataManagerForNoise(sel, generator, noise_adder)
        return SpectrogramImageDatasetForNoise(sidm, id_list)

    # Train Noise Fujie
    def build_train_dataset_Fujie(self,
                                  generator: SpectrogramImageGenerator,
                                  noise_adder: NoiseAdder = None):
        from ....corpus.noise import Fujie
        fj = Fujie()
        id_list = fj.get_id_list()
        sidm = SpecImageDataManagerForNoise(fj, generator, noise_adder)
        return SpectrogramImageDatasetForNoise(sidm, id_list)

    # Valid Speech CSJ
    def build_vaid_dataset_csj(self,
                               generator: SpectrogramImageGenerator,
                               noise_adder: NoiseAdder = None,
                               max_utterance_num=None):
        from ....corpus.speech.csj import CSJ
        csj = CSJ()
        id_list = csj.get_id_list(cond='[ASRD].*1$',
                                  # cond='A01M0015',
                                  )
        sidm = csj.get_spec_image_data_manager(generator, noise_adder)
        return SpectrogramImageDataset(sidm, id_list, max_utterance_num)

    def build_data(self):
        generator = SpectrogramImageGenerator()
        noise_adder = self.build_noise_adder()
        train_dataset_csjs = []
        for kind in ['[AR]', '[SD]']:
            for n in range(10):
                cond = "{}.*{}[^1]$".format(kind, n)
                train_dataset_csjs.append(
                    self.build_train_dataset_csj(cond,
                                                 generator,
                                                 noise_adder,
                                                 max_utterance_num=10))
        train_dataset_soundeffect_lab = \
            self.build_train_dataset_SoundeffectLab(generator)
        train_dataset_fujie = self.build_train_dataset_Fujie(generator)
        valid_dataset_csj = self.build_vaid_dataset_csj(generator,
                                                        max_utterance_num=10)
        train_datasets = []
        for ds in train_dataset_csjs:
            train_datasets.extend([
                train_dataset_soundeffect_lab,
                train_dataset_fujie,
                ds])
        train_dataset = ConcatDataset(train_datasets)
        valid_dataset = valid_dataset_csj

        total_len = len(train_dataset)
        each_len_ratio = [len(ds) / total_len for ds in train_dataset.datasets]
        print("EACH DATASET RATIO -> {}".format(' '.join(
            ['{:.3f}'.format(r) for r in each_len_ratio])))

        train_collate = CollateForSpectrogramImageDataset(train_dataset)
        valid_collate = CollateForSpectrogramImageDataset(valid_dataset)

        self._train_loader = DataLoader(train_dataset,
                                        batch_size=1000,
                                        shuffle=False,
                                        num_workers=8,
                                        collate_fn=train_collate)
        self._test_loader = DataLoader(valid_dataset,
                                       batch_size=100,
                                       shuffle=False,
                                       num_workers=8,
                                       collate_fn=valid_collate)
        # import ipdb; ipdb.set_trace()

    def get_criterion(self):
        return nn.MSELoss()

    def get_optimizer(self, model):
        return optim.Adam(model.parameters())

    def get_train_loader(self):
        if self._train_loader is None:
            self.build_data()
        return self._train_loader

    def get_validation_loader(self):
        if self._test_loader is None:
            self.build_data()
        return self._test_loader
