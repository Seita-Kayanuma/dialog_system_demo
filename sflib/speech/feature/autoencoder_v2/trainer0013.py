# trainer0010のTorch版
from corpus.noise.spec_image_torch import SpecImageDataManagerForNoiseTorch
from sound.sigproc.spec_image_torch import SpectrogramImageGeneratorTorch
from sound.sigproc.noise \
    import NoiseAdder, IntermittentNoiseAdder, MutipleNoiseAdder
from dataset_torch \
    import SpectrogramImageDatasetTorch, SpectrogramImageDatasetForNoiseTorch
from dataset_torch import CollateForSpectrogramImageDatasetTorch
from dataset_torch import ConcatDatasetWrapper
from base import SpectrogramImageAutoEncoderTrainer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from ext.torch.callbacks.early_stopper import EarlyStopper
from ext.torch.trainer import TorchTrainerCallback


class DatasetShuffleCallback(TorchTrainerCallback):
    def __init__(self, dataset):
        self._dataset = dataset

    def train_epoch_finish_callback(self, trainer, epoch, total_epoch,
                                    train_loss):
        self._dataset.shuffle()

        
class SpectrogramImageAutoEncoderTrainer0013(
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
        from corpus.noise import JEIDA, SoundffectLab, Fujie
        wav_path_list = JEIDA().get_wav_path_list()
        basic_noise_adder = NoiseAdder(wav_path_list)
        noise_wav_path_list = \
            SoundffectLab().get_wav_path_list() + Fujie().get_wav_path_list()

        interm_noise_adder = IntermittentNoiseAdder(noise_wav_path_list)
        return MutipleNoiseAdder([basic_noise_adder, interm_noise_adder])

    # Train Speech CSJ
    def build_train_dataset_csj(self,
                                cond: str,
                                generator: SpectrogramImageGeneratorTorch,
                                noise_adder: NoiseAdder,
                                max_utterance_num=None):
        from corpus.speech.csj import CSJ
        csj = CSJ()
        id_list = csj.get_id_list(
            cond,
            # cond='[AS].*[^1]$',
            # cond='A01M0014',
        )
        sidm = csj.get_spec_image_data_manager_torch(generator, noise_adder)
        return SpectrogramImageDatasetTorch(sidm, id_list, max_utterance_num)

    # Train Noise SoundeffectLab
    def build_train_dataset_SoundeffectLab(
            self,
            generator: SpectrogramImageGeneratorTorch,
            noise_adder: NoiseAdder = None):
        from corpus.noise import SoundffectLab
        sel = SoundffectLab()
        id_list = sel.get_id_list()
        sidm = SpecImageDataManagerForNoiseTorch(sel, generator, noise_adder)
        return SpectrogramImageDatasetForNoiseTorch(sidm, id_list)

    # Train Noise Fujie
    def build_train_dataset_Fujie(self,
                                  generator: SpectrogramImageGeneratorTorch,
                                  noise_adder: NoiseAdder = None):
        from corpus.noise import Fujie
        fj = Fujie()
        id_list = fj.get_id_list()
        sidm = SpecImageDataManagerForNoiseTorch(fj, generator, noise_adder)
        return SpectrogramImageDatasetForNoiseTorch(sidm, id_list)

    # Valid Speech CSJ
    def build_vaid_dataset_csj(self,
                               generator: SpectrogramImageGeneratorTorch,
                               noise_adder: NoiseAdder = None,
                               max_utterance_num=None):
        from corpus.speech.csj import CSJ
        csj = CSJ()
        id_list = csj.get_id_list(cond='[ASRD].*1$',
                                  # cond='A01M0015',
                                  )
        sidm = csj.get_spec_image_data_manager_torch(generator, noise_adder)
        return SpectrogramImageDatasetTorch(sidm, id_list, max_utterance_num)

    def build_data(self, is_add_noise=False):
        generator = SpectrogramImageGeneratorTorch()
        # generator.to('cuda')
        if self._device is not None:
            generator.to(self._device)

        if is_add_noise:
            noise_adder = self.build_noise_adder()
        else:
            noise_adder = None

        train_dataset_csjs = []
        for kind in ['[AR]', '[SD]']:
            for n in range(10):
                cond = "{}.*{}[^1]$".format(kind, n)
                train_dataset_csjs.append(
                    self.build_train_dataset_csj(cond,
                                                 generator,
                                                 noise_adder,
                                                 max_utterance_num=10))
        if is_add_noise:
            train_dataset_soundeffect_lab = \
                self.build_train_dataset_SoundeffectLab(generator)
            train_dataset_fujie = self.build_train_dataset_Fujie(generator)
        
        valid_dataset_csj = self.build_vaid_dataset_csj(generator,
                                                        max_utterance_num=10)
        
        train_datasets = []
        for ds in train_dataset_csjs:
            train_datasets.extend([ds])

            if is_add_noise:
                train_datasets.extend([
                    train_dataset_soundeffect_lab,
                    train_dataset_fujie,
                    ])


        train_dataset = ConcatDatasetWrapper(train_datasets)
        valid_dataset = valid_dataset_csj

        total_len = len(train_dataset)
        each_len_ratio = [len(ds) / total_len for ds in train_dataset.datasets]
        print("EACH DATASET RATIO -> {}".format(' '.join(
            ['{:.3f}'.format(r) for r in each_len_ratio])))

        train_collate = CollateForSpectrogramImageDatasetTorch(train_dataset)
        valid_collate = CollateForSpectrogramImageDatasetTorch(valid_dataset)

        self._train_dataset = train_dataset
        self._train_loader = DataLoader(train_dataset,
                                        batch_size=1000,
                                        shuffle=False,
                                        num_workers=0,
                                        collate_fn=train_collate)
        self._test_loader = DataLoader(valid_dataset,
                                       batch_size=100,
                                       shuffle=False,
                                       num_workers=0,
                                       collate_fn=valid_collate)
        # import ipdb; ipdb.set_trace()

    def get_criterion(self):
        return nn.MSELoss()

    def get_optimizer(self, model):
        return optim.Adam(model.parameters())

    def get_train_loader(self, is_add_noise=False):
        if self._train_loader is None:
            self.build_data(is_add_noise)
        return self._train_loader

    def get_validation_loader(self, is_add_noise=False):
        if self._test_loader is None:
            self.build_data(is_add_noise)
        return self._test_loader

    def get_additional_callbacks(self):
        if self._train_loader is None:
            self.build_data()
        return [DatasetShuffleCallback(self._train_dataset)]
    
