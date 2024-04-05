from .base import PhoneTypeWriterTrainer, TorchTrainerForPhoneTypeWriter
from ....ext.torch.callbacks.reporters \
    import StandardReporter, CsvWriterReporter
from ....ext.torch.callbacks.early_stopper import EarlyStopper
from ....ext.torch.callbacks.snapshot import Snapshot
from ....ext.torch.callbacks.train import ClippingGrad
from ....corpus.speech.csj import CSJ, WavDataWithTransManagerCSJ
from ....corpus.speech.rwcp_spxx import RWCP_SPXX, WavDataWithTransManagerRWCP
from ....corpus.speech.wav import WavDataWithTransManager
from sflib.sound.sigproc.noise import NoiseAdder
from sflib.corpus.noise import JEIDA
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence, PackedSequence

import copy
import numpy as np
from .base import PhoneTypeWriterFeatureExtractor, convert_phone_to_id
import tqdm

torch.multiprocessing.set_start_method('spawn', force=True)


class WavDataTransDataset(Dataset):
    def __init__(self,
                 wav_data_manager: WavDataWithTransManager,
                 id_list,
                 max_utterance_num=None):
        self._data_manager = wav_data_manager
        self._id_list = id_list
        self._max_utt_num = max_utterance_num

        self._wav_data_list = []
        count = 0
        for id in tqdm.tqdm(self._id_list):
            # 指定されたIDのデータリストのリストを取得
            datas = self._data_manager.get(id)
            for data_list in datas:
                # 各データリストについて処理をする
                if self._max_utt_num is not None and \
                   len(data_list) > self._max_utt_num:
                    # 最大発話数が指定されている場合は一旦シャッフルする
                    data_list = copy.copy(data_list)
                    # np.random.shuffle(data_list)
                    n = len(data_list)
                    m = self._max_utt_num
                    ind = np.arange((n // m + 1) * m).reshape(-1, m).T.ravel()
                    ind = ind[ind < n]
                    count = (count + 1) % len(ind)
                    ind = np.concatenate([ind[count:], ind[:count]])
                    data_list = np.array(data_list)[ind]
                    data_list = data_list[:m].tolist()
                # 採用するデータを入れるリスト
                new_data_list = []
                # データを絞り込む
                for data in data_list:
                    try:
                        # 読みのデータに直す
                        phones = convert_phone_to_id(
                            data.trans.pron.split(' '))
                        num_phones = len(phones)
                        # 音声の長さが短すぎる，長すぎる，
                        # 音素が少なすぎる場合は採用しない
                        if data.num_samples < 320 * 10 or \
                           data.num_samples < 320 * (num_phones + 10) or \
                           data.num_samples > 8 * 16000 or \
                           num_phones < 4:
                            # print ("DATA INVALID %d %d" % (data.num_samples, num_phones))
                            continue
                        # その他は採用
                        new_data_list.append(data)
                        # 最大発話数が決まっていて，それに到達した場合は終了
                        if self._max_utt_num is not None and \
                           len(new_data_list) >= self._max_utt_num:
                            break
                    except Exception as e:
                        print(e)
                # 本番のデータに追加する
                self._wav_data_list.extend(new_data_list)
        # 全体のデータ数
        self.__num_data = len(self._wav_data_list)
        print("total %d samples loaded" % (self.__num_data, ))

    def __len__(self):
        return self.__num_data

    def __getitem__(self, i):
        return self._wav_data_list[i]


class CollateWavData:
    def __init__(self,
                 feature_extractor: PhoneTypeWriterFeatureExtractor,
                 noise_adder: NoiseAdder = None):
        self._feature_extractor = feature_extractor
        self._noise_adder = noise_adder

    def __call__(self, wav_data_list):
        # 特徴量を生成
        wav_list = []
        for wav_data in wav_data_list:
            wav = wav_data.get_wav_data()
            if self._noise_adder is not None and np.random.randint(2) > 0:
                wav = self._noise_adder.add_noise(wav)
            wav_list.append(wav)
        self._feature_extractor.reset()
        feat = self._feature_extractor.calc(wav_list)

        # 今回はオートエンコーダは更新しないためデータはデタッチしておく
        feat = PackedSequence(feat.data.detach(), feat.batch_sizes,
                              feat.sorted_indices, feat.unsorted_indices)

        # 正解音素列を生成
        phones_list = [
            torch.tensor(convert_phone_to_id(wav_data.trans.pron.split(' ')))
            for wav_data in wav_data_list
        ]
        packed_phones_list = pack_sequence(phones_list, enforce_sorted=False)

        for wav_data in wav_data_list:
            wav_data.clear()

        return feat, packed_phones_list


class PhoneTypeWriterTrainerCSJRWCP0003(PhoneTypeWriterTrainer):
    def __init__(self, phone_type_writer):
        super().__init__(phone_type_writer)

    def build_torch_trainer(self, phone_type_writer):
        criterion = nn.CTCLoss()
        if phone_type_writer.device is not None:
            criterion.to(phone_type_writer.device)
        # optimizer = optim.SGD(phone_type_writer.torch_model.parameters(),
        #                       lr=0.03,
        #                       weight_decay=3e-7,
        #                       momentum=0.9,
        #                       nesterov=True)
        optimizer = optim.Adam(phone_type_writer.torch_model.parameters())

        # CSJ（学習用）
        csj = CSJ()
        id_list_csj_train = csj.get_id_list(cond='A.*[^1]$')  # <-
        # id_list_csj_train = csj.get_id_list(cond='A.*2$')
        wm_csj = WavDataWithTransManagerCSJ()
        # dataset_csj_train = WavDataTransDataset(wm_csj, id_list_csj_train)
        dataset_csj_train = WavDataTransDataset(wm_csj, id_list_csj_train,
                                                100)  # <-
        # dataset_csj_train = WavDataTransDataset(wm_csj, id_list_csj_train, 10)
        # RWCP（学習用）
        rwcp = RWCP_SPXX()
        id_list_rwcp_train = rwcp.get_id_list(cond='^.1')
        wm_rwcp = WavDataWithTransManagerRWCP()
        dataset_rwcp_train = WavDataTransDataset(wm_rwcp, id_list_rwcp_train,
                                                 100)  # <-
        # dataset_rwcp_train = WavDataTransDataset(wm_rwcp, id_list_rwcp_train, 10)
        # 学習用のシーケンス
        dataset_train = ConcatDataset([dataset_csj_train, dataset_rwcp_train])

        # CSJ（検証用）
        id_list_csj_vali = csj.get_id_list(cond='A.*1$')
        dataset_csj_vali = WavDataTransDataset(wm_csj, id_list_csj_vali, 10)
        # RWCP（検証用）
        id_list_rwcp_vali = rwcp.get_id_list(cond='^.2')
        dataset_rwcp_vali = WavDataTransDataset(wm_rwcp, id_list_rwcp_vali, 10)
        # 学習用のシーケンス
        dataset_vali = ConcatDataset([dataset_csj_vali, dataset_rwcp_vali])

        jeida = JEIDA()
        noise_adder = NoiseAdder(jeida.get_wav_path_list())
        # worker_init_fn_train = WorkerInitFn(
        #     phone_type_writer.feature_extractor, noise_adder)
        # worker_init_fn_vali = WorkerInitFn(phone_type_writer.feature_extractor)

        collate_fn_train = CollateWavData(phone_type_writer.feature_extractor,
                                          noise_adder)
        collate_fn_valid = CollateWavData(phone_type_writer.feature_extractor)

        # dataset_train = Subset(dataset_train, list(range(100)))
        # dataset_vali = Subset(dataset_vali, list(range(100)))

        train_loader = DataLoader(
            dataset_train,
            batch_size=1,
            collate_fn=collate_fn_train,
            num_workers=4,
            # num_workers=0,
            # worker_init_fn=worker_init_fn_train,
            # shuffle=True,
        )
        vali_loader = DataLoader(
            dataset_vali,
            batch_size=1,
            collate_fn=collate_fn_valid,
            # num_workers=8,
            num_workers=4,
            # worker_init_fn=worker_init_fn_vali,
        )
        # import ipdb; ipdb.set_trace()
        callbacks = [
            ClippingGrad(1.),
            StandardReporter(),
            CsvWriterReporter(self.get_csv_log_filename()),
            Snapshot(final_filename=self.get_model_filename()),
            EarlyStopper(patience=3, verbose=True),
        ]

        trainer = TorchTrainerForPhoneTypeWriter(
            phone_type_writer,
            criterion,
            optimizer,
            train_loader,
            vali_loader,
            callbacks=callbacks,
            device=phone_type_writer.device,
            # epoch=1,
        )

        return trainer


def train(device=None):
    from .feature_autoencoder_0002 \
        import PhoneTypeWriterFeatureExtractorAutoEncoder0002 as FeatureExtractor
    from .phone_type_writer_0005 \
        import PhoneTypeWriter0005PyTorch as PhoneTypeWriter
    from . import base as ab
    feature_extractor = FeatureExtractor(12,
                                         'csj_0006',
                                         'CSJ0006',
                                         device=device)
    phone_type_writer = PhoneTypeWriter(feature_extractor, device=device)
    trainer = PhoneTypeWriterTrainerCSJRWCP0003(phone_type_writer)
    trainer.train()
    ab.save_phone_type_writer(trainer, upload=True)
    trainer.upload_csv_log()


def train0206(device=None):
    from .feature_autoencoder_0002 \
        import PhoneTypeWriterFeatureExtractorAutoEncoder0002 as FeatureExtractor
    from .phone_type_writer_0006 \
        import PhoneTypeWriter0006PyTorch as PhoneTypeWriter
    from . import base as ab
    feature_extractor = FeatureExtractor(12, 'csj_0006', 'CSJ0006')
    phone_type_writer = PhoneTypeWriter(feature_extractor, device=device)
    trainer = PhoneTypeWriterTrainerCSJRWCP0003(phone_type_writer)
    trainer.train()
    ab.save_phone_type_writer(trainer, upload=True)
    trainer.upload_csv_log()


def train0205_ae1208(device=None):
    from .feature_autoencoder_0002 \
        import PhoneTypeWriterFeatureExtractorAutoEncoder0002 as FeatureExtractor
    from .phone_type_writer_0005 \
        import PhoneTypeWriter0005PyTorch as PhoneTypeWriter
    from . import base as ab
    feature_extractor = FeatureExtractor(12, 'csj_0008', 'CSJ0008')
    phone_type_writer = PhoneTypeWriter(feature_extractor, device=device)
    trainer = PhoneTypeWriterTrainerCSJRWCP0003(phone_type_writer)
    trainer.train()
    ab.save_phone_type_writer(trainer, upload=True)
    trainer.upload_csv_log()

    
def train0205_ae1209(device=None):
    from .feature_autoencoder_0002 \
        import PhoneTypeWriterFeatureExtractorAutoEncoder0002 as FeatureExtractor
    from .phone_type_writer_0005 \
        import PhoneTypeWriter0005PyTorch as PhoneTypeWriter
    from . import base as ab
    feature_extractor = FeatureExtractor(12, 'csj_0009', 'CSJ0009')
    phone_type_writer = PhoneTypeWriter(feature_extractor, device=device)
    trainer = PhoneTypeWriterTrainerCSJRWCP0003(phone_type_writer)
    trainer.train()
    ab.save_phone_type_writer(trainer, upload=True)
    trainer.upload_csv_log()
    
