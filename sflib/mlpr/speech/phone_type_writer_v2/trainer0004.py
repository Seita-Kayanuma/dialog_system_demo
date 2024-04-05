# 雑音のみのデータは nil と認識するように明示的に学習する
from .base import PhoneTypeWriterTrainer, TorchTrainerForPhoneTypeWriter
from ....ext.torch.callbacks.reporters \
    import StandardReporter, CsvWriterReporter
from ....ext.torch.callbacks.early_stopper import EarlyStopper
from ....ext.torch.callbacks.snapshot import Snapshot
from ....ext.torch.callbacks.train import ClippingGrad
from ....corpus.speech.csj import CSJ, WavDataWithTransManagerCSJ
from ....corpus.speech.rwcp_spxx import RWCP_SPXX, WavDataWithTransManagerRWCP
from ....corpus.speech.wav import WavDataWithTransManager
from ....corpus.speech.wav import WavDataWithTrans
from sflib.sound.sigproc.noise import NoiseAdder, IntermittentNoiseAdder
from sflib.corpus.noise import JEIDA, SoundffectLab, Fujie
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import copy
import numpy as np
from .base import convert_phone_to_id
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

    
# 5秒間の無音を含むダミーの音声データ．
class DummyDataWithTrans:
    def __init__(self):

        class DummyTrans:
            def __init__(self):
                self.pron = ' '.join(['nil'] * 5)
        self.trans = DummyTrans()
        
    def get_wav_data(self):
        # 完全な無音はよろしく無い？ので小さい乱数を入れておく
        return np.int16(np.random.rand(16000 * 5) * 128)

    def clear(self):
        pass

    
class WavDataTransDatasetWithDummy(Dataset):
    """n個に1回ダミーデータを挟むデータセット
    """
    def __init__(self, orig, n=10):
        self._orig = orig
        self._n = n
        self._dummy_len = n * len(orig) // (n - 1)
         
    def __len__(self):
        return self._dummy_len

    def __getitem__(self, i):
        if (i + 1) % self._n == 0:
            return DummyDataWithTrans()
        else:
            return self._orig[(self._n - 1) * i // self._n]
        

class CollateWavData:
    def __init__(self,
                 noise_adder: NoiseAdder = None,
                 interm_noise_adder: IntermittentNoiseAdder = None):
        self._noise_adder = noise_adder
        self._interm_noise_adder = interm_noise_adder

    def __call__(self, wav_data_list):
        # 特徴量を生成
        wav_list = []
        for wav_data in wav_data_list:
            wav = wav_data.get_wav_data()
            if self._noise_adder is not None and np.random.randint(2) > 0:
                wav = self._noise_adder.add_noise(wav)
            if self._interm_noise_adder is not None and np.random.randint(2) > 0:
                wav = self._interm_noise_adder.add_noise(wav)
            wav_list.append(wav)
        # 正解音素列を生成
        phones_list = [
            torch.tensor(convert_phone_to_id(wav_data.trans.pron.split(' ')))
            for wav_data in wav_data_list
        ]

        for wav_data in wav_data_list:
            wav_data.clear()

        return wav_list, phones_list
    
    
class PhoneTypeWriterTrainer0004(PhoneTypeWriterTrainer):
    def __init__(self):
        super().__init__()

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
        dataset_train = WavDataTransDatasetWithDummy(dataset_train)

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

        soundffectlab = SoundffectLab()
        fujie = Fujie()
        interm_noise_adder = IntermittentNoiseAdder(
            soundffectlab.get_wav_path_list() + fujie.get_wav_path_list())
        
        collate_fn_train = CollateWavData(noise_adder, interm_noise_adder)
        collate_fn_valid = CollateWavData()

        # dataset_train = Subset(dataset_train, list(range(100)))
        # dataset_vali = Subset(dataset_vali, list(range(100)))

        train_loader = DataLoader(
            dataset_train,
            batch_size=5,
            collate_fn=collate_fn_train,
            # num_workers=2,
            num_workers=0,
            # worker_init_fn=worker_init_fn_train,
            # shuffle=True,
            shuffle=False,
        )
        vali_loader = DataLoader(
            dataset_vali,
            batch_size=1,
            collate_fn=collate_fn_valid,
            # num_workers=8,
            # num_workers=4,
            num_workers=0,
            # worker_init_fn=worker_init_fn_vali,
        )
        # import ipdb; ipdb.set_trace()
        callbacks = [
            ClippingGrad(1.),
            StandardReporter(),
            CsvWriterReporter(phone_type_writer.get_csv_log_filename()),
            # Snapshot(final_filename=phone_type_writer.get_model_filename()),
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


