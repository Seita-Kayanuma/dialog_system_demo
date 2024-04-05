from ..trainer import TorchTrainerCallback, TorchTrainer
import copy


class EarlyStopper(TorchTrainerCallback):
    """
    Validation Lossをモニタリングして，
    条件が揃ったら早期に学習を終了させるためのコールバック．
    """

    def __init__(self, min_delta=0, patience=0, verbose=0):
        """
        Args:
          min_delta (float): 期待される差が，これより小さくなった場合に終了判定をする．
          patience (int): 最後に条件が満たされなくなった時に何回我慢するか．
          verbose (int): 1の時に冗長モード
        """
        self._min_delta = min_delta
        self._patience = patience
        self._verbose = verbose
        self.clear()

    def clear(self):
        # 最新の（最小）ロス
        self._latest_min_loss = None
        # 条件が満たされた回数
        self._count = 0
        # 最小ロス時のパラメタのコピー
        self._state_dict_at_min_loss = None

    def validation_epoch_finish_callback(self, trainer: TorchTrainer, epoch,
                                         total_epoch, validation_loss):
        if self._latest_min_loss is not None:
            if self._latest_min_loss - validation_loss < self._min_delta:
                # 条件を満たしていたらカウントを増やす
                self._count += 1
                if self._verbose:
                    print("EarlyStopper: count incremented ({}, delta={})".
                          format(self._count,
                                 self._latest_min_loss - validation_loss))
            else:
                # 条件を満たしていなければカウントをリセット
                self._count = 0
        # 最小値がまだない（最初のエポック）か，
        # 最小値が更新されていれば，コピーを取っておく
        if self._latest_min_loss is None or \
           self._latest_min_loss > validation_loss:
            self._latest_min_loss = validation_loss
            self._state_dict_at_min_loss = copy.deepcopy(
                trainer._model.state_dict())

        # カウントが閾値を超えていたら実際に早期終了させる
        if self._count > self._patience:
            trainer._early_stop_requested = True
            trainer._model.load_state_dict(self._state_dict_at_min_loss)
            if self._verbose:
                print("EarlyStopper: stop requested and state restored")

    def train_validation_epoch_finish_callback(self, trainer, epoch,
                                               total_epoch):
        if self._latest_min_loss is None:
            print("ERROR: Early stopper does not work without validation")

    def train_finish_callback(self, trainer):
        if self._latest_min_loss is not None:
            trainer._model.load_state_dict(self._state_dict_at_min_loss)
            if self._verbose:
                print("EarlyStopper: the best state is restored")
        self.clear()
