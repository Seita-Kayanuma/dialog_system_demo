# coding: utf-8
# 色々な継続時間長について管理をするクラス．
# 主に発話権維持／譲渡の実験用に利用する．
# VAD -> UTTERANCE -> TURN
# の順に区間長が短く，一つ手前の情報を使って次の情報を認定していく．
# つまり，UTTERANCEの単位認定は，VADの情報によって行われる．
# VADは，元々音声区間長が付与されているコーパスについては
# それを信用する．
#
# UTTERANCEの認定の基本ルール
#  VADの区間を，最長700msの無音区間（ギャップ）を無視して統合したもの
#  （対話相手の行為はあまり気にしない）
#
# TURNの認定の基本ルール
#  ・発話権をとる「ターン」と，相槌や短い応答などの「短い発話」に分けて考える．
#  ・基本的には早い者勝ちでターンを取る．
#  ・この時，1000ms以下の無音区間は埋めて考える．
#  ・1000ms以下の（独立した）短い音声区間は「短い発話」とみなし，
#    それだけではターンをとったことにはしない．
#  ・一度取られたターンは，1500ms以下の無音区間は無視されて統合される．
#  下記が詳しい．
#  https://drive.google.com/file/d/1EKEXunAZY7sp75aJ0mKhgD-WzVTzKIUZ

# 内部的なデータ構造は以下のようにする．
# 時刻は全て音声ファイルの開始時刻を0とするミリ秒単位の時刻．
#
# VAD, UTTERANCE
#   (開始時刻, 終了時刻) のタプルの列がチャネルごとにある．
#   vad[0][0][0] は，最初のチャネルの最初の区間の開始時刻を表す．
#
# TURN
#   (開始時刻, 終了時刻, 発話種類) のタプルの列がチャネルごとにある．
#   発話種類は，'T'がターン，'S'が短い発話．
#
# 全て，開始時間で昇順にソートされており，同じチャネル内では区間同士に
# 重複はないものとする．
import numpy as np
from os import path
from sflib.data.elan.data import EafInfo
from sflib.data.elan.io import write_to_eaf
import wave


def vad2utt(vad, MAX_UTTERANCE_GAP=700):
    """VAD情報をUTTERANCE情報に変換する．
    """
    result = []
    for vad_c in vad:
        r = []
        start = None
        end = None
        for s, e in vad_c:
            if start is None:
                start = s
                end = e
            else:
                if s - end < MAX_UTTERANCE_GAP:
                    end = e
                else:
                    r.append((
                        start,
                        end,
                    ))
                    start = s
                    end = e
        if start is not None:
            r.append((
                start,
                end,
            ))
        result.append(r)
    return result


def utt2turn(utt,
             THETA_SHORT_PAUSE=1000,
             THETA_KEEP_TURN=1500,
             THETA_SHORT_UTTERANCE=1000):
    if len(utt) != 2:
        raise RuntimeError("The number of channles must be 2.")

    # (チャネル, 開始時間, 終了時間) で開始時間でソートされた
    # リストを作成する
    mat_list = []
    for c, u in enumerate(utt):
        mat = np.concatenate([
            np.ones((len(u), 1), dtype=np.int32) * c,
            np.array(u, dtype=np.int32)
        ],
                             axis=1)
        mat_list.append(mat)
    mat_all = np.concatenate(mat_list, axis=0)
    utt_sorted = mat_all[np.argsort(mat_all[:, 1])].tolist()

    result = [[], []]

    def update_result(channel, start, end, label='T'):
        result[channel].append((start, end, label))

    X = None  # 0, 1, or None
    tsX = 0  # Xの開始時間
    teX = 0  # Xの終了時間
    dX = 0  # Xの継続時間
    Y = None  # 0, 1, or None
    tsY = 0  # Yの開始時間
    teY = 0  # Yの終了時間
    dY = 0  # Yの継続時間

    for i, row in enumerate(utt_sorted):
        S = row[0]  # 新しい発話のチャネル 0 or 1
        tsu = row[1]  # 開始時間
        teu = row[2]  # 終了時間
        du = teu - tsu  # 継続長

        if X is None:
            # X is None （最初の発話）
            #   X（現話者候補）をSにして状態を更新して次の発話へ
            X = S
            tsX = tsu
            teX = teu
            dX = du
            continue
        if S == X:
            # S is X （現ターン候補と同じ話者の発話が入ってきた場合）
            # フェーズ1の処理
            # Xを短い発話かターンとして認定する処理
            # （認定されずに新しい発話でXを拡張した場合は即次の発話へ移動）
            phase = 1
            t = tsu - teX  # 過去の発話候補と，新しい発話とのギャップ
            if t <= THETA_SHORT_PAUSE:
                # ギャップがSHORT_PAUSE以下
                #   -> 直前のXの発話と統合して次の発話へ
                teX = teu
                dX = teX - tsX
                continue
            elif t <= THETA_KEEP_TURN:
                # ギャップがKEEP_TURN以下
                if dX <= THETA_SHORT_UTTERANCE:
                    # XがSHORT_UTTERANCE以下の長さ
                    #   -> Xは短い発話として採用
                    #   -> Yの処理をする必要があるのでフェーズ2へ移行
                    update_result(X, tsX, teX, 'S')
                    phase = 2
                elif du > THETA_SHORT_UTTERANCE:
                    # 新しい発話がSHORT_UTTERANCEより長い
                    #   -> Xをuで拡張し次の発話へ
                    teX = teu
                    dX = teX - tsX
                    continue
                else:
                    # XがSHORT_UTTERANCEより長く（ターンキープ中）であり，
                    # 新しい発話がSHORT_UTTERANCE以下（で，
                    # その二つがSHORT_PAUSEより長くKEEP_TURNより短い間で発生している）の場合
                    #   -> Xを発話として認定
                    #   -> フェーズ2へ移行
                    update_result(X, tsX, teX)
                    phase = 2
            else:
                # ギャップがKEEP_TURNを超えて長い
                if dX <= THETA_SHORT_UTTERANCE:
                    # XがSHORT_UTTERANCE以下の長さ
                    #   -> Xは短い発話として採用
                    #   -> フェーズ2へ移行
                    update_result(X, tsX, teX, 'S')
                    phase = 2
                else:
                    # XがSHORT_UTTERANCEを超えた長さ
                    #   -> Xをターンとして認定
                    #   -> フェーズ2へ移行
                    update_result(X, tsX, teX)
                    phase = 2
            # フェーズ2の処理
            # Xを既に短い発話かターンとして認定した後の処理．
            # （実際にはフェーズ1のままここにくる処理はないようである）
            # 基本的には次の新しいターン候補（X）を決める処理．
            if phase == 2:
                if Y is not None and teY > teX:
                    # Y（ターン候補でなかった側の話者の発話）がNoneではなく
                    # その終了が，認定したXよりも後ろにある場合
                    #   -> Yを新しいターン候補Xにする
                    #   -> Sの処理はまだ終わっていないので処理を継続する
                    X = Y
                    tsX = tsY
                    teX = teY
                    dX = dY
                    Y = None
                else:
                    # YがNoneか，Yの方が認定したXより早く終わる場合
                    if Y is not None:
                        # YがNoneでは無い場合
                        #  -> YはXより遅く始まり，Xより早く終わっているということ．
                        #     長い発話である可能性はあるが，ターンは取れなかった
                        #     ということなので，短い発話として認定する．
                        #     （長い発話だった場合は別扱いにした方がいい可能性もある）
                        update_result(Y, tsY, teY, 'S')
                    # XもYも空席になった
                    # -> 新しい発話をターン候補（X）として次の発話へ
                    X = S
                    tsX = tsu
                    teX = teu
                    dX = teX - tsX
                    Y = None
                    continue
        # SがXでない（現ターン候補と異なる話者の発話が得られた）
        # または，Xをターンまたは短い発話として認定した後YをXにした後の処理．
        # 要は現ターン候補（X）と異なる話者の発話が得られた場合の処理．
        if Y is None:
            # YがNoneの場合（恐らくYをXに変更した場合にのみ起こる）
            phase = 1
            t = tsu - teX
            if t <= THETA_SHORT_PAUSE:
                # ギャップが短い発話より短い（Xが継続する可能性がある）
                #   -> 何もせずにフェーズ2（SをYにする）へ移行
                phase = 2
            elif t <= THETA_KEEP_TURN:
                # ギャップが短い発話より長く，ターン維持より短い
                if dX <= THETA_SHORT_UTTERANCE:
                    # Xが短い発話よりも短い
                    #   -> Xを短い発話として認定
                    #   -> フェーズ3（SをXにする）へ移行
                    update_result(X, tsX, teX, 'S')
                    phase = 3
                else:
                    # Xが短い発話よりも長い（Xが継続する可能性がある）
                    #   -> 何もせずにフェーズ2（SをYにする）へ移行
                    phase = 2
            else:
                # ギャップが更に長い
                if dX <= THETA_SHORT_UTTERANCE:
                    # Xが短い発話よりも短い
                    #   -> Xを短い発話として認定
                    #   -> フェーズ3（SをXにする）へ移行
                    update_result(X, tsX, teX, 'S')
                    phase = 3
                else:
                    # Xが短い発話よりも長い
                    #   -> Xをターンとして認定(?)
                    #      【注意!!】元のプログラムだと短い発話として認定している
                    #   -> フェーズ3（SをXにする）へ移行
                    update_result(X, tsX, teX)
                    phase = 3
            if phase == 2:
                # SをYにする
                Y = S
                tsY = tsu
                teY = teu
                dY = teY - tsY
                continue
            if phase == 3:
                # SをXにする
                X = S
                tsX = tsu
                teX = teu
                dX = teX - tsX
                continue
        else:
            # YがNoneでない（新しい発話がYのものであり，Xも別にある状況）
            t = tsu - teX
            if t <= 0:
                # XとS（Yの新しい発話）の一部，または全部が重複する場合
                t2 = tsu - teY
                if t2 <= THETA_SHORT_PAUSE:
                    # YとのギャップがSHORT_PAUSEより短ければYを拡張して次へ
                    teY = teu
                    dY = teY - tsY
                    continue
                elif t2 <= THETA_KEEP_TURN:
                    if dY <= THETA_SHORT_UTTERANCE:
                        # Yとのギャップがターン保持より短く，dYが短い発話より短い
                        #   -> Yを短い発話と認定
                        #   -> Sを新たなYとする
                        update_result(Y, tsY, teY, 'S')
                        Y = S
                        tsY = tsu
                        teY = teu
                        dY = teY - tsY
                        continue
                    else:
                        # Yとのギャップがターン保持より短く，dYが短い発話より長い
                        #   -> Yを拡張して次へ
                        teY = teu
                        dY = teY - tsY
                        continue
                else:
                    # Yとのギャップが更に長い
                    #   -> Yは短い発話として採用
                    update_result(Y, tsY, teY, 'S')
                    Y = S
                    tsY = tsu
                    teY = teu
                    dY = teY - tsY
                    continue
            else:
                # XとS（Yの新しい発話）が重複しない場合
                phase = 1
                if t <= THETA_SHORT_PAUSE:
                    phase = 2
                elif t <= THETA_KEEP_TURN:
                    if dX <= THETA_SHORT_UTTERANCE:
                        # Xを短い発話として採用
                        update_result(X, tsX, teX, 'S')
                        X = None
                        phase = 3
                    else:
                        phase = 2
                else:
                    if dX <= THETA_SHORT_UTTERANCE:
                        # Xを短い発話として採用
                        update_result(X, tsX, teX, 'S')
                        X = None
                        phase = 3
                    else:
                        # Xをターンとして認定
                        update_result(X, tsX, teX)
                        X = None
                        phase = 3
                if phase == 2:
                    t2 = tsu - teY
                    if t2 <= THETA_SHORT_PAUSE:
                        teY = teu
                        dY = teY - tsY
                        continue
                    elif t2 <= THETA_KEEP_TURN:
                        if dY <= THETA_SHORT_UTTERANCE:
                            # Yを短い発話として採用
                            update_result(Y, tsY, teY, 'S')
                            Y = S
                            tsY = tsu
                            teY = teu
                            dY = teY - tsY
                            continue
                        else:
                            teY = teu
                            dY = teY - tsY
                            continue
                    else:
                        # Yを短い発話として採用
                        update_result(Y, tsY, teY, 'S')
                        Y = S
                        tsY = tsu
                        teY = teu
                        dY = teY - tsY
                        continue
                if phase == 3:
                    t2 = tsu - teY
                    if t2 <= THETA_SHORT_PAUSE:
                        teY = teu
                        dY = teY - tsY
                        X = Y
                        tsX = tsY
                        teX = teY
                        dX = dY
                        Y = None
                        continue
                    elif t2 <= THETA_KEEP_TURN:
                        if dY <= THETA_SHORT_UTTERANCE:
                            update_result(Y, tsY, teY, 'S')
                            Y = None
                            phase = 4
                        else:
                            teY = teu
                            dY = teY - tsY
                            X = Y
                            tsX = tsY
                            teX = teY
                            dX = dY
                            Y = None
                            continue
                    else:
                        if dY <= THETA_SHORT_UTTERANCE:
                            update_result(Y, tsY, teY, 'S')
                            Y = None
                            phase = 4
                        else:
                            if teY > teX:
                                update_result(Y, tsY, teY)
                                Y = None
                            else:
                                update_result(Y, tsY, teY, 'S')
                                Y = None
                            phase = 4
                    if phase == 4:
                        X = S
                        tsX = tsu
                        teX = teu
                        dX = du
                        continue
    if X is not None and dX > THETA_SHORT_PAUSE:
        update_result(X, tsX, teX)

    return result


class DurationInfo:
    """
    一つの音声ファイルに対応する各種区間情報を保持するクラス．
    """

    def __init__(self, data, wav_filename):
        self._vad_info = data['VAD']
        self._utterance_info = data['UTTERANCE']
        self._turn_info = data['TURN']
        self._wav_filename = wav_filename
        self._wav = None

    @property
    def wav_filename(self):
        return self._wav_filename

    @property
    def wav(self):
        if self._wav is None:
            self._read_wav()
        return self._wav

    @property
    def vad(self):
        return self._vad_info

    @property
    def utterance(self):
        return self._utterance_info

    @property
    def turn(self):
        return self._turn_info

    def clear_cache(self):
        self._wav = None

    def to_eaf(self):
        """
        ELANのEAF情報へ変換する
        """
        eaf_info = EafInfo()
        for ch in range(2):
            for data_type, data_list in (('TURN', self.turn[ch]),
                                         ('UTTERANCE', self.utterance[ch]),
                                         ('VAD', self.vad[ch])):
                tier_name = "{}-{}".format(data_type, ch)
                for values in data_list:
                    start = values[0]
                    end = values[1]
                    if len(values) > 2:
                        anon = values[2]
                    else:
                        anon = ''
                    eaf_info.append_annotation(tier_name, start, end, anon)
        return eaf_info

    def _read_wav(self):
        wf = wave.open(self._wav_filename, 'r')
        channels = wf.getnchannels()
        data = wf.readframes(wf.getnframes())
        x = np.frombuffer(data, dtype=np.int16)
        x = x.reshape(-1, channels).T
        self._wav = x
        wf.close()

    def __repr__(self):
        return "DurationInfo(VAD=(%d, %d), UTTERANCE=(%d, %d), TURN=(%d, %d))" \
            % (len(self.vad[0]), len(self.vad[1]), len(self.utterance[0]),
               len(self.utterance[1]), len(self.turn[0]), len(self.turn[1]),)


class DurationInfoManager:
    """いくつかの種類（VAD，UTTERANCE，TURN）の音声区間情報を管理するクラス．
    """

    def __init__(self):
        pass

    def get_duration_info_dir_path(self):
        """
        区間情報を保存するディレクトリのパス（フルパス）を取得する．
        継承クラスで実装する必要がある．
        また，このメソッドが呼ばれた以降は当該ディレクトリは存在する
        ものとして実行される（したがって存在しなければ作成する必要がある）．
        """
        raise NotImplementedError()

    def get_wav_filename(self, id):
        pass

    def get_vad_info(self, id):
        """
        IDに対応するVAD区間情報を取得する．
        継承クラスで実装する必要がある．
        VAD区間情報は，音声チャネル毎に(開始時間, 終了時間)のリストを持つ．
        現状ステレオ限定なので，例えば
        info[0][0][0]はチャネル0の最初の発話の開始時間，
        info[1][0][1]はチャネル1の最初の発話の終了時間
        を表すようなデータを返却する必要がある．
        """
        pass

    def build_duration_info(self, id):
        """
        IDに対応する区間情報を構築する．
        """
        vad_info = self.get_vad_info(id)
        utt_info = vad2utt(vad_info)
        turn_info = utt2turn(vad_info)
        info = {'VAD': vad_info, 'UTTERANCE': utt_info, 'TURN': turn_info}
        return DurationInfo(info, self.get_wav_filename(id))

    def get_duration_info_path(self, id):
        """
        IDに対応する区間情報ファイルのパスを取得する
        """
        return path.join(self.get_duration_info_dir_path(), id + '.dur.txt')

    def get_duration_info(self, id):
        filename = self.get_duration_info_path(id)
        if not path.exists(filename):
            info = self.build_duration_info(id)
            self._write_duration(info, filename)
        else:
            info = self._read_duration(filename, self.get_wav_filename(id))
        return info

    def get_duration_info_eaf_dir_path(self):
        """
        区間情報を保存するディレクトリのパス（フルパス）を取得する．
        継承クラスで実装する必要がある．
        また，このメソッドが呼ばれた以降は当該ディレクトリは存在する
        ものとして実行される（したがって存在しなければ作成する必要がある）．
        """
        raise NotImplementedError()

    def get_duration_info_eaf_path(self, id):
        """
        IDに対応するEAFファイルの名前を取得する
        """
        return path.join(self.get_duration_info_eaf_dir_path(), id + '.eaf')

    def get_media_info_for_eaf(self, id):
        """
        IDに対応する音声ファイルの位置情報を返す．
        下記のもののタプルである必要がある
        絶対パスURL ... file:/somewhere/in/the/disk/foo.wav
        MIME type ... audio/wav でいいはず
        相対パスURL ... ../wav_safia/foo.wav など（eafファイルから見た相対位置）
        """
        pass

    def write_duration_info_eaf(self, id):
        eaf_info = self.get_duration_info(id).to_eaf()
        media_info = self.get_media_info_for_eaf(id)
        eaf_info.append_media(*media_info)
        write_to_eaf(eaf_info, self.get_duration_info_eaf_path(id))

    def _write_duration(self, info: DurationInfo, filename):
        with open(filename, 'w') as f:
            for duration_type, data in (('VAD', info.vad), ('UTTERANCE',
                                                            info.utterance),
                                        ('TURN', info.turn)):
                f.write(duration_type + "\n")
                for ch, values_list in enumerate(data):
                    f.write("CH %d BEGIN\n" % (ch, ))
                    for values in values_list:
                        f.write(','.join([str(x) for x in values]) + "\n")
                    f.write("CH %d END\n" % (ch, ))

    def _read_duration(self, filename, wav_filename):
        with open(filename, 'r') as f:
            result = []
            for duration_type in ('VAD', 'UTTERANCE', 'TURN'):
                if f.readline().rstrip() != duration_type:
                    raise RuntimeError('type not matched')
                result_for_type = []
                for ch in range(2):
                    if f.readline()[:2] != 'CH':
                        raise RuntimeError('channel info is not given')
                    result_for_channel = []
                    while True:
                        line = f.readline()
                        if line[:2] == 'CH':
                            break
                        values = line.rstrip().split(',')
                        # 最初の2つはintに変換しておく
                        values[0] = int(values[0])
                        values[1] = int(values[1])
                        result_for_channel.append(values)
                    result_for_type.append(result_for_channel)
                result.append(result_for_type)
        data = {'VAD': result[0], 'UTTERANCE': result[1], 'TURN': result[2]}
        return DurationInfo(data, wav_filename)
