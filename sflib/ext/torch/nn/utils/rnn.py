from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_packed_sequence
import torch


def pack_sequence_with_dummy_length(sequence: list) -> (PackedSequence, torch.Tensor):
    """長さ0を含むシーケンスのパッキング．
    PackedSequenceが長さ0のテンソルが与えられると作成できないため，
    長さ0のデータが与えられたらダミーのベクトルを一つ与えつつ
    パッキングする．これを使ったForward計算は，余計な要素が一つ
    計算されるが，PaddedSequenceよりはマシだろうという発想．
    
    Args:
      sequence(list): torch.Tensorのリスト．
        要素のtorch.Tensorは，(L, D) のテンソルであること．
        要素は1つ以上であること(sequence[0]が存在数こと）．
        Lは0を含んでいてもよい．
        Dは全てのTensorで同じであること．

    Returns:
      PackedSequence: パッキングの結果
      torch.Tensor: 本当の長さを要素にもつTensor．
    """
    item_shape = sequence[0].shape[1:]
    dtype = sequence[0].dtype
    device = sequence[0].device
    lens = [x.shape[0] for x in sequence]
    seq_to_pack = []
    for x in sequence:
        if x.shape[0] == 0:
            seq_to_pack.append(
                torch.zeros((1,) + item_shape, dtype=dtype).to(device))
        else:
            seq_to_pack.append(x)
    return pack_sequence(seq_to_pack, enforce_sorted=False), torch.tensor(lens)


def unpack_sequence(packed_sequence: PackedSequence,
                    lengths: torch.Tensor = None) -> list:
    """PackedSequenceをアンパックし，テンソルのリストに戻す

    Args:
      packed_sequence(PackedSequence): 戻すシーケンスが入ったPackedSequence
      lengts(torch.Tensor): 各シーケンスの実際の長さ
        （ダミーが含まれる場合があるため，0を含む）．
        Noneの場合はPackedSequence内の長さが遵守される．
    """
    padded_seq, _lengths = pad_packed_sequence(packed_sequence)
    batches = padded_seq.shape[1]
    if lengths is None:
        lengths = _lengths
    sequences = []
    for i in range(batches):
        sequences.append(padded_seq[:lengths[i], i])
    return sequences


