# utf-8


def flatten_eval_summary(eval_summary):
    import copy
    r = copy.copy(eval_summary)
    del r['err_stats']
    del r['errs_start']
    del r['errs_end']
    err_stats = eval_summary['err_stats']
    for name in ('start', 'start_negative', 'start_positive', 'end',
                 'end_negative', 'end_positive'):
        for sname in ('count', 'average', 'std', 'median', 'per90', 'max'):
            flatten_name = "{}_{}".format(name, sname)
            r[flatten_name] = err_stats[name][sname]
    return r


def pprint_eval_summary(eval_summary):
    r = eval_summary
    print("単純適合率: {:.4f}".format(r['simple_precision']))
    print("単純再現率: {:.4f}".format(r['simple_recall']))
    print("非発話区間誤棄却率: {:.4f}".format(r['untarget_false_negative']))
    print("発話区間誤棄却率: {:.4f}".format(r['target_false_negative']))
    print("非発話区間誤検出率: {:.4f}".format(r['untarget_false_positive']))
    print("発話区間誤検出率: {:.4f}".format(r['target_false_positive']))
    err_stats = r['err_stats']
    for name, jname in (('start', '開始'), ('start_negative', '開始先行'),
                        ('start_positive', '開始遅延'), ('end', '終了'),
                        ('end_negative', '終了先行'), ('end_positive', '終了遅延')):
        for sname, jsname in (('count', 'カウント'), ('average', '平均値'),
                              ('std', '標準偏差'), ('median', '中央値'),
                              ('per90', '90パーセンタイル'), ('max', '最大値')):
            print("{}誤差{}: {:.1f}".format(jname, jsname,
                                          err_stats[name][sname]))


def summarize_eval(eval_result, verbose=False):
    r = eval_result
    simple_precision = r['dur_correct'] / r['dur_detection']
    simple_recall = r['dur_correct'] / r['dur_target']
    untarget_false_negative = r['cnt_lacked_untarget'] / r['cnt_untarget']
    target_false_negative = r['cnt_lacked_target'] / r['cnt_target']
    untarget_false_positive = r['cnt_insertion_untarget'] / r[
        'cnt_detection_untarget']
    target_false_positive = r['cnt_insertion_target'] / r[
        'cnt_detection_target']
    # 誤差の統計値（平均，標準偏差，中央値，90パーセンタイル，最大）
    import numpy as np
    res = np.array(r['errs_start'])
    resn = -np.array(res[res < 0])
    resp = np.array(res[res > 0])
    ree = np.array(r['errs_end'])
    reen = -np.array(ree[ree < 0])
    reep = np.array(ree[ree > 0])
    err_stats = dict()
    for name, values in (('start', res), ('start_negative', resn),
                         ('start_positive', resp), ('end', ree),
                         ('end_negative', reen), ('end_positive', reep)):
        err_stats[name] = dict(
            count=len(values),
            average=np.average(values),
            std=np.std(values),
            median=np.median(values),
            per90=np.percentile(values, 90) if len(values) > 0 else 0,
            max=np.max(values) if len(values) > 0 else 0)

    if verbose:
        print("単純適合率: {:.2f}".format(simple_precision))
        print("単純再現率: {:.2f}".format(simple_recall))
        print("非発話区間誤棄却率: {:.2f}".format(untarget_false_negative))
        print("発話区間誤棄却率: {:.2f}".format(target_false_negative))
        print("非発話区間誤検出率: {:.2f}".format(untarget_false_positive))
        print("発話区間誤検出率: {:.2f}".format(target_false_positive))
        for name, jname in (('start', '開始'), ('start_negative', '開始先行'),
                            ('start_positive', '開始遅延'), ('end', '終了'),
                            ('end_negative', '終了先行'), ('end_positive',
                                                       '終了遅延')):
            for sname, jsname in (('count', 'カウント'), ('average', '平均値'),
                                  ('std', '標準偏差'), ('median', '中央値'),
                                  ('per90', '90パーセンタイル'), ('max', '最大値')):
                print("{}誤差{}: {:.1f}".format(jname, jsname,
                                              err_stats[name][sname]))

    r = dict(simple_precision=simple_precision,
             simple_recall=simple_recall,
             untarget_false_negative=untarget_false_negative,
             target_false_negative=target_false_negative,
             untarget_false_positive=untarget_false_positive,
             target_false_positive=target_false_positive,
             err_stats=err_stats,
             errs_start=r['errs_start'],
             errs_end=r['errs_end'])
    return r


def eval_interval(y, t):
    """区間検出問題の評価を行う．
    
    Args:
      y (list): 検出結果の区間情報のリスト
        y[0] は最初の区間の (開始時間, 終了時間) のタプル．
        検出した区間数分ある．
        時間の単位は，tと揃って入ればなんでも良い．
        標準的には秒かミリ秒だろう．
      t (list): yと同じ形式の正解区間情報のリスト
    """
    # --- 積算して行く各値 ---
    # ### 単純適合率，単純再現率を計算するための物 ###
    # 総検出時間
    dur_detection = 0.0
    # 総正解時間
    dur_target = 0.0
    # 適合時間
    dur_correct = 0.0
    # ### 発話区間消失率（ユーザ発話脱落率），非発話区間消失率（システム発話脱落率） ###
    # ターゲット区間数
    cnt_target = 0
    # 非ターゲット区間数（ターゲット区間が0から始まらない限り，ターゲット区間数と同じ）
    cnt_untarget = 0
    # 検出区間数
    cnt_detection_target = 0
    # 検出非区間数
    cnt_detection_untarget = 0
    # 脱落ターゲット区間
    cnt_lacked_target = 0
    # 脱落非ターゲット区間
    cnt_lacked_untarget = 0
    # 挿入ターゲット区間
    cnt_insertion_target = 0
    # 挿入非ターゲット区間
    cnt_insertion_untarget = 0
    # ### 誤差情報を蓄える．
    # 誤差は全てターゲットから見た検出の相対時間（早ければマイナス，遅ければプラス）
    # 開始誤差
    errs_start = []
    # 終了誤差
    errs_end = []

    # --- ---
    # tのインデクス
    i = 0
    # yのインデクス
    j = 0

    if len(t) == 0 or len(y) == 0:
        if len(t) > 0:
            # 全発話区間を取り逃がしているとみなせる
            for t_s, t_e in t:
                dur_target += t_e - t_s
                cnt_untarget += 1
                cnt_target += 1
                cnt_lacked_target += 1
        elif len(y) > 0:
            # 全ての発話区間が誤検出とみなせる
            for y_s, y_e in y:
                dur_detection += y_e - y_s
                cnt_detection_target += 1
                cnt_detection_untarget += 1
                cnt_insertion_target += 1
        r = dict(
            dur_detection=dur_detection,
            dur_target=dur_target,
            dur_correct=dur_correct,
            cnt_target=cnt_target,
            cnt_untarget=cnt_untarget,
            cnt_detection_target=cnt_detection_untarget,
            cnt_detection_untarget=cnt_detection_untarget,
            cnt_lacked_target=cnt_lacked_target,
            cnt_lacked_untarget=cnt_lacked_untarget,
            cnt_insertion_target=cnt_insertion_target,
            cnt_insertion_untarget=cnt_insertion_untarget,
            errs_start=errs_start,
            errs_end=errs_end,
        )
        return r

    # --- 最初のターゲット区間情報 ---
    t_s, t_e = t[i]
    while t_s < 0.0:
        i += 1
        t_s, t_e = t[i]
    dur_target += t_e - t_s
    cnt_target += 1
    if t_s > 0.0:
        cnt_untarget += 1
    # --- 最初の検出区間情報 ---
    y_s, y_e = y[j]
    while y_s < 0.0:
        j += 1
        y_s, y_e = t[j]
    dur_detection += y_e - y_s
    cnt_detection_target += 1
    if y_s > 0.0:
        cnt_detection_untarget += 1

    ci = -1  # 確認が取れた正解区間の番号
    cj = -1  # 確認が取れた検出区間の番号
    while i < len(t) or j < len(y):
        # print("t-> {}({}), y -> {}({})".format(i, ci, j, cj))
        if t_e > y_e:
            # 正解区間の終了の方が後ろにある場合
            if y_e <= t_s:
                # 検出区間の終了が正解区間の開始以前にある場合
                # -> 検出区間は全く重なっていない
                # 当該検出区間の対応が取れていなければ挿入カウントを増やす
                if j < len(y) and cj < j:
                    cnt_insertion_target += 1
                    cj = j
                if j >= len(y) and ci < i:
                    cnt_lacked_target += 1
                    ci = i
            elif j < len(y):
                # 検出区間の終了が正解区間の開始以降にある場合
                # -> 検出区間の一部が重なっている
                if y_s <= t_s:
                    # 検出区間の開始が正解区間の開始より前にある場合
                    # -> 開始先行誤差を持つ
                    if cj < j and ci < i:
                        errs_start.append(y_s - t_s)
                    # -> 重なっている部分は総正解時間
                    dur_correct += y_e - t_s
                else:
                    # 検出区間の開始が正解区間の開始より後にある場合
                    # -> 開始遅延誤差を持つ
                    if cj < j and ci < i:
                        errs_start.append(y_s - t_s)
                    # -> 重なっている部分は総正解時間
                    dur_correct += y_e - y_s
                if j < len(y) - 1 and y[j + 1][0] < t_e:
                    # 次の検出区間の開始が正解区間の終了より前にあれば，
                    # 非発話区間が挿入している
                    cnt_insertion_untarget += 1
                else:
                    # この検出区間の終了が正解区間の終了に対応する．
                    if y_e <= t_e:
                        # 検出区間の終了が正解区間の終了以前にある場合
                        # -> 終了先行誤差を持つ
                        errs_end.append(y_e - t_e)
                    else:
                        # 検出区間の終了が正解区間の終了より後にある場合
                        # -> 終了遅延誤差を持つ
                        errs_end.append(y_e - t_e)
                cj = j
                ci = i
            # 次のjに移動する
            if j < len(y):
                j += 1
                if j < len(y):
                    y_s, y_e = y[j]
                    dur_detection += y_e - y_s
                    cnt_detection_untarget += 1
                    cnt_detection_target += 1
            elif i < len(t):
                i += 1
                if i < len(t):
                    t_s, t_e = t[i]
                    dur_target += t_e - t_s
                    cnt_target += 1
                    cnt_untarget += 1
        else:
            # 検出区間の方が後ろにある場合
            if t_e <= y_s:
                # 正解区間の終了が検出区間の開始以前にある場合
                # -> 正解区間は全く重なっていない
                # 当該正解区間が確認を取れていなければ脱落カウントを増やす
                if i < len(t) and ci < i:
                    cnt_lacked_target += 1
                    ci = i
                if i >= len(t) and cj < j:
                    cnt_insertion_target += 1
                    cj = j
            elif i < len(t):
                # 正解区間の終了が検出区間の開始以降にある場合
                # -> 正解区間の一部が重なっている
                if y_s <= t_s:
                    # 検出区間の開始が正解区間の開始より前にある場合
                    # -> 開始先行誤差を持つ
                    if ci < i and cj < j:
                        errs_start.append(y_s - t_s)
                    # -> 重なっている部分は総正解時間
                    dur_correct += t_e - t_s
                else:
                    # 検出区間の開始が正解区間の開始より前にある場合
                    # -> 開始遅延誤差を持つ
                    if ci < i and cj < j:
                        errs_start.append(y_s - t_s)
                    # -> 重なっている部分は総正解時間
                    dur_correct += t_e - y_s
                if i < len(t) - 1 and t[i + 1][0] < y_e:
                    # 次の正解区間の開始が検出区間の終了より前にあれば，
                    # 非発話区間が脱落している
                    cnt_lacked_untarget += 1
                else:
                    # この正解区間の終了が検出区間の終了に対応する．
                    if y_e <= t_e:
                        # 検出区間の終了が正解区間の終了以前にある場合
                        # -> 終了先行誤差を持つ
                        errs_end.append(y_e - t_e)
                    else:
                        # 検出区間の終了が正解区間の終了より後にある場合
                        # -> 終了遅延誤差を持つ
                        errs_end.append(y_e - t_e)
                ci = i
                cj = j
            # 次のiに移動する
            if i < len(t):
                i += 1
                if i < len(t):
                    t_s, t_e = t[i]
                    dur_target += t_e - t_s
                    cnt_target += 1
                    cnt_untarget += 1
            elif j < len(y):
                j += 1
                if j < len(y):
                    y_s, y_e = y[j]
                    dur_detection += y_e - y_s
                    cnt_detection_untarget += 1
                    cnt_detection_target += 1

    r = dict(
        dur_detection=dur_detection,
        dur_target=dur_target,
        dur_correct=dur_correct,
        cnt_target=cnt_target,
        cnt_untarget=cnt_untarget,
        cnt_detection_target=cnt_detection_untarget,
        cnt_detection_untarget=cnt_detection_untarget,
        cnt_lacked_target=cnt_lacked_target,
        cnt_lacked_untarget=cnt_lacked_untarget,
        cnt_insertion_target=cnt_insertion_target,
        cnt_insertion_untarget=cnt_insertion_untarget,
        errs_start=errs_start,
        errs_end=errs_end,
    )
    return r


if __name__ == '__main__':
    from pprint import pprint

    print("TEST 1: 両方空")
    y = tuple()
    t = tuple()
    pprint(eval_interval(y, t))

    print("----")
    print("TEST 2: ターゲットのみ値がある")
    y = tuple()
    t = ((1.0, 2.0), (3.0, 4.0))
    pprint(eval_interval(y, t))

    print("----")
    print("TEST 3: 検出のみ値がある")
    y = ((1.0, 2.0), (3.0, 4.0))
    t = tuple()
    pprint(eval_interval(y, t))

    print("----")
    print("TEST 4: 両方一致している")
    y = ((1.0, 2.0), (3.0, 4.0))
    t = ((1.0, 2.0), (3.0, 4.0))
    pprint(eval_interval(y, t))

    print("----")
    print("TEST 5: 先行して外れている検出区間がいくつかある")
    y = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0))
    t = ((5.0, 6.0), (7.0, 8.0))
    pprint(eval_interval(y, t))

    print("----")
    print("TEST 6: 先行して外れているターゲット区間がいくつかある")
    y = ((5.0, 6.0), (7.0, 8.0))
    t = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0))
    pprint(eval_interval(y, t))

    print("----")
    print("TEST 7: 間に外れている検出区間がいくつかある")
    y = ((5.0, 6.0), (7.0, 8.0))
    t = ((1.0, 2.0), (3.0, 4.0), (9.0, 10.0), (11.0, 12.0))
    pprint(eval_interval(y, t))

    print("----")
    print("TEST 8: 間に外れているターゲット区間がいくつかある")
    y = ((1.0, 2.0), (3.0, 4.0), (9.0, 10.0), (11.0, 12.0))
    t = ((5.0, 6.0), (7.0, 8.0))
    pprint(eval_interval(y, t))

    print("----")
    print("TEST 9: 非ターゲット区間の誤検出がいくつかある")
    y = ((1.0, 2.0), (3.0, 4.0), (4.1, 5.0))
    t = ((1.0, 5.0), )
    pprint(eval_interval(y, t))

    print("----")
    print("TEST 10: 非ターゲット区間の誤棄却がいくつかある")
    y = ((1.0, 5.0), )
    t = ((1.0, 2.0), (3.0, 4.0), (4.1, 5.0))
    pprint(eval_interval(y, t))

    print("----")
    print("TEST 11: 区間開始先行誤差")
    y = ((1.0, 3.0), (4.0, 6.0))
    t = ((2.0, 3.0), (5.0, 6.0))
    pprint(eval_interval(y, t))

    print("----")
    print("TEST 12: 区間開始遅延誤差")
    y = ((2.0, 3.0), (5.0, 6.0))
    t = ((1.0, 3.0), (4.0, 6.0))
    pprint(eval_interval(y, t))

    print("----")
    print("TEST 13: 区間終了先行誤差")
    y = ((1.0, 2.0), (4.0, 5.0))
    t = ((1.0, 3.0), (4.0, 6.0))
    pprint(eval_interval(y, t))

    print("----")
    print("TEST 14: 区間終了遅延誤差")
    y = ((1.0, 3.0), (4.0, 6.0))
    t = ((1.0, 2.0), (4.0, 5.0))
    pprint(eval_interval(y, t))

    print("TEST 15: 正解区間内に検出区間が複数ある形で，開始誤差，終了誤差あり")
    y = ((1.0, 3.0), (4.0, 5.0), (6.0, 7.0), (8.0, 10.0))
    t = ((2.0, 9.0), )
    pprint(eval_interval(y, t))

    print("TEST 16: 検出区間内に正解区間が複数ある形で，開始誤差，終了誤差あり")
    y = ((2.0, 9.0), )
    t = ((1.0, 3.0), (4.0, 5.0), (6.0, 7.0), (8.0, 10.0))
    pprint(eval_interval(y, t))
