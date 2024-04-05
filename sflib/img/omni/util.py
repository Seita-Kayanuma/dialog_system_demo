# coding: utf-8
# ユーティリティ
import math
import time


def deg2rad(deg):
    """
    度（degree）をラジアンに変換する．

    Parameters
    ----------
    deg : int or float
        変換したい角度（degree）

    Returns
    -------
    rad : float
        変換後の角度（radian）
    """
    return deg / 180.0 * math.pi


class Timer:
    def __init__(self):
        self.t_start = None

    def start(self):
        self.t_start = time.time()

    def report(self, title=None):
        message = ''
        if title is not None:
            message += title + ": "
        ellapsed = time.time() - self.t_start
        message += "%.3fsec" % (ellapsed, )
        print(message)


class FpsCalculator:
    def __init__(self):
        self.count = 0
        self.avg = 0
        self.ptime = None

    def tick(self):
        ctime = time.time()
        if self.count > 0:
            self.avg = (self.count * self.avg + 1.0 /
                        (ctime - self.ptime)) / (self.count + 1)
        self.count += 1
        self.ptime = ctime
        return self.avg


class ResultTracker:
    def __init__(self):
        self.prev_result = []

    def track(self, result):
        if len(self.prev_result) == 0:
            self.prev_result = result
            return self.prev_result

        import copy
        result = copy.copy(result)

        # 前の結果にもっとも近い結果を採用する
        new_result = []
        for r in self.prev_result:
            if len(result) == 0:
                break
            nearest_index = None
            min_dist = None
            for i, r1 in enumerate(result):
                dx = r1[1] - r[1]
                dy = r1[2] - r[2]
                dist = dx * dx + dy * dy
                if min_dist is None or dist < min_dist:
                    nearest_index = i
                    min_dist = dist

            nearest_result = result[nearest_index]
            new_result.append(nearest_result)
            del result[nearest_index]
        new_result.extend(result)

        self.prev_result = new_result
        return new_result
