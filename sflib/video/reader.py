import cv2


class VideoReader:
    """ビデオファイルの読み込みユーティリティクラス
    """
    def __init__(self, filename):
        """
        Args:
          filename: ファイル名
        """
        self.__filename = filename
        self.__video = cv2.VideoCapture(filename)

    @property
    def framerate(self):
        return self.__video.get(cv2.CAP_PROP_FPS)

    @property
    def framecount(self):
        return self.__video.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_frame(self, index):
        if not self.__video.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise RuntimeError("frame position cannot be set to {}".format(index))
        ret, frame = self.__video.read()
        if not ret:
            raise RuntimeError("frame cannot be retrieved")
        return frame

    def get_next_frame(self):
        ret, frame = self.__video.read()
        if not ret:
            raise RuntimeError("frame cannot be retrieved")
        return frame
    
        
        
