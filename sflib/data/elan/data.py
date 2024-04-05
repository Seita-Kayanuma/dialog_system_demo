# coding: utf-8
from datetime import datetime
from pytz import timezone


class EafInfo:
    """Eafファイルの情報を保持するクラス"""

    def __init__(self):
        """コンストラクタ"""
        # 著者
        self.__author = ''
        # 作成日付
        self.__date = datetime.now(timezone('Asia/Tokyo'))
        # メディア情報:
        #   アノテーション対象の音声ファイルや動画ファイルの情報
        #   media_url, mime_type, relative_media_url を
        #   キーとするディクショナリのリスト．
        #   空の場合もある．
        self.__media = []
        # アノテーション情報（注釈層情報）:
        #   注釈層情報のリストになっている．
        #   一つの注釈層は，name（注釈層名）とannotaions（注釈）を
        #   をキーとするディクショナリになっている．
        #   annotationsは，
        #     開始時間，終了時間，注釈内容
        #   が並んだタプルのリスト．
        #   （アノテーションは数が多いため，ディクショナリを避けた）
        self.__tiers = []
        # 注釈層名から__tiersのインデクスに変換するディクショナリ
        self.__tier_name_to_index = {}

    @property
    def author(self):
        return self.__author

    @author.setter
    def author(self, value):
        self.__author = value

    @property
    def date(self):
        return self.__date

    @date.setter
    def date(self, value):
        self.__date = value

    @property
    def media(self):
        return self.__media

    def append_media(self, url, mime_type, relative_media_url):
        """メディア情報を追加する"""
        self.__media.append({
            'url': url,
            'mime_type': mime_type,
            'relative_media_url': relative_media_url
        })

    @property
    def tiers(self):
        return self.__tiers

    def append_tier(self, tier_name):
        if tier_name not in self.__tier_name_to_index:
            tier = {'name': tier_name, 'annotations': []}
            self.__tier_name_to_index[tier_name] = len(self.__tiers)
            self.__tiers.append(tier)

    def append_annotation(self, tier_name, time_start, time_end, value=None):
        """アノテーションを追加する．
        注釈層が無い場合は自動的に追加される．
        """
        # 注釈層を取得．無ければ新しいものを追加
        tier = None
        if tier_name not in self.__tier_name_to_index:
            self.append_tier(tier_name)
        tier = self.__tiers[self.__tier_name_to_index[tier_name]]

        # 注釈情報を追加
        tier['annotations'].append((
            time_start,
            time_end,
            value,
        ))
