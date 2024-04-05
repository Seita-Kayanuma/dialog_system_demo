# coding: utf-8
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
from apiclient.http import MediaFileUpload, MediaIoBaseDownload
import warnings
from os import path
from sflib import config


class GoogleDriveInterface():
    """
    Google Drive の sflib 用フォルダにアクセスして，
    ファイルをダウンロードしたりアップロードしたりするためのクラス．
    """

    DEFAULT_FOLDER_ID = '1CpWhkibEkewCoGxFcH3_zRL1m5yFwqWN'
    """str:
    shinya.fujie の My Drive の sflib_data フォルダのID 
    """

    DEFAULT_CREDENTIAL_PATH = path.join(
        config.get_package_data_dir(__package__), 'google_credentials.json')
    """str:
    Google Drive API を使うための認証コードの入ったファイルのパス．
    
    """

    DEFAULT_TOKEN_PATH = path.join(
        config.get_package_data_dir(__package__), 'google_token.json')
    """str:
    認証コード（アクセスするユーザ固有）を入れるためのファイルパス
    """

    #print(DEFAULT_CREDENTIAL_PATH)

    def __init__(self, read_only=True, folder_id=None, refresh_token=False):
        """
        Args:
          read_only(bool): 読み込み専用でアクセスするか，
            書き込みもできるようにしてアクセスするか．
            現状，藤江以外は読み込み専用でないとアクセスできない．
          folder_id(str): デフォルト以外のフォルダを利用する場合は指定する．
          refresh_token(bool): 認証をやり直したい場合はTrueを指定する．
        """
        self.read_only = read_only
        if folder_id is None:
            folder_id = GoogleDriveInterface.DEFAULT_FOLDER_ID
        self.folder_id = folder_id
        self._set_service(refresh_token)

    def _set_service(self, refresh_token):
        SCOPES = 'https://www.googleapis.com/auth/drive.readonly'
        if self.read_only is False:
            SCOPES += ' https://www.googleapis.com/auth/drive.file'
        flags = tools.argparser.parse_args(
            '--auth_host_name localhost --logging_level INFO'.split())
        warnings.filterwarnings('ignore')
        store = file.Storage(GoogleDriveInterface.DEFAULT_TOKEN_PATH)
        creds = store.get()
        if refresh_token:
            creds = None

        if not creds or creds.invalid:
            flow = client.flow_from_clientsecrets(
                GoogleDriveInterface.DEFAULT_CREDENTIAL_PATH, SCOPES)
            creds = tools.run_flow(flow, store, flags)
        self.service = build('drive', 'v3', http=creds.authorize(Http()))

    def download(self, title, filename):
        """
        ファイルをダウンロードする．
        
        Args:
          title(str): Google Drive上でのファイル名．
          filename(str): ローカルファイルのファイル名（フルパス）．
        """
        query = "name = '%s' and '%s' in parents" % (title, self.folder_id)
        result = self.service.files().list(q=query).execute()
        if len(result['files']) != 1:
            print("no files or multiple files are discovered")
            return False
        file_id = result['files'][0]['id']
        request = self.service.files().get_media(fileId=file_id)
        with open(filename, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=1048576)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(
                    "\rdownloaded %d%%." % int(status.progress() * 100),
                    end='')
        print("")
        print("download complete.")

    def download_with_filename_pattern(self, body, pattern, outdir, overwrite=True):
        """パターンにマッチするファイルを全てダウンロードする．
        
        Args:
          body(str): Google Drive API で検索する際の絞り込みに使う
                     ファイル名の部分文字列(正規表現が使えないので）
          pattern: 検索の結果引っかかったファイル名をさらにフィルタリング
                   するためのパターン（reモジュールで処理される）
          outdir: ローカルの出力ディレクトリ
          overwrite: Falseの場合はローカルに同名のファイルがある場合はダウンロードしない．
        """
        query = "name contains '{}' and '{}' in parents".format(body, self.folder_id)
        result = self.service.files().list(q=query).execute()
        print("{} files are found.".format(len(result['files'])))
        import re
        p = re.compile(pattern)
        for r in result['files']:
            filename = r['name']
            file_id = r['id']
            if p.match(filename):
                local_path = path.join(outdir, filename)
                if path.exists(local_path) and not overwrite:
                    print("{} exists already and is not downloaded".format(filename))
                    continue
                request = self.service.files().get_media(fileId=file_id)
                with open(local_path, 'wb') as fh:
                    downloader = MediaIoBaseDownload(fh, request, chunksize=1048576)
                    done = False
                    while done is False:
                          status, done = downloader.next_chunk()
                          print(
                              "\r{}: {}%.".format(filename, int(status.progress() * 100)),
                              end='')
                    print("\r{}: 100%.".format(filename))
                pass
        pass
        
    def upload(self,
               filename,
               title=None,
               mediaType='application/octet-stream'):
        """ファイルをアップロードする．
        
        Args:
          filename(str): アップロードするファイルのファイル名
          title(str): Google Drive 上でのファイル名．
            デフォルトではfilenameの中のファイル名（basename）をそのまま利用する．
          mimeType(str): 必要に応じてMIMEタイプを指定．
        """
        if title is None:
            title = path.basename(filename)
        self.delete(title)  # 重複は認めない（既存のファイルは先に削除）

        file_metadata = {
            'name': title,
            'mimeType': mediaType,
            'parents': [self.folder_id]
        }
        media = MediaFileUpload(
            str(filename),
            mimetype=mediaType,
            chunksize=1048576,
            resumable=True)
        request = self.service.files().create(
            body=file_metadata, media_body=media, fields='id')
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(
                    "\ruploaded %d%%." % int(status.progress() * 100), end='')
        print("")
        print("upload complete.")

    def delete(self, title):
        """ファイルを削除する．

        Args:
          title(str): Google Drive 上でのファイル名．
        """
        query = "name = '%s' and '%s' in parents" % (title, self.folder_id)
        result = self.service.files().list(q=query).execute()
        for r in result['files']:
            id = r['id']
            self.service.files().delete(fileId=id).execute()
