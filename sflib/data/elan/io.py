# coding: utf-8
from xml.dom import minidom
from datetime import datetime
from pytz import timezone
from pandas import DataFrame, read_csv
import re
from xml.etree import ElementTree

from . import data


def read_from_eaf(filename):
    info = data.EafInfo()

    # パージング
    dom = minidom.parse(filename)

    # ドキュメント要素の取得
    doc = dom.getElementsByTagName('ANNOTATION_DOCUMENT')[0]

    # 著者の取得
    info.author = doc.getAttribute('AUTHOR')

    # 日付の取得
    dateString = doc.getAttribute('DATE')
    dt = datetime.strptime(dateString[:-6], '%Y-%m-%dT%H:%M:%S')
    dt = dt.astimezone(timezone('Asia/Tokyo'))
    info.date = dt

    # メディア情報の取得
    for m in dom.getElementsByTagName('MEDIA_DESCRIPTOR'):
        url = m.getAttribute('MEDIA_URL')
        mime_type = m.getAttribute('MIME_TYPE')
        relative_media_url = m.getAttribute('RELATIVE_MEDIA_URL')
        info.append_media(url, mime_type, relative_media_url)

    # 時間スロット情報を取得
    ts_dict = {}
    for ts in dom.getElementsByTagName('TIME_SLOT'):
        id = ts.getAttribute('TIME_SLOT_ID')
        value = int(ts.getAttribute('TIME_VALUE'))
        ts_dict[id] = value

    # アノテーション情報を取得（時間も埋め込む）
    for t in dom.getElementsByTagName('TIER'):
        tier_id = t.getAttribute('TIER_ID')
        info.append_tier(tier_id)
        for a in t.getElementsByTagName('ANNOTATION'):
            for aa in a.getElementsByTagName('ALIGNABLE_ANNOTATION'):
                ts_start = aa.getAttribute('TIME_SLOT_REF1')
                ts_end = aa.getAttribute('TIME_SLOT_REF2')
                time_start = ts_dict[ts_start]
                time_end = ts_dict[ts_end]
                value = aa.getElementsByTagName('ANNOTATION_VALUE')[0]
                text = ''
                for n in value.childNodes:
                    if n.nodeType == n.TEXT_NODE:
                        text += n.nodeValue
                # import ipdb; ipdb.set_trace()
                info.append_annotation(tier_id, time_start, time_end, text)

    return info


def convert_to_dataframe(info, name_pattern=None):
    name_list = []
    start_list = []
    end_list = []
    value_list = []
    for tier in info.tiers:
        name = tier['name']
        if name_pattern is not None:
            if re.match(name_pattern, name) is None:
                continue
        for a in tier['annotations']:
            name_list.append(name)
            start_list.append(a[0])
            end_list.append(a[1])
            value_list.append(a[2])
    df = DataFrame(
        data={
            'name': name_list,
            'start': start_list,
            'end': end_list,
            'value': value_list
        })
    return df


def write_to_csv(info, filename):
    df = convert_to_dataframe(info)
    df.to_csv(filename)


def write_to_eaf(info, filename):
    root = ElementTree.Element(
        'ANNOTATION_DOCUMENT', {
            'AUTHOR':
            info.author,
            'DATE':
            info.date.isoformat(),
            'FORMAT':
            '3.0',
            'VERSION':
            '3.0',
            'xmlns:xsi':
            'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:noNamespaceSchemaLocation':
            'http://www.mpi.nl/tools/elan/EAFv3.0.xsd'
        })
    header = ElementTree.SubElement(root, 'HEADER', {
        'MEDIA_FILE': '',
        'TIME_UNITS': 'milliseconds'
    })
    for m in info.media:
        ElementTree.SubElement(
            header, 'MEDIA_DESCRIPTOR', {
                'MEDIA_URL': m['url'],
                'MIME_TYPE': m['mime_type'],
                'RELATIVE_MEDIA_URL': m['relative_media_url'],
            })

    # アノテーション情報を，書き出し可能な形に変換する
    tiers = []
    time_slots = []
    for t in info.tiers:
        tier_name = t['name']
        annotaions = []
        for a in t['annotations']:
            t_start = str(a[0])
            t_end = str(a[1])
            ts_start = 'ts{}'.format(len(time_slots) + 1)
            ts_end = 'ts{}'.format(len(time_slots) + 2)
            time_slots.append((ts_start, t_start))
            time_slots.append((ts_end, t_end))
            annotaions.append((ts_start, ts_end, a[2]))
        tiers.append({'name': tier_name, 'annotations': annotaions})

    time_order = ElementTree.SubElement(root, 'TIME_ORDER')
    for ts in time_slots:
        name = ts[0]
        value = ts[1]
        ElementTree.SubElement(time_order, 'TIME_SLOT', {
            'TIME_SLOT_ID': name,
            'TIME_VALUE': value
        })

    count = 1
    for tier in tiers:
        tier_elem = ElementTree.SubElement(
            root, 'TIER', {
                'LINGUISTIC_TYPE_REF': 'default-lt',
                'TIER_ID': tier['name']
            })
        for a in tier['annotations']:
            a_elem = ElementTree.SubElement(tier_elem, 'ANNOTATION')
            aa_elem = ElementTree.SubElement(
                a_elem, 'ALIGNABLE_ANNOTATION', {
                    'ANNOTATION_ID': 'a{}'.format(count),
                    'TIME_SLOT_REF1': a[0],
                    'TIME_SLOT_REF2': a[1]
                })
            av_elem = ElementTree.SubElement(aa_elem, 'ANNOTATION_VALUE')
            av_elem.text = a[2]
            count += 1
    ElementTree.SubElement(
        root, 'LINGUISTIC_TYPE', {
            'GRAPHIC_REFERENCES': 'false',
            'LINGUISTIC_TYPE_ID': 'default-lt',
            'TIME_ALIGNABLE': 'true'
        })
    ElementTree.SubElement(
        root, 'CONSTRAINT', {
            'DESCRIPTION':
            'Time subdivision of parent annotation\'s time interval, no time gaps allowed within this interval',
            'STEREOTYPE':
            'Time_Subdivision'
        })
    ElementTree.SubElement(
        root, 'CONSTRAINT', {
            'DESCRIPTION':
            'Symbolic subdivision of a parent annotation. Annotations refering to the same parent are ordered',
            'STEREOTYPE':
            'Symbolic_Subdivision'
        })
    ElementTree.SubElement(
        root, 'CONSTRAINT', {
            'DESCRIPTION': '1-1 association with a parent annotation',
            'STEREOTYPE': 'Symbolic_Association'
        })
    ElementTree.SubElement(
        root, 'CONSTRAINT', {
            'DESCRIPTION':
            'Time alignable annotations within the parent annotation\'s time interval, gaps are allowed',
            'STEREOTYPE':
            'Included_In'
        })

    xml = minidom.parseString(ElementTree.tostring(root, encoding='unicode'))
    pretty_xml = xml.toprettyxml(encoding='utf-8')
    with open(filename, 'wb') as f:
        f.write(pretty_xml)


def read_from_csv(filename, author='', date=None, media=None):
    """CSVファイルを読み込みEafInfoを生成する．

    引数
    ----
    filename は読みこむCSVファイルのファイル名．
    CSVファイルは1行目がヘッダである必要がある．
    各行には注釈情報が入っている必要があり，
    1列目はインデクス，2列目は注釈層名，
    3列目が開始時間（ミリ秒），4列目が終了時間（ミリ秒），
    5列目が注釈の値（空も可能である）
    author は著者名，dateは日付（datetimeオブジェクトである必要がある），
    mediaはメディア情報（url, mime_type, relative_media_urlをキーとして持つディクショナリのリスト）．
    author以降はいずれもオプションで，空やNoneでよい．
    
    戻り値
    ------
    EafInfoのインスタンス
    """
    info = data.EafInfo()

    if isinstance(author, str): 
        info.author = author
    else:
        info.author = ''

    if date is None or not isinstance(date, datetime):
        info.date = datetime.now(timezone('Asia/Tokyo'))
    else:
        info.date = date
        
    if media is not None:
        for m in media:
            info.append_media(m['url'], m['mime_type'], m['relative_media_url'])
            
    df = read_csv(filename, index_col=0)
    tier_names = df['name'].unique()
    for tier_name in tier_names:
        df_sub = df[df['name'] == tier_name]
        annotations = []
        for i, row in df_sub.iterrows():
            value = row['value']
            info.append_annotation(tier_name, row['start'], row['end'], row['value'])
            
    return info

