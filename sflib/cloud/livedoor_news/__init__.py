# cloudってわけでもないけどネットワーク関係ということで...
import urllib.request
import lxml.html
import sqlite3
import os
import re
import datetime
import traceback
from os import path
from ... import config


def _get_dom(url, encoding='euc-jp') -> lxml.html.HtmlElement:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as res:
        body = res.read()
    body = body.decode(encoding)
    return lxml.html.fromstring(body)

    
def _open_article_db(create=False):
    filename = path.join(config.get_package_data_dir(__package__),
                         'article.db')
    
    if create is True and path.exists(filename):
        os.unlink(filename)
    table_creation_required = False
    if not path.exists(filename):
        table_creation_required = True
        
    conn = sqlite3.connect(filename, 
                           detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
    sqlite3.dbapi2.converters['DATETIME'] = sqlite3.dbapi2.converters['TIMESTAMP']

    if table_creation_required:
        c = conn.cursor()
    
        c.execute('CREATE TABLE IF NOT EXISTS ' +
                  'category(parent_name text, name text, url text)')
        c.execute('CREATE TABLE IF NOT EXISTS ' +
                  'article_title(cat_row_id int, article_id int, ' +
                  'title text, datetime datetime)')
        c.execute('CREATE TABLE IF NOT EXISTS ' +
                  'topics_detail(article_id int, ' +
                  'title text, datetime datetime, ' +
                  'type int, content text)')
        c.execute('CREATE TABLE IF NOT EXISTS ' +
                  'article_detail(article_id int, ' +
                  'title text, datetime datetime, ' +
                  'vendor_name text, vendor_url name, content text)')
        conn.commit()
    
    return conn


def fetch_category_info_list():
    """LivedoorニュースからWeb経由でカテゴリ情報を取得する．

    Returns:
      (親カテゴリ名, カテゴリ名，リストのURL，略称)のリスト
      親カテゴリ名: 対応する親カテゴリの名前．自身がトップカテゴリの場合は'null'.
      カテゴリ名: カテゴリの名前（例: 主要，国内，社会）
      リストのURL: ニュース一覧のURL
    """
    # 基本，親と子の2階層しかない．
    # 親のリストを「トップ」から抜き取り，順に処理する．
    url = "https://news.livedoor.com/"
    dom = _get_dom(url)
    dom_nav_inner = dom.find_class('navInner')
    dom_ul = dom_nav_inner[0].find('nav/ul')

    parents = []
    # 最初は「トップ」なので飛ばす
    for d in dom_ul.find_class("parent")[1:]:
        dom_a = d.find('a')
        parents.append([dom_a.text, dom_a.attrib['href']])

    categories = []
    for p_name, p_url in parents:
        categories.append(['null', p_name, p_url])
        dom = _get_dom(p_url)
        dom_nav_inner = dom.find_class('navInner')
        if len(dom_nav_inner) == 0:
            continue
        dom_uls = dom_nav_inner[0].find('nav/ul').find_class('child')
        if len(dom_uls) == 0:
            continue
        for d in dom_uls[0].findall('li'):
            dom_a = d.find('a')
            categories.append((p_name,
                               dom_a.text,
                               dom_a.attrib['href'],))
    return categories


def update_category_table(cursor, categories):
    for p_name, name, url in categories:
        cursor.execute('SELECT EXISTS(SELECT 1 FROM category ' +
                       'WHERE url=?)', (url,))
        if cursor.fetchone()[0] == 1:
            continue
        cursor.execute('INSERT INTO category VALUES(?, ?, ?)', (p_name, name, url))
    
    
def read_category_list(cursor):
    cursor.execute('SELECT parent_name, name, url, rowid FROM category')
    categories = cursor.fetchall()
    
    return categories


def fetch_article_title_list(url, p=1):
    dom = _get_dom(url + '?p={}'.format(p))

    dom_article_list = dom.find_class('articleList')[0]
    dom_articles = dom_article_list.findall('li/a')

    articles = []
    for d in dom_articles:
        article_url = d.attrib['href']
        article_title = d.find_class('articleListTtl')[0].text
        article_datetime = d.find_class('articleListDate')[0].attrib['datetime']
        articles.append((article_url, article_title, article_datetime))
    return articles
    

__article_id_matcher = re.compile(r'/([0-9]+)/$')


def extract_article_id_from_url(url):
    m = __article_id_matcher.search(url)
    if m is None:
        return None
    return int(m[1])


def insert_article_title(cursor, cat_id, article_id, title, datetime):
    cursor.execute('SELECT EXISTS(SELECT 1 FROM article_title ' +
                   'WHERE cat_row_id=? AND article_id=?)', (cat_id, article_id))
    if cursor.fetchone()[0] == 1:
        return
    cursor.execute('INSERT INTO article_title VALUES(?, ?, ?, ?)',
                   (cat_id, article_id, title, datetime))
    

def insert_article_title_list(cursor, cat_id, article_titles):
    for article_title in article_titles:
        id = extract_article_id_from_url(article_title[0])
        title = article_title[1]
        datetime = article_title[2]
        insert_article_title(cursor, cat_id, id, title, datetime)

        
def article_id_to_topics_detail_url(id):
    return 'https://news.livedoor.com/topics/detail/{:08d}/'.format(id)


def fetch_topics_detail(id):
    url = article_id_to_topics_detail_url(id)
    dom = _get_dom(url)

    # タイトルを取得
    title = dom.find_class('topicsTtl')[0].find('a').text
    # 日付を取得
    dt = dom.find_class('topicsTime')[0].text
    dt = datetime.datetime.strptime(dt, '%Y年%m月%d日 %H時%M分')
    
    # summaryListというクラスを見つけにいく
    dom_summary_list = dom.find_class('summaryList')
    text = ''
    detail_type = 0
    if len(dom_summary_list) > 0:
        # summaryListがある場合は，要約を取得する
        for l in dom_summary_list[0].findall('li'):
            text += l.text + "。\n"
        detail_type = 3
    else:
        # summaryListが無い場合は，文章として要約を取得する．
        # こちらは要約というよりは冒頭を抜き出したものになる
        text = dom.find_class('articleBody')[0].text
    return (id, title, dt, detail_type, text)


def is_id_in_topics_detail(cursor, id):
    cursor.execute('SELECT EXISTS(SELECT 1 FROM topics_detail ' +
                   'WHERE article_id=?)', (id,))
    if cursor.fetchone()[0] == 1:
        return True
    return False
    

def insert_topics_detail(cursor, detail):
    article_id = detail[0]
    if is_id_in_topics_detail(cursor, article_id):
        return
    cursor.execute('INSERT INTO topics_detail VALUES(?, ?, ?, ?, ?)', detail)

    
def article_id_to_article_detail_url(id):
    return 'https://news.livedoor.com/article/detail/{:08d}/'.format(id)


def fetch_article_detail(id):
    url = article_id_to_article_detail_url(id)
    dom = _get_dom(url)
    
    # タイトルを取得
    article_title = dom.find_class('articleTtl')[0].text
    # 日時を取得
    article_datetime = dom.find_class('articleDate')[0].attrib['content']
    # ベンダを取得
    try:
        article_vendor_name = dom.find_class('articleVender')[0].find('a/span/span').text.rstrip()
        article_vendor_url = dom.find_class('articleVender')[0].find('a').attrib['href']
    except Exception:
        article_vendor_name = ''
        article_vendor_url = ''

    # 記事内容を取得
    text = ''
    for d in dom.find_class('articleBody')[0].find('span'):
        if d.tag == 'br':
            text += '\n'
        if d.text is not None:
            text += d.text
        if d.tail is not None:
            text += d.tail
            
    return (id, article_title, article_datetime,
            article_vendor_name, article_vendor_url, text)
    

def is_id_in_article_detail(cursor, id):
    cursor.execute('SELECT EXISTS(SELECT 1 FROM article_detail ' +
                   'WHERE article_id=?)', (id,))
    if cursor.fetchone()[0] == 1:
        return True
    return False


def insert_article_detail(cursor, detail):
    article_id = detail[0]
    if is_id_in_article_detail(cursor, article_id):
        return
    cursor.execute('INSERT INTO article_detail VALUES(?, ?, ?, ?, ?, ?)', detail)

    
def fetch_and_save_all(p=1):
    conn = _open_article_db()

    categories = fetch_category_info_list()
    update_category_table(conn.cursor(), categories)

    categories = read_category_list(conn.cursor())

    for _, cat_name, cat_url, cat_id in categories:
        if cat_name in ['今日のニュース', 'ランキング', '話題のニュース', 'インタビュー']:
            continue
        print("「{}」カテゴリの処理を開始".format(cat_name))
        try:
            article_titles = fetch_article_title_list(cat_url)
            print("「{}」カテゴリの一覧を取得（計{}件）".format(cat_name, len(article_titles)))
            insert_article_title_list(conn.cursor(), cat_id, article_titles)
            conn.commit()
            # print("article_title テーブルへ保存")
            for url, title, _ in article_titles:
                article_id = extract_article_id_from_url(url)
                print("{} （{}） ... ".format(article_id, title), end='')
                if is_id_in_topics_detail(conn.cursor(), article_id):
                    # print("topics_detail テーブルへは保存済", end='')
                    print("skip topic detail ... ", end='')
                else:
                    try:
                        td = fetch_topics_detail(article_id)
                        # print("サマリを取得")
                        insert_topics_detail(conn.cursor(), td)
                        # print("topics_detail テーブルへ保存", end='')
                        print("save topic detail ...", end='')
                    except Exception:
                        # print("サマリの取得に失敗", end='')
                        print("fail to fetch topic detail ...", end='')
                if is_id_in_article_detail(conn.cursor(), article_id):
                    # print("article_detail テーブルへは保存済")
                    print("skip article detail")
                else:
                    try:
                        ad = fetch_article_detail(article_id)
                        # print("全文を取得")
                        insert_article_detail(conn.cursor(), ad)
                        # print("article_detail テーブルへ保存")
                        print("save article detail")
                    except Exception:
                        # print("全文の取得に失敗")
                        print("fail to fetch article detail")
                conn.commit()
        except Exception:
            # traceback.print_exc()
            raise
        
    conn.close()

    
def fetch_and_save_all_loop(time_to_sleep=60*10, max_page=30):
    import time
    while True:
        for p in range(1, max_page + 1):
            print("begin page {} ... ".format(p))
            fetch_and_save_all(p=p)
            print("sleep {} seconds... ".format(time_to_sleep))
            time.sleep(time_to_sleep)
