# -*- coding: utf-8 -*-
"""chatgrow Discourse siralamasi + birlestirme (--order / --merge).

YAKALANDI (27.09 olcumu, forum.pardus.org.tr, want=400):
    order='posts'   -> 400 baslik, en yeni 2026-08-21, ort 31.5 yorum
    order='created' -> 400 baslik, en yeni 2026-09-28, ort  5.8 yorum
    kesisim         -> 16 baslik = forumun %96'si hic gorulmemis

'posts' yorum sayisina gore sirar, yani pencereyi en cok konusulan
basliklar doldurur ve en yeniler HIC GIRMEZ. Sonuc: pencere donuyor,
19 kosu boyunca dosya tam 30 ciftte kaldi, sonra 335'te sabitlendi.

Duzeltme: --order secenegi. 'posts' zenginligi korur, 'created' her
kosuda yeni baslik getirir. Ikisi birlikte, AYRI dosyalara yazilir
(ci workflow). Ayrica --merge: pencere kaydirildiginda yeni ciftler
silinmesin diye eski ciftler korunur.
"""
import io
import json
import os
import sys
import tempfile
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import chatgrow


class _FakeDiscourse:
    """Discourse'un order + sayfa davranisini taklit eder.

    order'a DUYARLI: gercek forumda oldugu gibi
      posts   -> yorum sayisina gore (en cok konusulan once)
      created -> tarihe gore (en yeni once)
    yani iki siralama FARKLI baslik listesi dondurur.
    """

    def __init__(self, toplam=500, per_page=30):
        self.toplam = toplam
        self.per_page = per_page
        self.sorgular = []          # (path, params)
        self.temel = None
        # i=0 en cok yorumlu/en eski, i=toplam-1 en az yorumlu/en yeni
        self._liste = sorted(range(toplam),
                             key=lambda i: (-(toplam - i), i))

    def _t(self, i):
        return {
            'id': i,
            'title': 'konu %d' % i,
            'posts_count': 3,
            'created_at': '2026-01-%02dT00:00:00Z' % (i % 28 + 1),
            'post_stream': {'posts': [
                {'post_number': 1, 'username': 'u1',
                 'cooked': '<p>%d numarali soru burada biraz uzun bir soru</p>' % i},
                {'post_number': 2, 'username': 'u2',
                 'cooked': '<p>bu da cevap icin yeterince uzun bir metin parcasidir</p>'},
            ]},
        }

    def get(self, base, path, params=None, retries=3):
        if path.startswith('/t/'):
            tid = int(path.split('/t/')[1].split('.json')[0])
            t = self._t(tid)
            return {'post_stream': {'posts': t['post_stream']['posts']}}
        params = params or {}
        self.sorgular.append((path, dict(params)))
        self.temel = params
        sayfa = int(params.get('page', 0))
        order = params.get('order')
        if order == 'created':
            # en yeni once: liste tersten
            sira = list(reversed(self._liste))
        elif order == 'posts':
            sira = self._liste
        else:
            sira = list(range(self.toplam))
        bas = sayfa * self.per_page
        if bas >= self.toplam:
            return {'topic_list': {'topics': [], 'per_page': self.per_page,
                                   'more_topics_url': None}}
        son = min(bas + self.per_page, self.toplam)
        return {'topic_list': {
            'topics': [self._t(i) for i in sira[bas:son]],
            'per_page': self.per_page,
            'more_topics_url': ('/latest?page=%d' % (sayfa + 1)
                                if son < self.toplam else None),
        }}


class TestDiscourseOrder(unittest.TestCase):
    """--order siralamayi Discourse'a gecirmeli."""

    def setUp(self):
        self.gercek = chatgrow.discourse_get
        self.uyku = chatgrow.time.sleep
        chatgrow.time.sleep = lambda *a, **k: None

    def tearDown(self):
        chatgrow.discourse_get = self.gercek
        chatgrow.time.sleep = self.uyku

    def test_order_posts_is_default(self):
        """Gecmis davranis korunur: sayfalama testleri 'posts' bekliyor."""
        sahte = _FakeDiscourse(toplam=60)
        chatgrow.discourse_get = sahte.get
        chatgrow.discourse_topics('x', None, 60)
        self.assertEqual(sahte.temel.get('order'), 'posts',
                         'varsayilan siralama posts olmali')

    def test_order_created_is_forwarded(self):
        """Asil duzeltme: 'created' Discourse'a gecmeli."""
        sahte = _FakeDiscourse(toplam=60)
        chatgrow.discourse_get = sahte.get
        chatgrow.discourse_topics('x', None, 60, order='created')
        self.assertEqual(sahte.temel.get('order'), 'created',
                         'order=created API\'ye gecmedi, pencere yine doner')

    def test_order_none_omits_param(self):
        """order=None -> parametre hic gonderilmemeli (varsayilan siralama)."""
        sahte = _FakeDiscourse(toplam=30)
        chatgrow.discourse_get = sahte.get
        chatgrow.discourse_topics('x', None, 30, order=None)
        self.assertNotIn('order', sahte.temel,
                         'order=None ikbosluk gondermemeli')

    def test_build_chats_forwards_order(self):
        sahte = _FakeDiscourse(toplam=60)
        chatgrow.discourse_get = sahte.get
        chatgrow.build_discourse_chats('x', None, 60, order='created')
        self.assertEqual(sahte.temel.get('order'), 'created')

    def test_two_windows_give_different_topics(self):
        """Iki pencere FARKLI basliklar getirmeli (yoksa biri ise yaramaz).

        PENCERE bir dilim: forum 500 baslik, istenen 60. 'posts' en cok
        konusulan 60'i, 'created' en yeni 60'i verir -> kesisim kucuk.
        Gercek olcum (forum 400 baslik): kesisim 16/400 = %4.
        """
        sahte = _FakeDiscourse(toplam=500)
        chatgrow.discourse_get = sahte.get
        a = chatgrow.build_discourse_chats('x', None, 60, order='posts')
        b = chatgrow.build_discourse_chats('x', None, 60, order='created')
        ida = {c['topic_id'] for c in a}
        idb = {c['topic_id'] for c in b}
        self.assertTrue(a and b, 'iki pencere de cift uretmeli')
        self.assertNotEqual(ida, idb, 'iki pencere ayni basliklari verdi')
        kesisim = len(ida & idb)
        self.assertLess(kesisim, 10,
                        'kesisim %d/60: pencereler aslinda ayni, '
                        'order duzeltmesi ise yaramamis' % kesisim)

    def test_pagination_still_works_with_created(self):
        """Sayfalama order'dan bagimsiz calismali."""
        sahte = _FakeDiscourse(toplam=500)
        chatgrow.discourse_get = sahte.get
        tp = chatgrow.discourse_topics('x', None, 120, order='created')
        self.assertEqual(len(tp), 120)
        self.assertIn(3, [p.get('page') for _y, p in sahte.sorgular])


class TestChatKey(unittest.TestCase):

    def test_same_topic_same_key(self):
        a = {'source': 'discourse', 'topic_id': 7, 'query': 'Nasil Yapilir?'}
        b = {'source': 'discourse', 'topic_id': 7, 'query': 'nasil yapilir?'}
        self.assertEqual(chatgrow.chat_key(a), chatgrow.chat_key(b),
                         'buyuk/kucuk harf farki ayirt etmemeli')

    def test_different_topic_different_key(self):
        a = {'source': 'discourse', 'topic_id': 7, 'query': 'soru'}
        b = {'source': 'discourse', 'topic_id': 8, 'query': 'soru'}
        self.assertNotEqual(chatgrow.chat_key(a), chatgrow.chat_key(b))

    def test_key_survives_missing_topic_id(self):
        """Reddit tarafi topic_id tasimaz; kirilmemeli."""
        a = {'source': 'reddit', 'query': 'soru'}
        b = {'source': 'reddit', 'query': 'soru'}
        self.assertEqual(chatgrow.chat_key(a), chatgrow.chat_key(b))
        self.assertIsInstance(chatgrow.chat_key(a), tuple)

    def test_edited_question_is_not_a_new_pair(self):
        """Soru metni duzeltilse de ayni konudur -> kopya olusmamali."""
        a = {'source': 'discourse', 'topic_id': 7, 'query': 'ilk hali'}
        b = {'source': 'discourse', 'topic_id': 7, 'query': 'duzeltilmis hali'}
        self.assertEqual(chatgrow.chat_key(a), chatgrow.chat_key(b),
                         'konu no ayniysa tek cift sayilmali')


class TestMergeChats(unittest.TestCase):

    def test_new_pairs_are_added(self):
        eski = [{'source': 'discourse', 'topic_id': 1, 'query': 'a'}]
        yeni = [{'source': 'discourse', 'topic_id': 2, 'query': 'b'}]
        son, e, g = chatgrow.merge_chats(eski, yeni)
        self.assertEqual(len(son), 2, 'yeni cift EKLENMELI')
        self.assertEqual((e, g), (1, 0))

    def test_existing_pair_refreshes_answer(self):
        """Ayni konu tekrar bulunursa yeni cevaplar alinmali."""
        eski = [{'source': 'discourse', 'topic_id': 1, 'query': 'a',
                 'answer': ['eski cevap']}]
        yeni = [{'source': 'discourse', 'topic_id': 1, 'query': 'a',
                 'answer': ['taze cevap']}]
        son, e, g = chatgrow.merge_chats(eski, yeni)
        self.assertEqual(len(son), 1, 'kopya olusmamali')
        self.assertEqual(son[0]['answer'], ['taze cevap'])
        self.assertEqual((e, g), (0, 1))

    def test_order_is_preserved(self):
        eski = [{'source': 'd', 'topic_id': i, 'query': 'q%d' % i}
                for i in range(5)]
        yeni = [{'source': 'd', 'topic_id': 99, 'query': 'yeni'}]
        son, _e, _g = chatgrow.merge_chats(eski, yeni)
        self.assertEqual([c['topic_id'] for c in son], [0, 1, 2, 3, 4, 99])

    def test_keep_drops_oldest(self):
        eski = [{'source': 'd', 'topic_id': i, 'query': 'q%d' % i}
                for i in range(10)]
        yeni = [{'source': 'd', 'topic_id': 99, 'query': 'yeni'}]
        son, _e, _g = chatgrow.merge_chats(eski, yeni, keep=4)
        self.assertEqual(len(son), 4)
        self.assertEqual([c['topic_id'] for c in son], [7, 8, 9, 99],
                         'en yeniler korunmali')

    def test_keep_zero_means_unlimited(self):
        eski = [{'source': 'd', 'topic_id': i, 'query': 'q%d' % i}
                for i in range(50)]
        son, _e, _g = chatgrow.merge_chats(eski, [], keep=0)
        self.assertEqual(len(son), 50)

    def test_no_duplicates_in_result(self):
        eski = [{'source': 'd', 'topic_id': 1, 'query': 'a'}]
        yeni = [{'source': 'd', 'topic_id': 1, 'query': 'a'},
                {'source': 'd', 'topic_id': 1, 'query': 'A'}]
        son, _e, _g = chatgrow.merge_chats(eski, yeni)
        self.assertEqual(len(son), 1)


class TestReadChats(unittest.TestCase):

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.yol = os.path.join(self.d, 'c.jsonl')

    def _yaz(self, satirlar):
        with io.open(self.yol, 'w', encoding='utf-8') as f:
            for s in satirlar:
                f.write(s + '\n')

    def test_missing_file_is_empty(self):
        self.assertEqual(chatgrow.read_chats(
            os.path.join(self.d, 'yok.jsonl')), [])

    def test_broken_line_is_skipped(self):
        """Boz satir tum dosyayi kaybettirmemeli."""
        self._yaz(['{"query": "a", "answer": ["x"]}', '{boz json',
                   '{"query": "b", "answer": ["y"]}'])
        r = chatgrow.read_chats(self.yol)
        self.assertEqual([c['query'] for c in r], ['a', 'b'])

    def test_line_without_query_dropped(self):
        self._yaz(['{"answer": ["sorusuz"]}', '{"query": "q"}'])
        self.assertEqual(len(chatgrow.read_chats(self.yol)), 1)

    def test_roundtrip_with_merge(self):
        """Uret -> dosya -> yeniden oku -> birlestir: veri KAYBOLMAMALI."""
        ilk = [{'source': 'discourse', 'topic_id': 1, 'query': 'a',
                'answer': ['x']}]
        with io.open(self.yol, 'w', encoding='utf-8') as f:
            for c in ilk:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
        ikinci = [{'source': 'discourse', 'topic_id': 2, 'query': 'b',
                   'answer': ['y']}]
        son, e, _g = chatgrow.merge_chats(chatgrow.read_chats(self.yol),
                                          ikinci)
        self.assertEqual(e, 1)
        with io.open(self.yol, 'w', encoding='utf-8') as f:
            for c in son:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
        self.assertEqual(len(chatgrow.read_chats(self.yol)), 2)


class TestWorkflowHasBothWindows(unittest.TestCase):
    """ci.yml iki pencereyi de cagirmali (biri dusurse veri doner)."""

    YOL = os.path.join(BASE, '.github', 'workflows', 'chatgrow.yml')

    def setUp(self):
        with io.open(self.YOL, encoding='utf-8') as f:
            self.yml = f.read()

    def test_posts_window_keeps_rich_data(self):
        self.assertIn('--order posts', self.yml)
        self.assertIn('chatgrow_discourse_pardus.jsonl', self.yml)

    def test_created_window_writes_separate_file(self):
        self.assertIn('--order created', self.yml,
                      'en yeni baslik penceresi yok, veri doner')
        self.assertIn('chatgrow_discourse_pardus_yeni.jsonl', self.yml)

    def test_both_windows_use_merge(self):
        """--merge olmazsa pencere kayinca yeni ciftler SILINIR."""
        self.assertEqual(self.yml.count('--merge'), 2,
                         'iki pencere de --merge kullanmali')

    def test_new_window_is_bigger_than_old(self):
        """Yeni pencere daha derin olmali (cevapsiz basliklar cogu)."""
        import re
        bloklar = re.findall(r'--limit (\d+).*?--out (\S+)', self.yml,
                             re.S)
        limitler = {f: int(l) for l, f in bloklar if f.endswith('.jsonl')}
        self.assertGreater(limitler.get('chatgrow_discourse_pardus_yeni.jsonl', 0),
                           limitler.get('chatgrow_discourse_pardus.jsonl', 999),
                           'yeni pencere en az eskisi kadar derin olmali')


class TestDataFilesUntouched(unittest.TestCase):
    """Veri dosyalarina bu degisiklik dokunmamali.

    Not: sabit mutlak yol taramasi icin ayri bir koruma testi var
    (tests/test_no_hardcoded_paths.py) - repoyu tarar, buraya
    tekrarlanmaz. Tekrar edilirse taranan test dosyasinin kendi
    dokumunu kovalamak kaliplara donusur.
    """

    HARFI = ('intents.json', 'knowledge_map.jsonl', 'corpus.jsonl',
             'corpus_ids.jsonl', 'chatgrow_discourse_pardus.jsonl')

    def test_chatgrow_py_never_writes_data_files_directly(self):
        """chatgrow.py yalnizca --out ile yazmali (veri dosyasi elle degil)."""
        with io.open(os.path.join(BASE, 'chatgrow.py'), encoding='utf-8') as f:
            ic = f.read()
        for ad in self.HARFI:
            self.assertNotIn("'" + ad + "'", ic.replace(
                'chatgrow_discourse_pardus', 'X'),
                'chatgrow.py %s adini gomulu yaziyor' % ad)

    def test_writes_only_through_out_argument(self):
        """Cift yazimi tek noktada: --out dosyasi."""
        with io.open(os.path.join(BASE, 'chatgrow.py'), encoding='utf-8') as f:
            ic = f.read()
        yaz = [ln.strip() for ln in ic.splitlines()
               if "io.open(args.out" in ln or "open(args.out" in ln]
        self.assertEqual(len(yaz), 1,
                         'cift yazimi tek noktada olmali, bulunan: %s' % yaz)


if __name__ == '__main__':
    unittest.main()
