# -*- coding: utf-8 -*-
"""chatgrow Discourse sayfalamasi: /latest.json tek sayfada 30 konu doner.

Yakalandi: build_discourse_chats() listing'i sayfalamiyordu, 'page'
parametresi hic gecmiyordu. Kalici olarak sayfa 0 okunuyordu:
  --limit  60 -> 30 cift
  --limit 200 -> 30 cift
18 commit boyunca hep ayni 30 cift, +0 yeni.

Duzeltme: discourse_topics() sayfa sayfa ilerliyor.
"""
import json
import os
import sys
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import chatgrow


class _FakeDiscourse:
    """Discourse'un sayfa davranisini taklit eder: sayfa basina 30 konu."""

    def __init__(self, toplam=500, per_page=30):
        self.toplam = toplam
        self.per_page = per_page
        self.istekler = []

    def _t(self, i):
        return {
            'id': i,
            'title': 'konu %d' % i,
            'posts_count': 3,
            'post_stream': {'posts': [
                {'post_number': 1, 'username': 'u1',
                 'cooked': '<p>%d numarali soru burada biraz uzun bir soru</p>' % i},
                {'post_number': 2, 'username': 'u2',
                 'cooked': '<p>bu da cevap icin yeterince uzun bir metin parcasidir</p>'},
            ]},
        }

    def get(self, base, path, params=None, retries=3):
        # /t/<id>.json -> konunun mesajlari (fetch_topic_posts bekliyor)
        if path.startswith('/t/'):
            tid = int(path.split('/t/')[1].split('.json')[0])
            t = self._t(tid)
            return {'post_stream': {'posts': t['post_stream']['posts']}}
        params = params or {}
        self.istekler.append(params.get('page', 0))
        sayfa = int(params.get('page', 0))
        bas = sayfa * self.per_page
        if bas >= self.toplam:
            return {'topic_list': {'topics': [], 'per_page': self.per_page,
                                   'more_topics_url': None}}
        son = min(bas + self.per_page, self.toplam)
        return {'topic_list': {
            'topics': [self._t(i) for i in range(bas, son)],
            'per_page': self.per_page,
            'more_topics_url': ('/latest?order=posts&page=%d' % (sayfa + 1)
                                if son < self.toplam else None),
        }}


class TestDiscourseTopicPagination(unittest.TestCase):

    def setUp(self):
        self.gercek = chatgrow.discourse_get
        self.uyku = chatgrow.time.sleep
        chatgrow.time.sleep = lambda *a, **k: None   # testi hizlandir

    def tearDown(self):
        chatgrow.discourse_get = self.gercek
        chatgrow.time.sleep = self.uyku

    def test_single_page_caps_at_thirty(self):
        """Sayfalama OLMADIGI halde de davranis degismemeli (tek sayfa)."""
        sahte = _FakeDiscourse(toplam=30)
        chatgrow.discourse_get = sahte.get
        tp = chatgrow.discourse_topics('x', None, 400)
        self.assertEqual(len(tp), 30,
                         'toplam 30 konu varsa fazlasi uretilmemeli')

    def test_pagination_crosses_page_boundary(self):
        """120 konu istenince 4 sayfa taranmali (30+30+30+30)."""
        sahte = _FakeDiscourse(toplam=500)
        chatgrow.discourse_get = sahte.get
        tp = chatgrow.discourse_topics('x', None, 120)
        self.assertEqual(len(tp), 120)
        self.assertEqual(sorted(set(sahte.istekler)), [0, 1, 2, 3],
                         'sayfa parametresi gecilmeli')

    def test_page_param_actually_sent(self):
        """Eski hatanin kendisi: 'page' hic gecmiyordu."""
        sahte = _FakeDiscourse(toplam=500)
        chatgrow.discourse_get = sahte.get
        chatgrow.discourse_topics('x', None, 90)
        self.assertIn(1, sahte.istekler, 'sayfa 1 hic istenmedi')
        self.assertIn(2, sahte.istekler, 'sayfa 2 hic istenmedi')

    def test_stops_at_end_of_site(self):
        """Sitenin sonu: bos sayfa gelince durmali, sonsuz dongu olmamali."""
        sahte = _FakeDiscourse(toplam=45)
        chatgrow.discourse_get = sahte.get
        tp = chatgrow.discourse_topics('x', None, 400)
        self.assertEqual(len(tp), 45)

    def test_build_chats_beyond_thirty(self):
        """Asil olcum: cift sayisi 30'u net sekilde asmali.

        30 eski tavan; sayfalama calisiyorsa 120 istenince ~110-120
        cift gelmeli. Tam sayiya baglanmiyoruz: temizlik filtreleri
        (kirpinti/yabanci yazim) bir kaci eleyebilir, bu normal.
        """
        sahte = _FakeDiscourse(toplam=500)
        chatgrow.discourse_get = sahte.get
        c = chatgrow.build_discourse_chats('x', None, 120)
        self.assertGreater(len(c), 100,
                           'cift sayisi %d: 30\'luk tavan duruyor, '
                           'sayfalama ise yaramamis' % len(c))

    def test_no_duplicate_topics_across_pages(self):
        sahte = _FakeDiscourse(toplam=500)
        chatgrow.discourse_get = sahte.get
        c = chatgrow.build_discourse_chats('x', None, 200)
        tid = [x['topic_id'] for x in c]
        self.assertEqual(len(tid), len(set(tid)), 'ayni konu iki kez alindi')

    def test_respects_limit(self):
        sahte = _FakeDiscourse(toplam=500)
        chatgrow.discourse_get = sahte.get
        c = chatgrow.build_discourse_chats('x', None, 45)
        self.assertLessEqual(len(c), 45)

    def test_max_pages_guard(self):
        """Bir site sonsuz sayfa donerse sonsuz donguye girmemeli."""
        sahte = _FakeDiscourse(toplam=100000)
        chatgrow.discourse_get = sahte.get
        tp = chatgrow.discourse_topics('x', None, 100000, max_pages=5)
        self.assertEqual(len(tp), 150, '5 sayfa x 30 = 150 olmali')


if __name__ == '__main__':
    unittest.main()
