"""chatgrow saf fonksiyon birim testleri (network yok)."""

import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatgrow import (clean_line, ascii_normalize, DEFAULT_SUBREDDITS,
                      build_chats, build_discourse_chats, discourse_get,
                      html_to_text)
from train_llm import load_chatgrow_pairs


class TestCleanLine(unittest.TestCase):

    def test_accepts_normal_chat(self):
        text = "Bugun hava cok guzel degil mi sizce ne yapiyorsunuz"
        self.assertEqual(clean_line(text), text)

    def test_rejects_url(self):
        self.assertIsNone(clean_line(
            "Detaylari buradan okuyabilirsiniz https://example.com/abc"))

    def test_rejects_email_and_phone(self):
        self.assertIsNone(clean_line(
            "Bana eposta@site.com yerine 0532 111 22 33 yazabilirsiniz"))

    def test_rejects_username_mention(self):
        got = clean_line("u/kullanici ile r/Turkey'de konusuyoruz cok iyi")
        self.assertNotIn('u/', got)
        self.assertNotIn('r/', got)
        self.assertIsNotNone(got)

    def test_rejects_harmful(self):
        self.assertIsNone(clean_line(
            "Bir seri katil vakasi hakkinda bugun konusalm mi ne dersin"))

    def test_rejects_too_short(self):
        self.assertIsNone(clean_line("evet olur"))

    def test_rejects_foreign_scripts(self):
        self.assertIsNone(clean_line("Привет мир как дела хорошо сегодня"))

    def test_rejects_repeated_single_char(self):
        self.assertIsNone(clean_line("aaa aaa aaa aaa aaa aaa aaa"))

    def test_strips_markdown_noise(self):
        got = clean_line("**bugun** cok yoruldum *gercekten* yani")
        self.assertIsNotNone(got)
        self.assertNotIn('*', got)


class TestBuildChats(unittest.TestCase):

    def test_deduplicates_queries(self):
        subs = {'r/test': [
            {'title': 'Ilk sorum gunluk konusma icin mi yoksa', 'over_18': False,
             'num_comments': 20, 'score': 50, 'permalink': '/r/test/comments/1'},
            {'title': 'Ilk sorum gunluk konusma icin mi yoksa', 'over_18': False,
             'num_comments': 20, 'score': 50, 'permalink': '/r/test/comments/2'},
        ]}

        def fake_posts(token, sub, section, limit):
            return subs[sub]

        def fake_comments(token, permalink):
            return ["Herkes icin iyi bir baslangic olabilir bence",
                    "Kesinlikle haklisin bunu dusunmemisim"]

        orig_posts, orig_comments = None, None
        import chatgrow
        orig_posts = chatgrow.fetch_posts
        orig_comments = chatgrow.fetch_comments
        try:
            chatgrow.fetch_posts = fake_posts
            chatgrow.fetch_comments = fake_comments
            chats = build_chats('tok', 'r/test', 'top', 10)
        finally:
            chatgrow.fetch_posts = orig_posts
            chatgrow.fetch_comments = orig_comments

        self.assertEqual(len(chats), 1)
        self.assertIn('answer', chats[0])
        self.assertEqual(chats[0]['source'], 'reddit')

    def test_skips_weak_threads(self):
        import chatgrow

        def fake_posts(token, sub, section, limit):
            return [
                {'title': 'Cok dusuk yorumlu baslik buraya atilmaz', 'over_18': False,
                 'num_comments': 1, 'score': 50, 'permalink': '/x'},
                {'title': 'NSFW bir paylasim burada elenmeli', 'over_18': True,
                 'num_comments': 50, 'score': 99, 'permalink': '/y'},
            ]

        orig = chatgrow.fetch_posts
        try:
            chatgrow.fetch_posts = fake_posts
            chats = build_chats('tok', 'r/test', 'top', 10)
        finally:
            chatgrow.fetch_posts = orig

        self.assertEqual(chats, [])


class TestLoadChatgrowPairs(unittest.TestCase):

    def _write(self, rows):
        fd, path = tempfile.mkstemp(suffix='.jsonl')
        os.close(fd)
        with io.open(path, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        self.addCleanup(os.unlink, path)
        return path

    def test_expands_answer_list(self):
        path = self._write([
            {'query': 'aksam ne pisirsem bilmiyorum',
             'answer': ['Kofte ve cacik iyi olur', 'Makarna pratik bir cozum']},
        ])
        pairs = load_chatgrow_pairs(path)
        self.assertEqual(len(pairs), 2)
        qs = {p[0] for p in pairs}
        self.assertEqual(qs, {'aksam ne pisirsem bilmiyorum'})

    def test_single_answer_string(self):
        path = self._write([{'query': 'cok yoruldum', 'answer': 'Haklisin dinlen'}])
        pairs = load_chatgrow_pairs(path)
        self.assertEqual(len(pairs), 1)

    def test_skips_short_and_duplicate_answers(self):
        path = self._write([
            {'query': 'ne yapsam',
             'answer': ['Ayni cok kisa', 'Ayni cok kisa']},
        ])
        pairs = load_chatgrow_pairs(path)
        self.assertEqual(len(pairs), 1)

    def test_ignores_bad_lines(self):
        fd, path = tempfile.mkstemp(suffix='.jsonl')
        os.close(fd)
        with io.open(path, 'w', encoding='utf-8') as f:
            f.write('{{{ bozuk json\n')
            f.write('{"query":"iyi soru","answer":"Iyi bir yanit burada yazar"}\n')
        self.addCleanup(os.unlink, path)
        self.assertEqual(len(load_chatgrow_pairs(path)), 1)

    def test_missing_file_is_empty(self):
        self.assertEqual(load_chatgrow_pairs('yok_boyle.jsonl'), [])

    def test_normalizes_like_training(self):
        path = self._write([
            {'query': 'Çok Sıcak Bugün Hava',
             'answer': ['Serin bir Kafe kurtarır, ısı düşer']},
        ])
        pairs = load_chatgrow_pairs(path)
        self.assertEqual(pairs[0][0], 'cok sicak bugun hava')
        self.assertNotIn('\u00e7', pairs[0][1].lower())


class TestModuleHygiene(unittest.TestCase):

    def test_default_subreddits(self):
        self.assertIn('r/Turkey', DEFAULT_SUBREDDITS)

    def test_clean_key_normalizer(self):
        self.assertEqual(ascii_normalize('Bugün Çok'.lower()), 'bugun cok')


class TestHtmlToText(unittest.TestCase):

    def test_strips_tags_and_entities(self):
        self.assertEqual(html_to_text('<p>Merhaba &amp; hoş geldin</p>'),
                         'Merhaba & hoş geldin')

    def test_removes_script_blocks(self):
        html = '<script>var x = 1;</script><p>Metin</p>'
        self.assertNotIn('script', html_to_text(html))

    def test_empty_input(self):
        self.assertEqual(html_to_text(None), '')
        self.assertEqual(html_to_text(''), '')

    def test_strips_shortcode_emojis(self):
        self.assertEqual(html_to_text('<p>:earthafrica: Naber?</p>'), 'Naber?')


class TestBuildDiscourseChats(unittest.TestCase):
    """discourse_get mock'lanir; network yoktur."""

    TOPICS = {
        'topic_list': {'topics': [
            {'id': 11, 'title': 'Sıcak havada ne yapmalı?', 'posts_count': 3},
            {'id': 12, 'title': 'Tek mesajlı konu', 'posts_count': 1},
        ]},
    }
    POSTS_11 = {'post_stream': {'posts': [
        {'post_number': 1, 'username': 'Ali',
         'cooked': '<p>Hava çok sıcak, elektrikler kesilebilir.</p>'},
        {'post_number': 2, 'username': 'System',
         'cooked': '<p>Otomatik kayıt, önemsiz.</p>'},
        {'post_number': 3, 'username': 'Zeynep',
         'cooked': '<p>Pencereleri kapatıp panjur kullanın.</p>'},
        {'post_number': 4, 'username': 'Mehmet',
         'cooked': '<p>Ayrıca soğuk duş işe yarıyor.</p>'},
    ]}}
    POSTS_12 = {'post_stream': {'posts': [
        {'post_number': 1, 'username': 'Veli',
         'cooked': '<p>Yalnız tek mesaj var.</p>'},
    ]}}

    def _fake_get(self, base, path, params=None, retries=3):
        if path == '/c/genel-sohbet/l/latest.json':
            return self.TOPICS
        if path == '/t/11.json':
            return self.POSTS_11
        if path == '/t/12.json':
            return self.POSTS_12
        return None

    def test_builds_pairs_and_skips_system(self):
        chatgrow_mod = sys.modules['chatgrow']
        orig = chatgrow_mod.discourse_get
        chatgrow_mod.discourse_get = self._fake_get
        try:
            chats = build_discourse_chats('https://forum.ornek.org', 'genel-sohbet', 5)
        finally:
            chatgrow_mod.discourse_get = orig
        self.assertEqual(len(chats), 1)
        c = chats[0]
        self.assertEqual(c['source'], 'discourse')
        self.assertEqual(c['topic_id'], 11)
        # System otomatik iletisi (2) elenmeli; kalan iki yanit duzeltilmeli
        self.assertEqual(len(c['answer']), 2)
        self.assertIn('panjur', ascii_normalize(c['answer'][0].lower()))
        self.assertIn('soguk dus', ascii_normalize(c['answer'][1].lower()))
        self.assertNotIn('otomatik', c['query'])

    def test_empty_listing(self):
        chatgrow_mod = sys.modules['chatgrow']
        orig = chatgrow_mod.discourse_get
        chatgrow_mod.discourse_get = lambda *a, **k: None
        try:
            self.assertEqual(build_discourse_chats('https://x.org', '', 5), [])
        finally:
            chatgrow_mod.discourse_get = orig

    def test_dry_run_requires_forum(self):
        from chatgrow import main as gm
        saved = sys.argv
        sys.argv = ['chatgrow.py', '--source', 'discourse', '--dry-run']
        try:
            self.assertEqual(gm(), 1)
        finally:
            sys.argv = saved


if __name__ == '__main__':
    unittest.main()