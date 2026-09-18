"""Nextgen AI - BPE tokenizer test paketi (CI icin).

Turkce subword tokenizer (bpe.py) davranisini dogrular: egitim
(artikimli BPE), Turkce buyuk-kucuk harf, rond-trip encode/decode,
ozel tokenlar ve kayit/yukleme.
    python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest

# Proje koku import yoluna eklenir (tests/ dizini altindayiz)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import bpe  # noqa: E402
from bpe import (  # noqa: E402
    BPETokenizer,
    clean_text,
    train_bpe,
    turkish_lower,
    PAD,
    BOS,
    SEP,
    EOS,
    SPECIALS,
)

SAMPLE = 'merhaba nasılsın bugün hava çok güzel istanbuldan geliyorum'


class TestTurkishNormalization(unittest.TestCase):
    """Turkce imla: İ/I ve karakter temizligi."""

    def test_turkish_lower_keeps_dotless(self):
        self.assertEqual(turkish_lower('İSTANBUL'), 'istanbul')
        self.assertEqual(turkish_lower('MUSTAFA KEMAL'), 'mustafa kemal')

    def test_turkish_lower_preserves_ascii_i(self):
        # 'i' ve 'I' ayrimi korunur; 'I' -> 'i' degil 'ı'
        self.assertEqual(turkish_lower('GÜZEL IŞIK'), 'güzel ışık')

    def test_clean_text_keeps_turkish_letters(self):
        c = clean_text('Şırnak öğrenci Çağrı İğde Üsküp')
        self.assertEqual(c, 'şırnak öğrenci çağrı iğde üsküp')

    def test_clean_text_collapses_spaces(self):
        self.assertEqual(clean_text('iki   bosluk      bitti'), 'iki bosluk bitti')

    def test_clean_text_drops_unknown(self):
        self.assertEqual(clean_text('merhaba 🚀 araba'), 'merhaba araba')


class TestBPEBasics(unittest.TestCase):
    """Kucuk korpus ile temel BPE davranisi."""

    CORPUS = [
        'merhaba nasılsın iyiyim sen nasılsın',
        'bugün hava çok güzel yürüyecek misin',
        'nerelisin istanbuldan geliyorum ve orada yaşıyorum',
        'kitap okumak insanın ufkunu açar geçmişi anlamak için',
        'yapay zeka geleceğin teknolojisidir öğrenmek önemli',
    ]

    def test_vocab_size_is_upper_bound(self):
        tok = train_bpe(self.CORPUS, vocab_size=300, min_freq=1,
                        min_word_freq=1)
        # Kucuk korpus ciftleri tuketirse vocab_size'a ulasamaz
        self.assertGreater(len(tok), 64)          # 4 special + temel karakterler
        self.assertLessEqual(len(tok), 300)
        self.assertEqual(tok.vocab_size, len(tok))

    def test_specials_first(self):
        tok = train_bpe(self.CORPUS, vocab_size=300, min_freq=1,
                        min_word_freq=1)
        self.assertEqual(tok.pieces[PAD], '<pad>')
        self.assertEqual(tok.pieces[BOS], '<bos>')
        self.assertEqual(tok.pieces[SEP], '<sep>')
        self.assertEqual(tok.pieces[EOS], '<eos>')
        self.assertEqual(tok.specials, list(SPECIALS))

    def test_merges_build_words(self):
        corpus = (['merhaba nasılsın iyiyim sen nasılsın'] * 20
                  + ['bugün hava çok güzel yürüyecek misin'] * 10)
        tok = train_bpe(corpus, vocab_size=200, min_freq=2,
                        min_word_freq=1)
        # 'merhaba' cok sik gectigi icin tek token olmali
        ids = tok.encode('merhaba')
        self.assertEqual(len(ids), 1)
        self.assertEqual(tok.decode(ids), 'merhaba')

    def test_round_trip(self):
        tok = train_bpe(self.CORPUS, vocab_size=300, min_freq=1,
                        min_word_freq=1)
        s = clean_text(SAMPLE)
        self.assertEqual(tok.decode(tok.encode(s)), s)

    def test_round_trip_with_punctuation(self):
        tok = train_bpe(self.CORPUS, vocab_size=300, min_freq=1,
                        min_word_freq=1)
        s = 'nasılsın, iyiyim! sen nasılsın?'
        self.assertEqual(tok.decode(tok.encode(s)), 'nasılsın, iyiyim! sen nasılsın?')

    def test_deterministic_merges(self):
        t1 = train_bpe(self.CORPUS, vocab_size=250, min_freq=1, min_word_freq=1)
        t2 = train_bpe(self.CORPUS, vocab_size=250, min_freq=1, min_word_freq=1)
        self.assertEqual(t1.merges, t2.merges)

    def test_space_is_token(self):
        tok = train_bpe(['iki kelime', 'bir kelime'], vocab_size=200,
                        min_freq=1, min_word_freq=1)
        ids = tok.encode('iki kelime')
        self.assertIn(' ', [tok.pieces[i] for i in ids])

    def test_empty_encode_pads(self):
        tok = train_bpe(self.CORPUS, vocab_size=100, min_freq=1,
                        min_word_freq=1)
        self.assertTrue(tok.encode(''))


class TestTokenizerPersistence(unittest.TestCase):
    """save/load rond-trip."""

    def setUp(self):
        self.tok = train_bpe(TestBPEBasics.CORPUS, vocab_size=400,
                             min_freq=1, min_word_freq=1)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'bpe.json')
            self.tok.save(path)
            tok2 = BPETokenizer.load(path)
            self.assertEqual(tok2.merges, self.tok.merges)
            self.assertEqual(tok2.chars, self.tok.chars)
            s = clean_text(SAMPLE)
            self.assertEqual(tok2.encode(s), self.tok.encode(s))
            self.assertEqual(tok2.decode(tok2.encode(s)),
                             self.tok.decode(self.tok.encode(s)))

    def test_load_trainer_output(self):
        # Egitim CLI'sinin urettigi gercek dosya varsa kontrol edilir
        real = os.path.join(BASE, 'tokenizer', 'bpe.json')
        if not os.path.exists(real):
            self.skipTest('tokenizer/bpe.json henuz yok (train_tokenizer.py)')
        tok = BPETokenizer.load(real)
        self.assertEqual(len(tok), 16000)
        self.assertEqual(tok.decode(tok.encode(SAMPLE)), clean_text(SAMPLE))


class TestLLMBridge(unittest.TestCase):
    """llm.py ile BPE entegrasyonu (geriye donuk char modu da)."""

    CORPUS = ['merhaba nasilsin iyiyim sen nasilsin',
              'bugun hava cok guzel yuruyecek misin',
              'nerelisin istanbuldan geliyorum']

    def setUp(self):
        import numpy as np
        self.np = np
        from llm import LLM, encode_llm, load_llm, save_llm, load_tokenizer
        from bpe import train_bpe
        self.LLM, self.encode_llm = LLM, encode_llm
        self.load_llm, self.save_llm = load_llm, save_llm
        self.load_tokenizer = load_tokenizer
        self.tok = train_bpe(self.CORPUS, vocab_size=200, min_freq=1,
                             min_word_freq=1)

    def _model(self, tokenizer=None):
        return self.LLM(d_model=32, num_blocks=2, num_heads=2,
                        max_ctx_len=40, max_seq_len=80, seed=7,
                        tokenizer=tokenizer)

    def test_bpe_vocab_size(self):
        m = self._model(self.tok)
        self.assertEqual(m.V, len(self.tok))
        self.assertIs(m.tokenizer, self.tok)

    def test_bpe_forward_shape(self):
        m = self._model(self.tok)
        import numpy as np
        ids = self.tok.encode(self.CORPUS[0])
        logits = m.forward(np.array([ids], np.int64))
        self.assertEqual(logits.shape, (1, len(ids), len(self.tok)))

    def test_bpe_encode_llm_layout(self):
        m = self._model(self.tok)
        seq, smask = self.encode_llm(m, 'nasilsin', 'iyiyim')
        self.assertEqual(seq.ndim, 1)
        self.assertEqual(len(seq), m.max_seq_len)
        self.assertEqual(seq.shape, smask.shape)
        # BOS ile baslar, ilk SEP'ten sonra mask altinda kalir
        self.assertEqual(seq[0], 1)
        sep0 = int(self.np.where(seq == 2)[0][0])
        self.assertEqual(seq[sep0], 2)
        self.assertGreater(float(smask.sum()), 0.0)

    def test_bpe_model_round_trip(self):
        m = self._model(self.tok)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'llm.json')
            self.save_llm(m, path)
            m2 = self.load_llm(path)
            self.assertIsNotNone(m2)
            self.assertIsNotNone(m2.tokenizer)
            self.assertEqual(m2.tokenizer.merges, self.tok.merges)
            ids1 = m2.tokenizer.encode('nasilsin bugun')
            ids2 = self.tok.encode('nasilsin bugun')
            self.assertEqual(ids1, ids2)

    def test_char_model_still_loads(self):
        # Geriye donuk char-mod: tokenizer olmadan model JSON'dan doner
        m = self._model(tokenizer=None)
        d = m.to_dict()
        self.assertEqual(d['tok_mode'], 'char')
        m2 = self.LLM(['<PAD>', '<BOS>', '<SEP>', '<EOS>']).from_dict(d)
        self.assertIsNone(m2.tokenizer)
        self.assertEqual(m2.V, m.V)

    def test_real_tokenizer_loads(self):
        tok = self.load_tokenizer(os.path.join(BASE, 'tokenizer', 'bpe.json'))
        if tok is None:
            from llm import TOKENIZER_PATH
            tok = self.load_tokenizer()
        if tok is None:
            self.skipTest('tokenizer/bpe.json henuz yok')
        self.assertGreaterEqual(len(tok), 1000)
        self.assertEqual(tok.pieces[1], '<bos>')

    def test_bpe_sample_returns_str(self):
        m = self._model(self.tok)
        out = m.sample('nasilsin bugun hava', max_len=12, temperature=0.9)
        self.assertIsInstance(out, str)


if __name__ == '__main__':
    unittest.main()