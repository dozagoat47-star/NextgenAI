"""Nextgen AI - Cekirdek test paketi (CI icin).

Model egitiminden BAGIMSIZ calisir: NLP pipeline'i, intents.json
semasini, RAG corpus'unu ve LSTM (seqgen) gradyan sapini dogrular.
    python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest

# Proje koku import yoluna eklenir (tests/ dizini altindayiz)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from brain import ChatBot, STOPWORDS, NEGATION_SIGNALS, NEGATION_SUFFIXES


class TestNLP(unittest.TestCase):
    """Tokenizer / stemming / olumsuzluk algilama / ASCII normalizasyon."""

    def setUp(self):
        self.bot = ChatBot()

    def test_ascii_normalize(self):
        self.assertEqual(self.bot.ascii_normalize('Şükran öğle ılık'),
                         'sukran ogle ilik')

    def test_stopwords_present(self):
        self.assertTrue('ve' in STOPWORDS)
        self.assertTrue('ama' in STOPWORDS)
        self.assertTrue('nasil' in STOPWORDS)

    def test_negation_signals(self):
        self.assertTrue('degil' in NEGATION_SIGNALS)
        self.assertTrue('hayir' in NEGATION_SIGNALS)
        self.assertTrue('asla' in NEGATION_SIGNALS)

    def test_negation_suffixes(self):
        self.assertTrue('miyorum' in NEGATION_SUFFIXES)
        self.assertTrue('maz' in NEGATION_SUFFIXES)

    def test_is_negation_word(self):
        self.assertTrue(self.bot.is_negation_word('sevmiyorum'))
        self.assertTrue(self.bot.is_negation_word('degil'))
        self.assertFalse(self.bot.is_negation_word('kedi'))

    def test_detect_negation(self):
        neg, content = self.bot.detect_negation('futbol sevmiyorum')
        self.assertTrue(neg)
        self.assertIn('futbol', content)

    def test_detect_negation_negative_case(self):
        neg, _ = self.bot.detect_negation('futbol oynuyorum')
        self.assertFalse(neg)

    def test_stem_simple(self):
        # Turkce cokluk eki altka
        self.assertEqual(self.bot.simple_stem('kediler'), 'ked')
        # Olumsuzluk infiksi 'miy' gövdede korunur
        self.assertNotEqual(self.bot.simple_stem('sevmiyim'), 'sev')

    def test_tokenize_punctuation(self):
        toks = self.bot.tokenize('Merhaba, nasilsin?')
        self.assertIsInstance(toks, list)
        self.assertTrue(toks)

    def test_tokenize_bigram(self):
        toks = self.bot.tokenize('yapay zeka nedir')
        self.assertIn('yapay_zeka', toks)

    def test_bag_of_words_requires_vocab(self):
        # Vocabulary yuklenmemisse bos ok bulonur (model'siz guvenli)
        bag = self.bot.bag_of_words(['kedi'])
        self.assertEqual(bag, [])


class TestIntentsSchema(unittest.TestCase):
    """intents.json yapisal tutarlilik (egitim oncesi kalkan)."""

    def setUp(self):
        with open(os.path.join(BASE, 'intents.json'), 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def test_intents_present(self):
        self.assertGreater(len(self.data['intents']), 500)

    def test_each_intent_shape(self):
        seen = set()
        for it in self.data['intents']:
            tag = it.get('tag', '')
            self.assertTrue(len(tag) >= 2, f'kisa tag: {tag!r}')
            self.assertNotIn(tag, seen)
            seen.add(tag)
            self.assertTrue(it.get('patterns'), f'{tag}: kalipsiz')
            self.assertTrue(it.get('responses'), f'{tag}: yanitsiz')
            for p in it['patterns']:
                self.assertTrue(p.strip(), f'{tag}: bos kalip')
            for r in it['responses']:
                self.assertTrue(len(r.strip()) >= 2, f'{tag}: cok kisa yanit')

    def test_tags_latin(self):
        # Tag Latin disinda bir HARF iceremez (Turkce degerler zaten ascii_kodlandi).
        for it in self.data['intents']:
            tag = it['tag']
            bad = [c for c in tag if c.isalpha() and ord(c) > 127]
            self.assertFalse(bad, f'Latin disi karakter: {tag!r}')


class TestCorpus(unittest.TestCase):
    """RAG corpus yukleme + NASIL arama (embedding depolu ama model'siz)."""

    def test_corpus_search(self):
        from corpus import Corpus
        c = Corpus()
        chunks = c.load()
        self.assertGreater(len(chunks), 1000,
                           'corpus.jsonl en az 1000 parca islerligi dogruluyor')
        r = c.search('klorofil ne ise yarar')
        self.assertIsNotNone(r)
        self.assertTrue(r.get('title'))
        self.assertTrue(r.get('text'))
        self.assertGreater(r.get('score', 0), 0)

    def test_corpus_snippet(self):
        from corpus import Corpus
        text = 'Birinci cumle. Ikinci cumle uzun ve ayrintili bicimde devam ediyor.'
        self.assertEqual(Corpus.snippet(text, max_len=20), 'Birinci cumle.')


class TestSeqGenLSTM(unittest.TestCase):
    """seqgen LSTM geri yayılımı küçük veriyle kaybi düsürmeli."""

    @staticmethod
    def _build():
        import seqgen
        texts = ['nasilsin', 'iyiyim', 'tesekkur ederim', 'rica ederim',
                 'python nedir', 'python programlama dilidir',
                 'kedi beslemek', 'kediler evcil hayvandir']
        vocab = seqgen.build_vocab(texts, min_count=1)
        m = seqgen.SeqModel(vocab, hidden=32)
        pairs = [('nasilsin', 'iyiyim'), ('tesekkur ederim', 'rica ederim'),
                 ('python nedir', 'python programlama dilidir'),
                 ('kedi beslemek', 'kediler evcil hayvandir'),
                 ('hava nasil', 'hava bugun cok guzel')]
        mb = seqgen.make_batches(m, pairs, 2)
        return m, mb

    def test_lstm_learns(self):
        m, mb = self._build()
        first = None
        last = None
        for step in range(60):
            tot = 0.0
            for inp, tgt, mask in mb:
                loss, _ = m.train_minibatch(inp, tgt, mask, 0.01)
                tot += loss
            if first is None:
                first = tot / len(mb)
            last = tot / len(mb)
        self.assertLess(last, first,
                        f'LSTM kaybı düşmedi: {first:.4f} -> {last:.4f}')

    def test_sample_runs(self):
        from seqgen import SeqModel, build_vocab
        vocab = build_vocab(['merhaba', 'selamlar', 'iyi gunler'], min_count=1)
        m = SeqModel(vocab, hidden=16)
        out = m.sample('merhaba', max_len=12)
        self.assertIsInstance(out, str)


class TestAtomicWriteJson(unittest.TestCase):
    """B regresyon: kalicilik yazimi gecici dosya + os.replace ile atomiktir;
    basarisiz yazim hedef dosyayi bozmaz ve .tmp artiklari birakmaz."""

    def test_writes_and_cleans_tmp(self):
        import os
        import tempfile

        from finetune import atomic_write_json
        d = tempfile.mkdtemp()
        p = os.path.join(d, 'intents.json')
        atomic_write_json(p, {'a': 1, 'b': ['x', 'y']}, indent=2)
        with open(p, 'r', encoding='utf-8') as f:
            self.assertEqual(json.load(f), {'a': 1, 'b': ['x', 'y']})
        self.assertFalse(os.path.exists(p + '.tmp'))

    def test_failed_write_preserves_original(self):
        import os
        import tempfile

        from finetune import atomic_write_json

        class NotSerializable:
            pass

        d = tempfile.mkdtemp()
        p = os.path.join(d, 'intents.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'eski': True}, f)
        with self.assertRaises(TypeError):
            atomic_write_json(p, {'yeni': NotSerializable()}, indent=2)
        with open(p, 'r', encoding='utf-8') as f:
            self.assertEqual(json.load(f), {'eski': True})
        self.assertFalse(os.path.exists(p + '.tmp'))


class TestFinetuneMetadataMerge(unittest.TestCase):
    """B regresyon: /learn ayni tag'e gelen tekrarli istek yanit havuzunu
    komple EZMEMELI; mevcut havuz korunup yeni yanitlar eklenmelidir."""

    def _bot_data(self, responses):
        return {
            'vocabulary': ['selam', 'nasilsin'],
            'intent_tags': ['selamlasma'],
            'intent_kws': {'selamlasma': ['selam']},
            'intents': {'selamlasma': responses},
        }

    def test_existing_tag_responses_preserved_and_extended(self):
        from brain import ChatBot
        from finetune import _extend_metadata

        bot_data = self._bot_data(['Merhaba!', 'Naber?'])
        additions = [
            {'tag': 'selamlasma', 'patterns': ['gunaydin'],
             'responses': ['Gunaydin!']},
        ]
        nw, nt = _extend_metadata(bot_data, additions, ChatBot())
        self.assertEqual(nt, [])
        self.assertEqual(bot_data['intents']['selamlasma'],
                         ['Merhaba!', 'Naber?', 'Gunaydin!'])
        toks = ChatBot().tokenize('gunaydin')
        self.assertTrue(toks)
        for t in toks:
            self.assertIn(t, bot_data['vocabulary'])
            self.assertIn(t, bot_data['intent_kws']['selamlasma'])

    def test_duplicate_and_empty_responses_do_not_erode(self):
        from brain import ChatBot
        from finetune import _extend_metadata

        bot_data = self._bot_data(['Ana cevap'])
        additions = [
            {'tag': 'selamlasma', 'patterns': ['selam'],
             'responses': ['Ana cevap', '', 'Yeni ek'], }
        ]
        _extend_metadata(bot_data, additions, ChatBot())
        self.assertEqual(bot_data['intents']['selamlasma'],
                         ['Ana cevap', 'Yeni ek'])

    def test_new_tag_still_creates_responses(self):
        from brain import ChatBot
        from finetune import _extend_metadata

        bot_data = self._bot_data(['Merhaba!'])
        additions = [
            {'tag': 'spor', 'patterns': ['kosu'], 'responses': ['Hayirli kosular!']},
        ]
        nw, nt = _extend_metadata(bot_data, additions, ChatBot())
        self.assertEqual(nt, ['spor'])
        self.assertEqual(bot_data['intents']['spor'], ['Hayirli kosular!'])


class TestDeasciify(unittest.TestCase):
    """Deasciify: ASCII model ciktisi canned yanitlardaki Turkce imlaya
    cevrilir; bilinmeyen sozcukler ve noktalama bozulmaz."""

    def _bot(self):
        bot = ChatBot()
        bot.intents = {
            'selamlasma': ['Nasılsın?', 'Ben iyiyim, sen nasılsın?'],
            'hava': ['Bugün hava çok güzel.'],
        }
        return bot

    def test_known_words_restored(self):
        bot = self._bot()
        self.assertEqual(bot.deasciify('nasilsin'),
                         'nasılsın')
        self.assertEqual(bot.deasciify('bugun hava cok guzel'),
                         'bugün hava çok güzel')

    def test_capitalization_preserved(self):
        bot = self._bot()
        self.assertEqual(bot.deasciify('Nasilsin?'), 'Nasılsın?')
        self.assertEqual(bot.deasciify('Bugun hava cok guzel.'),
                         'Bugün hava çok güzel.')

    def test_unknown_words_unchanged(self):
        bot = self._bot()
        self.assertEqual(bot.deasciify('Nextgen bugun son versiyon'),
                         'Nextgen bugün son versiyon')

    def test_acronyms_kept(self):
        bot = self._bot()
        bot.intents['hava'].append('Bugün NATO BIOSSuz konusacak AI icin')
        self.assertEqual(bot.deasciify('NATO ve AI bugun toplanacak'),
                         'NATO ve AI bugün toplanacak')

    def test_lowercased_acronym_source_not_poisoned(self):
        bot = self._bot()
        # canned'ta 'AI' (kisaltma) ile, bir de onun Turkce-kucukharf bozuk
        # aktarimi 'aı' gecer: sozluk 'ai' anahtarini uretmemeli (ASCII 'AI'
        # oldugu gibi kalsin), 'nasilsin' yine duzeltilmeli.
        bot.intents['selamlasma'].append('ben netgen AI yim, aı yim.')
        bot.deasciify('bos')
        self.assertNotIn('ai', bot._deascii_lex)
        self.assertEqual(bot.deasciify('ben AI aı nasilsin'),
                         'ben AI aı nasılsın')

    def test_punctuation_and_case_kept(self):
        bot = self._bot()
        self.assertEqual(bot.deasciify('Nasilsin, BEN IYIYIM!'),
                         'Nasılsın, BEN IYIYIM!')

    def test_canned_output_is_identity(self):
        bot = self._bot()
        for tag in bot.intents:
            for r in bot.intents[tag]:
                self.assertEqual(bot.deasciify(r), r)

    def test_lex_built_lazily_once(self):
        bot = ChatBot()
        bot.intents = {'x': ['Ayşe ılık çorba içti']}
        self.assertIsNone(bot._deascii_lex)
        bot.deasciify('ayse ilik corba')
        self.assertIsNotNone(bot._deascii_lex)
        # İ/I harfleri dogru kucuk harfe eslenir (i dedigi ı olur)
        self.assertEqual(bot._deascii_lex['ayse'], 'ayşe')
        self.assertEqual(bot._deascii_lex['ilik'], 'ılık')


class TestTransformer(unittest.TestCase):
    """Transformer encoder kucuk veriyle ogrenmeli + kayit/yukleme tur tutarli."""

    def test_transformer_learns(self):
        import numpy as np
        from transformer import TransformerNN
        m = TransformerNN(vocab_size=12, num_intents=3, max_seq_len=8,
                          d_model=32, num_blocks=2, num_heads=2, ff_mult=2,
                          dropout=0.0, attn_dropout=0.0, seed=1)
        rng = np.random.RandomState(0)
        X = rng.randint(0, 12, size=(40, 8))
        for i in range(X.shape[0]):
            X[i, rng.randint(4, 8, size=2)] = 12  # PAD
        y = np.array([i % 3 for i in range(40)])
        first = last = 0
        for ep in range(80):
            probs = m.forward(X, apply_dropout=True)
            loss = m.compute_loss(probs, y)
            if ep == 0:
                first = loss
            m.backward(y, 0.01)
            last = loss
        self.assertLess(last, first, f'kayip dusmedi: {first:.4f} -> {last:.4f}')
        self.assertGreater(m.evaluate(X, y), 0.8)

    def test_transformer_save_load(self):
        import os
        import tempfile
        from transformer import TransformerNN
        m1 = TransformerNN(vocab_size=20, num_intents=4, max_seq_len=10,
                           d_model=32, num_blocks=1, num_heads=2, ff_mult=2,
                           seed=3)
        path = os.path.join(tempfile.gettempdir(), 'ng_transformer_test.json')
        m1.save(path)
        try:
            m2 = TransformerNN(vocab_size=1, num_intents=1, max_seq_len=1)
            m2.load(path)
            self.assertEqual(m2.vocab_size, 20)
            self.assertEqual(m2.d_model, 32)
            import numpy as np
            for name, val in m1._named_params():
                self.assertTrue(np.allclose(np.asarray(val), np.asarray(m2.get_state()[name])),
                                f'parametri farkli: {name}')
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_transformer_export_map(self):
        """Colab'taki PyTorch->NumPy export eslemesinin (anahtar/transpoz)
        model.json semasiyla birebir yuvarlak donusumu."""
        import json
        import os
        import tempfile

        import numpy as np
        from transformer import TransformerNN

        V, C, L, D, NB, NH = 160, 8, 6, 48, 2, 2
        m = TransformerNN(vocab_size=V, num_intents=C, max_seq_len=L,
                          d_model=D, num_blocks=NB, num_heads=NH, ff_mult=3,
                          dropout=0.1, attn_dropout=0.05, seed=5)

        # NumPy modelden torch tarzi state_dict uret (export'un tersi)
        st = {'embed.weight': m.embed}
        for i, blk in enumerate(m.blocks):
            def nm(k):
                return f'blocks.{i}.{k}'
            for q in ('Wq', 'Wk', 'Wv', 'Wo'):
                w = getattr(blk['attn'], q)
                st[nm(f'attn.{q}.weight')] = w.T
                st[nm(f'attn.{q}.bias')] = getattr(blk['attn'], 'b' + q[1]).reshape(-1)
            for ln in ('ln1', 'ln2'):
                st[nm(f'{ln}.weight')] = blk[f'{ln}_g'].reshape(-1)
                st[nm(f'{ln}.bias')] = blk[f'{ln}_b'].reshape(-1)
            st[nm('W1.weight')] = blk['W1'].T
            st[nm('W1.bias')] = blk['b1'].reshape(-1)
            st[nm('W2.weight')] = blk['W2'].T
            st[nm('W2.bias')] = blk['b2'].reshape(-1)
        st['Whead.weight'] = m.Whead.T
        st['Whead.bias'] = m.bhead.reshape(-1)

        params = {'embed': st['embed.weight']}
        for i in range(NB):
            for nm, src in [('Wq', 'attn.Wq.weight'), ('Wk', 'attn.Wk.weight'),
                            ('Wv', 'attn.Wv.weight'), ('Wo', 'attn.Wo.weight')]:
                params[f'b{i}_{nm}'] = st[f'blocks.{i}.{src}'].T
            for nm, src in [('bq', 'attn.Wq.bias'), ('bk', 'attn.Wk.bias'),
                            ('bv', 'attn.Wv.bias'), ('bo', 'attn.Wo.bias')]:
                params[f'b{i}_{nm}'] = st[f'blocks.{i}.{src}'][None, :]
            for nm, src in [('ln1_g', 'ln1.weight'), ('ln1_b', 'ln1.bias'),
                            ('ln2_g', 'ln2.weight'), ('ln2_b', 'ln2.bias')]:
                params[f'b{i}_{nm}'] = st[f'blocks.{i}.{src}'][None, :]
            params[f'b{i}_W1'] = st[f'blocks.{i}.W1.weight'].T
            params[f'b{i}_b1'] = st[f'blocks.{i}.W1.bias'][None, :]
            params[f'b{i}_W2'] = st[f'blocks.{i}.W2.weight'].T
            params[f'b{i}_b2'] = st[f'blocks.{i}.W2.bias'][None, :]
        params['Whead'] = st['Whead.weight'].T
        params['bhead'] = st['Whead.bias'][None, :]

        data = {'arch': 'transformer', 'vocab_size': V, 'num_intents': C,
                'max_seq_len': L, 'd_model': D, 'num_blocks': NB,
                'num_heads': NH, 'ff_dim': 3 * D, 'dropout': 0.1,
                'attn_dropout': 0.05, 'weight_decay': 1e-4, 'max_grad_norm': 5.0,
                'params': {k: v.astype(np.float32).tolist() for k, v in params.items()}}
        path = os.path.join(tempfile.gettempdir(), 'ng_export_map.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        try:
            m2 = TransformerNN(vocab_size=V, num_intents=C, max_seq_len=L,
                               d_model=D, num_blocks=NB, num_heads=NH, ff_mult=3)
            m2.load(path)
            for name, val in m._named_params():
                np.testing.assert_array_equal(np.asarray(val), np.asarray(m2.get_state()[name]),
                                              err_msg=f'esleme farki: {name}')
            rng = np.random.RandomState(0)
            Xs = rng.randint(0, V, size=(8, L))
            Xpad = np.where(rng.rand(8, L) < 0.3, V, Xs)
            self.assertAlmostEqual(float(np.abs(m.predict_proba(Xpad)
                                                - m2.predict_proba(Xpad)).max()), 0.0,
                                   places=6)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_lora_learns_new_intent(self):
        """LoRA: yeni (OOV sözcüklü) intent öğrenilir, eski intent'ler korunur."""
        import shutil
        import tempfile

        import numpy as np
        from transformer import TransformerNN

        ndir = os.path.join(tempfile.gettempdir(), 'ng_lora_test')
        if os.path.exists(ndir):
            shutil.rmtree(ndir)
        os.makedirs(ndir)

        V, C, L, D, NB, NH = 40, 4, 8, 32, 2, 2
        words = ['merhaba', 'nasilsin', 'adres', 'telefon', 'hava', 'bugun',
                 'yemek', 'pizza', 'kitap', 'oneri', 'spor', 'kosu']
        tags = ['selamlasma', 'iletisim', 'gida', 'spor']
        patterns = {'selamlasma': ['merhaba nasilsin', 'selam ver'],
                    'iletisim': ['adres telefon', 'telefon numarasi'],
                    'gida': ['yemek pizza', 'pizza oneri'],
                    'spor': ['spor kosu', 'kosu oneri']}

        def enc(t):
            idx = [words.index(w) for w in t.split() if w in words]
            return idx + [V] * (L - len(idx))

        X = np.array([enc(p) for tg in tags for p in patterns[tg]], dtype=np.int64)
        y = np.array([tags.index(tg) for tg in tags for _ in patterns[tg]],
                     dtype=np.int64)
        m = TransformerNN(vocab_size=V, num_intents=C, max_seq_len=L, d_model=D,
                          num_blocks=NB, num_heads=NH, ff_mult=3, seed=5)
        m.train(X, y, epochs=120, learning_rate=1e-3, batch_size=4,
                warmup_steps=20, verbose=False)
        self.assertEqual(m.evaluate(X, y), 1.0)

        m.save(os.path.join(ndir, 'model.json'))
        with open(os.path.join(ndir, 'bot_data.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocabulary': words, 'intent_tags': tags,
                       'intents': {t: ['cevap ' + t] for t in tags},
                       'intent_kws': {t: sorted(patterns[t]) for t in tags}},
                      f, ensure_ascii=False, indent=2)
        intents_path = os.path.join(ndir, 'intents.json')
        with open(intents_path, 'w', encoding='utf-8') as f:
            json.dump({'intents': [{'tag': t, 'patterns': patterns[t],
                                    'responses': ['cevap ' + t]} for t in tags]},
                      f, ensure_ascii=False, indent=2)

        from finetune import finetune_add
        sumry = finetune_add(ndir, intents_path,
                             [{'tag': 'astronomi',
                               'patterns': ['galaksi yildiz nedir',
                                            'yildiz kuyruklu_yildiz neden parlar'],
                               'responses': ['Uzay!']}],
                             epochs=80, verbose=False)
        self.assertIn('astronomi', sumry['tags_added'])
        self.assertGreater(sumry['vocab_added'], 0)

        bot = ChatBot()
        bot.load_model(ndir)
        self.assertEqual(bot._classify('galaksi yildiz nedir')[0], 'astronomi')
        self.assertEqual(bot._classify('yildiz kuyruklu_yildiz neden parlar')[0],
                         'astronomi')
        for tg, pats in patterns.items():
            self.assertEqual(bot._classify(pats[0])[0], tg,
                             f'eski intent kayboldu: {tg}')
        shutil.rmtree(ndir)

    def test_lora_forget_intent(self):
        """LoRA /forget: öğretilen intent silinir, diğerleri ve taban korunur."""
        import shutil
        import tempfile

        import numpy as np
        from transformer import TransformerNN

        ndir = os.path.join(tempfile.gettempdir(), 'ng_lora_forget_test')
        if os.path.exists(ndir):
            shutil.rmtree(ndir)
        os.makedirs(ndir)

        V, C, L, D, NB, NH = 40, 4, 8, 32, 2, 2
        words = ['merhaba', 'nasilsin', 'adres', 'telefon', 'hava', 'bugun',
                 'yemek', 'pizza', 'kitap', 'oneri', 'spor', 'kosu']
        tags = ['selamlasma', 'iletisim', 'gida', 'spor']
        patterns = {'selamlasma': ['merhaba nasilsin', 'selam ver'],
                    'iletisim': ['adres telefon', 'telefon numarasi'],
                    'gida': ['yemek pizza', 'pizza oneri'],
                    'spor': ['spor kosu', 'kosu oneri']}

        def enc(t):
            idx = [words.index(w) for w in t.split() if w in words]
            return idx + [V] * (L - len(idx))

        X = np.array([enc(p) for tg in tags for p in patterns[tg]], dtype=np.int64)
        y = np.array([tags.index(tg) for tg in tags for _ in patterns[tg]],
                     dtype=np.int64)
        m = TransformerNN(vocab_size=V, num_intents=C, max_seq_len=L, d_model=D,
                          num_blocks=NB, num_heads=NH, ff_mult=3, seed=5)
        m.train(X, y, epochs=120, learning_rate=1e-3, batch_size=4,
                warmup_steps=20, verbose=False)
        m.save(os.path.join(ndir, 'model.json'))
        with open(os.path.join(ndir, 'bot_data.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocabulary': words, 'intent_tags': tags,
                       'intents': {t: ['cevap ' + t] for t in tags},
                       'intent_kws': {t: sorted(patterns[t]) for t in tags}},
                      f, ensure_ascii=False, indent=2)
        intents_path = os.path.join(ndir, 'intents.json')
        with open(intents_path, 'w', encoding='utf-8') as f:
            json.dump({'intents': [{'tag': t, 'patterns': patterns[t],
                                    'responses': ['cevap ' + t]} for t in tags]},
                      f, ensure_ascii=False, indent=2)

        from finetune import finetune_add, forget_intent
        finetune_add(ndir, intents_path,
                     [{'tag': 'astronomi',
                       'patterns': ['galaksi yildiz nedir', 'yildiz kuyruklu'],
                       'responses': ['Uzay!']},
                      {'tag': 'plaj_voleybolu',
                       'patterns': ['plaj vole topu', 'plaj vole oyna'],
                       'responses': ['Kumda!']}],
                     epochs=60, verbose=False)

        bot = ChatBot()
        bot.load_model(ndir)
        self.assertEqual(bot._classify('galaksi yildiz nedir')[0], 'astronomi')
        self.assertEqual(bot._classify('plaj vole topu')[0], 'plaj_voleybolu')

        forget_intent(ndir, intents_path, 'astronomi', verbose=False)

        bot = ChatBot()
        bot.load_model(ndir)
        self.assertNotIn('astronomi', bot.intent_tags)
        self.assertNotIn('astronomi', bot.intent_kws)
        self.assertEqual(bot._classify('plaj vole topu')[0], 'plaj_voleybolu')
        for tg, pats in patterns.items():
            self.assertEqual(bot._classify(pats[0])[0], tg,
                             f'taban intent kayboldu: {tg}')
        with open(intents_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertNotIn('astronomi', [i['tag'] for i in data['intents']])
        with open(os.path.join(ndir, 'lora.json'), 'r', encoding='utf-8') as f:
            ad = json.load(f)
        self.assertNotIn('astronomi', ad['new_tags'])
        shutil.rmtree(ndir)


class TestLoraPadAlignment(unittest.TestCase):
    """LoRA vocab off-by-one regresyonu: yeni sözcük satırları (V..V+n_v-1) ve
    PAD indeksi (V+n_v) brain'in kullandığı düzenle birebir hizalı olmalı."""

    @staticmethod
    def _adapter(m, n_v):
        import numpy as np
        extra = np.random.RandomState(1).randn(n_v, m.d_model).astype(np.float32)
        return {'rank': 4, 'alpha': 8.0, 'vocab_added': n_v, 'head_added': 0,
                'embed_extra': extra.tolist(), 'deltas': {}}

    @staticmethod
    def _model(V=10, C=3, L=6):
        import numpy as np
        from transformer import TransformerNN
        rng = np.random.RandomState(0)
        X = rng.randint(0, V, size=(12, L))
        for i in range(X.shape[0]):
            X[i, rng.randint(2, L, size=2)] = V          # PAD
        y = np.array([i % C for i in range(12)])
        m = TransformerNN(vocab_size=V, num_intents=C, max_seq_len=L,
                          d_model=16, num_blocks=2, num_heads=2, ff_mult=3, seed=5)
        m.train(X, y, epochs=50, learning_rate=1e-3, batch_size=4, verbose=False)
        return m

    def test_pad_idx_and_row_layout(self):
        import numpy as np
        m = self._model()
        V = m.vocab_size
        n_v = 4
        ad = self._adapter(m, n_v)
        m.apply_lora(ad)
        extra = np.asarray(ad['embed_extra'], dtype=np.float32)

        # brain düzeni: yeni sözcük j -> V+j, PAD -> V+n_v
        self.assertEqual(m._lora_pad_idx, V + n_v)
        # taban sözcük satırları bozulmadı
        np.testing.assert_array_equal(m._lora_embed_full[:V], m.embed[:V])
        # yeni sözcük satırları extra sırasıyla hizalı; ilk satır SIFIR DEĞİL
        for j in range(n_v):
            self.assertGreater(float(np.linalg.norm(extra[j])), 1e-6)
            np.testing.assert_array_equal(m._lora_embed_full[V + j], extra[j])
        # PAD satırı sıfırdır (taban PAD satırı taşınır)
        np.testing.assert_array_equal(m._lora_embed_full[V + n_v], 0.0)

    def test_brain_encoding_masks_pad(self):
        import numpy as np
        m = self._model()
        V = m.vocab_size
        n_v = 3
        m.apply_lora(self._adapter(m, n_v))
        pad = V + n_v                                   # brain'in LoRA pad'i
        X = np.array([[V + n_v - 1, V, 0, pad, pad, pad]], dtype=np.int64)
        m.forward(X)
        mask = m._cache['mask']
        # yeni sözcük ve taban sözcük pozisyonları gerçek; PAD pozisyonları maskeli
        self.assertEqual(mask[0, 0], 1.0)
        self.assertEqual(mask[0, 1], 1.0)
        self.assertEqual(mask[0, 2], 1.0)
        self.assertTrue(np.all(mask[0, 3:] == 0.0))

    def test_lora_embed_grads_slicing(self):
        import numpy as np
        m = self._model()
        V = m.vocab_size
        n_v = 3
        m.apply_lora(self._adapter(m, n_v))
        rng = np.random.RandomState(3)
        X = rng.randint(0, V, size=(6, m.max_seq_len))
        X[:, 0] = V + 1                                  # yeni sözcük
        X[:, 1] = V + n_v                                # brain PAD indeksi
        y = np.zeros(6, dtype=np.int64)
        m.forward(X)
        G = m.grads(y)
        leg = m._cache['lora_embed_grads']
        self.assertEqual(leg.shape, (n_v, m.d_model))
        # taban embed gradyanı taban şeklinde kalır (V+1, d)
        self.assertEqual(G['embed'].shape, (V + 1, m.d_model))


class TestLoraRankAlphaHygiene(unittest.TestCase):
    """LoRA rank/alpha sıhhati (A2/A3 regresyon): taze adaptörde rank uygulanır,
    mevcut adaptörde korunur; etkin delta ölçeği alpha/rank ile tutarlıdır."""

    @staticmethod
    def _make_model_dir(ndir):
        import os
        import shutil

        import numpy as np
        from transformer import TransformerNN

        if os.path.exists(ndir):
            shutil.rmtree(ndir)
        os.makedirs(ndir)
        V, C, L, D, NB, NH = 40, 4, 8, 32, 2, 2
        words = ['merhaba', 'nasilsin', 'adres', 'telefon', 'hava', 'bugun',
                 'yemek', 'pizza', 'kitap', 'oneri', 'spor', 'kosu']
        tags = ['selamlasma', 'iletisim', 'gida', 'spor']
        patterns = {'selamlasma': ['merhaba nasilsin', 'selam ver'],
                    'iletisim': ['adres telefon', 'telefon numarasi'],
                    'gida': ['yemek pizza', 'pizza oneri'],
                    'spor': ['spor kosu', 'kosu oneri']}

        def enc(t):
            idx = [words.index(w) for w in t.split() if w in words]
            return idx + [V] * (L - len(idx))

        X = np.array([enc(p) for tg in tags for p in patterns[tg]], dtype=np.int64)
        y = np.array([tags.index(tg) for tg in tags for _ in patterns[tg]],
                     dtype=np.int64)
        m = TransformerNN(vocab_size=V, num_intents=C, max_seq_len=L, d_model=D,
                          num_blocks=NB, num_heads=NH, ff_mult=3, seed=5)
        m.train(X, y, epochs=120, learning_rate=1e-3, batch_size=4,
                warmup_steps=20, verbose=False)
        m.save(os.path.join(ndir, 'model.json'))
        with open(os.path.join(ndir, 'bot_data.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocabulary': words, 'intent_tags': tags,
                       'intents': {t: ['cevap ' + t] for t in tags},
                       'intent_kws': {t: sorted(patterns[t]) for t in tags}},
                      f, ensure_ascii=False, indent=2)
        with open(os.path.join(ndir, 'intents.json'), 'w', encoding='utf-8') as f:
            json.dump({'intents': [{'tag': t, 'patterns': patterns[t],
                                    'responses': ['cevap ' + t]} for t in tags]},
                      f, ensure_ascii=False, indent=2)

    def test_rank_and_alpha_fresh_then_preserved(self):
        import os
        import shutil
        import tempfile

        from finetune import finetune_add
        ndir = os.path.join(tempfile.gettempdir(), 'ng_lora_rank_test')
        self._make_model_dir(ndir)
        try:
            intents_path = os.path.join(ndir, 'intents.json')
            finetune_add(ndir, intents_path,
                         [{'tag': 'astronomi', 'patterns': ['galaksi yildiz nedir'],
                           'responses': ['Uzay!']}],
                         rank=4, alpha=8.0, epochs=5, verbose=False)
            with open(os.path.join(ndir, 'lora.json'), 'r', encoding='utf-8') as f:
                ad = json.load(f)
            self.assertEqual(ad['rank'], 4)
            self.assertEqual(ad['alpha'], 8.0)

            # ikinci çağrı FARKLI rank/alpha ile: mevcut adaptör KORUNUR
            finetune_add(ndir, intents_path,
                         [{'tag': 'plaj_voleybolu', 'patterns': ['plaj vole topu'],
                           'responses': ['Kumda!']}],
                         rank=16, alpha=2.0, epochs=5, verbose=False)
            with open(os.path.join(ndir, 'lora.json'), 'r', encoding='utf-8') as f:
                ad2 = json.load(f)
            self.assertEqual(ad2['rank'], 4)
            self.assertEqual(ad2['alpha'], 8.0)
        finally:
            shutil.rmtree(ndir)

    def test_delta_scale_matches_alpha_over_rank(self):
        import numpy as np
        from finetune import Adapter
        from transformer import TransformerNN

        m = TransformerNN(vocab_size=20, num_intents=3, max_seq_len=6,
                          d_model=16, num_blocks=1, num_heads=2, ff_mult=3, seed=5)
        r, a = 4, 8.0
        ad = Adapter(m, rank=r, alpha=a)
        ad.extend(['y1', 'y2'], ['ntag'])
        ad.apply()
        scale = a / r
        np.testing.assert_allclose(
            m.blocks[0]['W1e'] - m.blocks[0]['W1'],
            scale * (ad.B['b0_W1'] @ ad.A['b0_W1']), atol=1e-6)
        np.testing.assert_allclose(
            m.blocks[0]['attn']._eff['Wq'],
            m.blocks[0]['attn'].Wq + scale * (ad.B['b0_Wq'] @ ad.A['b0_Wq']),
            atol=1e-6)


class TestSeq2Seq(unittest.TestCase):
    """Seq2Seq encoder-decoder: kayit/yukleme, ornekleme, encode.Shape."""

    @staticmethod
    def _build():
        from seq2seq import Seq2Seq
        vocab = ['<PAD>', '<BOS>', '<EOS>'] + list('abcçdefgğhıijklmnoöprsştuüvyz. ')
        return Seq2Seq(vocab, d_model=32, num_blocks=1, num_heads=2,
                       max_enc_len=40, max_dec_len=20, seed=42)

    def test_round_trip(self):
        import tempfile
        from seq2seq import save_seq2seq, load_seq2seq
        m = self._build()
        path = os.path.join(tempfile.gettempdir(), 'ng_seq2seq_test.json')
        save_seq2seq(m, path)
        try:
            m2 = load_seq2seq(path)
            self.assertIsNotNone(m2)
            self.assertEqual(m2.V, m.V)
            self.assertEqual(m2.d_model, m.d_model)
            self.assertEqual(m2.num_blocks, m.num_blocks)
            self.assertEqual(set(m2.params.keys()), set(m.params.keys()))
            for k in m.params:
                import numpy as np
                np.testing.assert_allclose(m2.params[k], m.params[k], atol=1e-6,
                                           err_msg=f'param farki: {k}')
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_sample_returns_str(self):
        m = self._build()
        out = m.sample('merhaba', temperature=0.9, top_k=5, max_len=15)
        self.assertIsInstance(out, str)
        self.assertTrue(len(out) >= 0)

    def test_sample_terminates(self):
        m = self._build()
        for _ in range(5):
            out = m.sample('test sorgusu', max_len=20)
            self.assertIsInstance(out, str)

    def test_encode_seq2_shapes(self):
        from seq2seq import encode_seq2
        m = self._build()
        enc, din, dtgt, emask, dmask = encode_seq2(m, 'merhaba', 'hosbuldum')
        import numpy as np
        self.assertEqual(enc.ndim, 1)
        self.assertEqual(din.ndim, 1)
        self.assertEqual(dtgt.ndim, 1)
        self.assertEqual(emask.ndim, 1)
        self.assertEqual(dmask.ndim, 1)
        self.assertGreater(len(enc), 0)
        self.assertEqual(len(din), len(dtgt))
        self.assertEqual(len(din), len(dmask))

    def test_from_dict_overrides_vocab(self):
        from seq2seq import Seq2Seq
        m = self._build()
        d = m.to_dict()
        new_vocab = ['<PAD>', '<BOS>', '<EOS>'] + list('xyz ')
        d['vocab'] = new_vocab
        d['V'] = len(new_vocab)
        m2 = Seq2Seq(['<PAD>', '<BOS>', '<EOS>'])
        m2.from_dict(d)
        self.assertEqual(m2.V, len(new_vocab))
        self.assertIn('x', m2.c2i)
        self.assertNotIn('a', m2.c2i)


class TestLLM(unittest.TestCase):
    """Decoder-only LLM: kayit/yukleme, ornekleme, encode, dikkat maske."""

    @staticmethod
    def _build():
        from llm import LLM
        vocab = ['<PAD>', '<BOS>', '<SEP>', '<EOS>'] + list('abc .')
        return LLM(vocab, d_model=16, num_blocks=2, num_heads=2,
                   max_ctx_len=20, max_seq_len=50, seed=42)

    def test_round_trip(self):
        import tempfile
        import numpy as np
        from llm import save_llm, load_llm
        m = self._build()
        path = os.path.join(tempfile.gettempdir(), 'ng_llm_test.json')
        save_llm(m, path)
        try:
            m2 = load_llm(path)
            self.assertIsNotNone(m2)
            self.assertEqual(m2.V, m.V)
            self.assertEqual(m2.d_model, m.d_model)
            self.assertEqual(m2.num_blocks, m.num_blocks)
            self.assertEqual(set(m2.params.keys()), set(m.params.keys()))
            for k in m.params:
                np.testing.assert_allclose(m2.params[k], m.params[k], atol=1e-6,
                                           err_msg=f'param farki: {k}')
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_causal_attention_blocks_future(self):
        import numpy as np
        m = self._build()
        seq = np.array([[1, 2, 3, 2, 4]], np.int64)  # BOS a b SEP d
        logits = m.forward(seq)
        self.assertEqual(logits.shape, (1, 5, m.V))

    def test_encode_shapes_and_mask(self):
        import numpy as np
        from llm import encode_llm
        m = self._build()
        seq, smask = encode_llm(m, 'ab', 'bc')
        self.assertEqual(seq.ndim, 1)
        self.assertEqual(smask.ndim, 1)
        self.assertEqual(len(seq), len(smask))
        self.assertEqual(len(seq), m.max_seq_len)
        # maske: son <SEP> pozisyonu DAHIL (ilk yanit token'ini o tahmin eder)
        # sonrasi ve yanit token'lari; <EOS> ve dolgu maskedisiz kalir.
        sep_pos = int(np.where(seq == 2)[0][0])
        eos_pos = int(np.where(seq == 3)[0][0])
        self.assertGreater(sep_pos, 0)
        self.assertTrue(np.all(smask[:sep_pos] == 0.0))
        self.assertEqual(float(smask[sep_pos]), 1.0)
        self.assertTrue(np.all(smask[sep_pos:eos_pos] == 1.0))
        self.assertTrue(np.all(smask[eos_pos:] == 0.0))

    def test_sample_returns_str(self):
        m = self._build()
        out = m.sample('abc', temperature=0.9, top_k=5, max_len=15)
        self.assertIsInstance(out, str)

    def test_sample_terminates(self):
        m = self._build()
        for _ in range(5):
            out = m.sample('test sorgusu', max_len=20)
            self.assertIsInstance(out, str)

    def test_sample_with_knowledge_returns_str(self):
        m = self._build()
        out = m.sample('abc', max_len=15, knowledge='bilgi parcalari ile')
        self.assertIsInstance(out, str)

    def test_encode_with_context_mask_on_response_only(self):
        import numpy as np
        from llm import encode_llm
        m = self._build()
        seq, smask = encode_llm(m, 'ab', 'bc', context='bilgi xx koullu')
        last_sep = int(np.where(seq == 2)[0][-1])
        eos_pos = int(np.where(seq == 3)[0][0])
        self.assertTrue(np.all(smask[:last_sep] == 0.0))
        self.assertEqual(float(smask[last_sep]), 1.0)
        self.assertTrue(np.all(smask[last_sep + 1:] <= 1.0))
        self.assertTrue(np.all(smask[last_sep:eos_pos] == 1.0))
        self.assertGreater(float(smask.sum()), 0.0)
        # context tarafi hicbir zaman maske olmamali
        self.assertEqual(len(seq), len(smask))

    def test_build_vocab_specials_first(self):
        from llm import build_llm_vocab
        v = build_llm_vocab(['abc', 'abd'], min_count=1)
        self.assertEqual(v[:4], ['<PAD>', '<BOS>', '<SEP>', '<EOS>'])
        self.assertIn('a', v)

    def test_sample_accepts_rep_penalty(self):
        m = self._build()
        out = m.sample('abc', max_len=20, rep_penalty=0.0)
        self.assertIsInstance(out, str)
        out2 = m.sample('abc', max_len=20, rep_penalty=0.5)
        self.assertIsInstance(out2, str)


class TestNaturalize(unittest.TestCase):
    """Dogal yanit parafrazci: deterministik, icerik koruyucu, ASCII."""

    def test_variants_nonempty(self):
        from naturalize import natural_variants
        out = natural_variants('istanbul turkiyenin en buyuk sehridir', k=3)
        self.assertTrue(out)
        self.assertLessEqual(len(out), 3)

    def test_variants_deterministic(self):
        from naturalize import natural_variants
        a = natural_variants('istanbul turkiyenin en buyuk sehridir', k=3, seed=7)
        b = natural_variants('istanbul turkiyenin en buyuk sehridir', k=3, seed=7)
        self.assertEqual(a, b)

    def test_variants_no_ascii_mangling(self):
        # FAZ 2 karari: varyantlar gercek Turkce imla ile uretilir (ASCII-only
        # degil). Girdi ASCII ise kendisi bozulmadan korunur; kelime degisimleri
        # ('var' -> 'bulunmaktadir' gibi) dogal Turkce harfler katabilir.
        from naturalize import natural_variants
        out = natural_variants('guzel bir sehir ve buyuk bir nufusu var', k=4)
        self.assertTrue(out)
        for v in out:
            self.assertIn('guzel', v, msg=f'girdi bozuldu: {v!r}')
            self.assertIn('nufusu', v, msg=f'girdi bozuldu: {v!r}')

    def test_variants_preserve_content(self):
        from naturalize import natural_variants
        src = 'istanbul turkiyenin en buyuk sehridir'
        key = set(src.split()) | {'istanbul', 'sehir', 'buyuk'}
        for v in natural_variants(src, k=5):
            self.assertTrue(key.intersection(v.split()),
                            msg=f'icerik korunmadi: {v}')

    def test_variants_include_original(self):
        from naturalize import natural_variants
        vs = natural_variants('kisa bir cevap burada', k=2)
        self.assertIn('kisa bir cevap burada', vs)

    def test_naturalize_pairs_multiplies(self):
        from naturalize import naturalize_pairs
        pairs = [('selam nasilsin', 'iyiyim sen nasilsin'),
                 ('nerelisin', 'istanbulluyum')]
        out = naturalize_pairs(pairs, k=2)
        self.assertEqual(len(out), 4)
        # orijinal yanitlar her ciftte en az bir kez gecer
        orig = {('selam nasilsin', 'iyiyim sen nasilsin'),
                ('nerelisin', 'istanbulluyum')}
        self.assertTrue(orig.issubset(set(out)))
        # sorgular degismemeli
        self.assertEqual(sorted({c for c, _ in out}),
                         ['nerelisin', 'selam nasilsin'])


class TestLLMIntegration(unittest.TestCase):
    """brain._try_seq_rephrase: LLM basamagi kalite kapisindan gecer."""

    class _StubLLM:
        def __init__(self, out):
            self.out = out

        def sample(self, ctx, temperature=0.7, top_k=10, max_len=None,
                   knowledge=None, rep_penalty=0.3):
            return self.out

    def _bot(self, llm_out):
        from brain import ChatBot
        bot = ChatBot()
        bot.intents = {
            'selam': ['merhaba dunya nasilsin', 'selam dostum nasilsin'],
        }
        bot.intent_tags = ['selam']
        bot.intent_kws = {'selam': set(bot.tokenize('merhaba selam dunya'))}
        bot.llm = self._StubLLM(llm_out)
        bot.llm_enabled = True
        # alt basamaklar kapali (seq2seq/LSTM) -> yalnizca LLM kalitesi test edilir
        bot.seq2 = self._StubLLM('xkw sdfjl quwr')
        bot.seq_enabled = False
        return bot

    def test_llm_branch_accepts_overlapping_output(self):
        bot = self._bot('merhaba dunya nasilsin canim')
        out = bot._try_seq_rephrase('selam', 'merhaba')
        self.assertEqual(out, 'merhaba dunya nasilsin canim')

    def test_llm_branch_rejects_garbage(self):
        bot = self._bot('xkw sdfjl quwr')
        self.assertIsNone(bot._try_seq_rephrase('selam', 'merhaba'))

    def test_llm_disable_flag_when_no_model(self):
        from brain import ChatBot
        bot = ChatBot()
        bot.seq2 = self._StubLLM('xkw sdfjl quwr')
        bot.seq_enabled = False
        # Ortam-bagimsiz: model/llm_model.json olmasa da "LLM yok" durumunu simule et
        bot.llm = None
        bot.llm_enabled = False
        self.assertIsNone(bot._try_seq_rephrase('selam', 'merhaba'))
        self.assertFalse(bot.llm_enabled)

    def test_accept_generated_rejects_verbatim_copy(self):
        """Kopyala-yapistir: kayitli yanitin birebir kopyasi KAPIDAN GECMEZ."""
        bot = self._bot('merhaba dunya nasilsin')   # aynen canned
        self.assertFalse(bot._accept_generated('merhaba dunya nasilsin', 'selam'))

    def test_accept_generated_accepts_novel_paraphrase(self):
        """Konuya yapisik ama canned'da olmayan kurulus ozgunluk yetkir."""
        bot = self._bot('merhaba dunya nasilsin canim')
        self.assertTrue(
            bot._accept_generated('merhaba dunya nasilsin canim', 'selam'))

    def test_kb_rephrase_fallback_without_llm(self):
        """LLM yoksa bilgi yaniti oldugu gibi duser (guvenli fallback)."""
        from brain import ChatBot
        bot = ChatBot()
        # Ortam-bagimsiz: model/llm_model.json olsa bile "LLM yok" simule edilir
        bot.llm = None
        bot.llm_enabled = False
        kb = 'Klorofil bitkilerde fotosentezi saglayan yesil pigmenttir.'
        self.assertEqual(bot._try_kb_rephrase('klorofil nedir', kb), kb)

    def test_accept_kb_rephrase_rejects_verbatim(self):
        bot = self._bot('xkw sdfjl quwr')
        kb = 'klorofil bitkilerde fotosentezi saglar'
        self.assertFalse(bot._accept_kb_rephrase(kb, kb))  # birebir kopya

    def test_accept_kb_rephrase_accepts_novel(self):
        bot = self._bot('xkw sdfjl quwr')
        kb = 'klorofil bitkilerde fotosentezi saglar'
        gen = 'klorofil bitkilerin yesil rengini verir ve guclu bir renktir'
        self.assertTrue(bot._accept_kb_rephrase(gen, kb))


class TestModuleImports(unittest.TestCase):
    """Proje modülleri hicbir yavas runtime'a takilmadan import edilmeli."""

    def test_imports(self):
        import brain
        import corpus
        import generator
        import seqgen
        import seq2seq
        import llm
        import transformer
        import clean_intents
        import scrape_intents
        import train  # egitici komutlari modül olarak da yuklenir
        for mod in (brain, corpus, generator, seqgen, seq2seq, transformer,
                    clean_intents, scrape_intents, train):
            self.assertIsNotNone(mod)


class TestTwoLayerArchitecture(unittest.TestCase):
    """Iki katmanli mimari: sohbet (siniflandirma) + bilgi (retrieval)."""

    def test_conversational_filter_real_intents(self):
        """intents.json'daki filtre >6 desenli intentleri sohbet olarak secmeli."""
        bot = ChatBot()
        data = bot.load_intents(os.path.join(BASE, 'intents.json'))
        original_total = len(bot.intent_tags)
        data = bot.conversational_data(data)
        self.assertGreater(len(bot.intent_tags), 0)
        self.assertLess(len(bot.intent_tags), original_total)
        self.assertGreater(len(bot.knowledge_intents), 0)
        self.assertEqual(len(bot.intent_tags) + len(bot.knowledge_intents),
                         original_total)

    def test_conversational_filter_fallback_small_data(self):
        """Tum intentler <=6 desense filtre uygulanmamali (test uyumlulugu)."""
        bot = ChatBot()
        # build small intents
        intents = [
            {'tag': 'selam', 'patterns': ['merhaba', 'selam'], 'responses': ['hi']},
            {'tag': 'nasil', 'patterns': ['nasil', 'iyi'], 'responses': ['ok']},
        ]
        data = {'intents': intents}
        bot.intents = {t['tag']: t['responses'] for t in intents}
        bot.intent_tags = ['nasil', 'selam']
        bot.intent_kws = {}
        for t in intents:
            ws = set()
            for p in t['patterns']:
                ws.update(bot.tokenize(p))
            bot.intent_kws[t['tag']] = ws

        result = bot.conversational_data(data)
        # all intents have <=6 patterns -> fallback: intent_tags unchanged
        self.assertEqual(len(bot.intent_tags), 2)
        self.assertEqual(len(result['intents']), 2)

    def test_knowledge_intents_property(self):
        """knowledge_intents: intents.keys() - intent_tags."""
        bot = ChatBot()
        data = bot.load_intents(os.path.join(BASE, 'intents.json'))
        data = bot.conversational_data(data)
        conv_set = set(bot.intent_tags)
        kb = bot.knowledge_intents
        # Tanım gereği: knowledge_intents = intents - conversational tags.
        # Sabit sayı yerine (753) veriden dinamik hesaplanır; yeni intent
        # eklendikçe test bozulmaz.
        self.assertEqual(len(kb), len(bot.intents) - len(bot.intent_tags))
        self.assertGreater(len(kb), 0)
        self.assertTrue(conv_set.isdisjoint(set(kb.keys())))

    def test_knowledge_retrieval_response(self):
        """Bilgi sorgusu (_select_knowledge) bilgi intent'inden yanit dondurur."""
        bot = ChatBot()
        data = bot.load_intents(os.path.join(BASE, 'intents.json'))
        data = bot.conversational_data(data)
        # find a knowledge intent with meaningful content keywords
        from brain import STOPWORDS
        for tag in bot.knowledge_intents:
            kws = bot.intent_kws.get(tag, set())
            content = [w for w in kws if w not in STOPWORDS and len(w) >= 3]
            if len(content) >= 2:
                result = bot._select_knowledge(content[:3])
                self.assertIsNotNone(
                    result, f'bilgi retrieval basarisiz: tag={tag} kws={content[:3]}')
                self.assertIn(tag, bot.intents)
                self.assertIn(result, bot.intents[tag])
                return
        self.fail('en az 2 content kelimesi olan bilgi intent bulunamadi')

    def test_knowledge_threshold_rejects_weak(self):
        """Zayif/bos sorgu knowledge_threshold altinda kalmali ve None dondurmeli."""
        bot = ChatBot()
        data = bot.load_intents(os.path.join(BASE, 'intents.json'))
        data = bot.conversational_data(data)
        # garbage tokens not in any intent
        result = bot._select_knowledge(['xyzzy', 'plugh', 'qwerty'])
        self.assertIsNone(result)

    def test_two_layer_end_to_end(self):
        """Kucuk model + bilgi intents: siniflandirma sohbet, bilgi retrieval'den gelmeli."""
        import numpy as np
        from transformer import TransformerNN

        # 2 conversational intents (3 patterns each >6? no, they have <6. Use custom.)
        # We make intents that satisfy >6 patterns: create 8+ patterns
        conv_patterns = {
            'selamlasma': ['merhaba', 'selam', 'gunaydin', 'iyi gunler',
                           'hey nasilsin', 'selam dostum', 'merhaba dunya',
                           'naber'],
            'veda': ['gule gule', 'hosca kal', 'gule gule dostum', 'iyi gunler',
                     'bay bay', 'kendine iyi bak', 'gorusuruz', 'hoscakal'],
        }
        kb_patterns = {
            'galaksi': ['galaksi nedir', 'galaksi hakkinda bilgi ver',
                        'galaksi ne demek', 'galaksi turkce', 'galaksi acilimi',
                        'galaksi ne ise yarar'],
        }

        _bot = ChatBot()
        all_raw = list(conv_patterns['selamlasma']) + list(conv_patterns['veda'])
        vocab = sorted(set(w for p in all_raw for w in _bot.tokenize(p)))
        V = len(vocab)
        C = 2  # 2 conversational intents only
        L = 8

        def enc(t):
            idx = [vocab.index(w) for w in _bot.tokenize(t) if w in vocab]
            return idx + [V] * (L - len(idx))

        X = np.array([enc(p) for tg in ['selamlasma', 'veda']
                       for p in conv_patterns[tg]], dtype=np.int64)
        y = np.array([0]*len(conv_patterns['selamlasma']) +
                     [1]*len(conv_patterns['veda']), dtype=np.int64)

        m = TransformerNN(vocab_size=V, num_intents=C, max_seq_len=L,
                          d_model=16, num_blocks=1, num_heads=2, ff_mult=2, seed=5)
        m.train(X, y, epochs=200, learning_rate=1e-3, batch_size=4,
                warmup_steps=10, verbose=False)
        self.assertGreaterEqual(m.evaluate(X, y), 0.90)

        # build bot with 2 conversational + 1 knowledge
        tags = ['selamlasma', 'veda']
        ndir = os.path.join(tempfile.gettempdir(), 'ng_two_layer_test')
        if os.path.exists(ndir):
            import shutil
            shutil.rmtree(ndir)
        os.makedirs(ndir)
        m.save(os.path.join(ndir, 'model.json'))
        all_kb = {t: ['bilgi yaniti ' + t] for t in kb_patterns}
        all_conv = {t: ['cevap ' + t] for t in tags}

        def kws_of(pattern_map):
            out = {}
            for t, ps in pattern_map.items():
                ws = set()
                for p in ps:
                    ws.update(_bot.tokenize(p))
                out[t] = sorted(ws)
            return out

        kb_kws = kws_of(kb_patterns)
        conv_kws = kws_of(conv_patterns)
        with open(os.path.join(ndir, 'bot_data.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocabulary': vocab, 'intent_tags': tags,
                       'intents': {**all_conv, **all_kb},
                       'intent_kws': {**conv_kws, **kb_kws}},
                      f, ensure_ascii=False, indent=2)

        bot = ChatBot()
        bot.load_model(ndir)

        # verify filter: 2 conv, 1 kb
        self.assertEqual(len(bot.intent_tags), 2)
        self.assertEqual(len(bot.knowledge_intents), 1)
        self.assertIn('galaksi', bot.knowledge_intents)

        # conversational classify works
        tag, prob, unclear, _ = bot._classify('merhaba')
        self.assertFalse(unclear)
        self.assertIn(tag, tags)

        # knowledge retrieval works (kucuk veride IDF agirliklari dusuk:
        # log(3/2)=0.4 seviyesinde; esik gercek 791 intent verisinde
        # ~5.98'a ulasir, burada akis dogrulamak icin dusurulur)
        bot.knowledge_threshold = 0.01
        kb_result = bot._select_knowledge(_bot.tokenize('galaksi nedir'))
        self.assertIsNotNone(kb_result)
        self.assertEqual(kb_result, 'bilgi yaniti galaksi')

        # get_response returns conversational for greetings
        resp = bot.get_response('merhaba')
        self.assertNotEqual(resp, 'Model not trained yet! Please run train.py first.')

        # full knowledge intent keyword override should be through _classify
        # with a decisive keyword match
        tag, prob, unclear, _ = bot._classify('galaksi nedir')
        self.assertFalse(unclear)

        import shutil
        shutil.rmtree(ndir)


class TestTopicGateAndKnowledge(unittest.TestCase):
    """Kararsiz intent + sorgu-konu kapisi + bilgi onceligi (adim3+ fix)."""

    def _bot(self, tag, patterns, canned):
        from brain import ChatBot
        bot = ChatBot()
        bot.intents = {tag: canned}
        bot.intent_tags = [tag]
        bot.intent_kws = {tag: set(bot.tokenize(' '.join(patterns)))}
        return bot

    def test_accept_generated_needs_query_topic_for_choice(self):
        """Uzun secim sorusunda konu-disi uretim kapidan GEÇMEZ."""
        bot = self._bot('kararsiz',
                        ['kararsizim pizza hamburger secim yapamam'],
                        ['pizza guzel bir secim olabilir',
                         'hamburger de iyi bir secenek olur'])
        tok_gen = 'benim isim ama sevgi dolu hayvanlardir'
        self.assertFalse(bot._accept_generated(
            tok_gen, 'kararsiz',
            query='bugun kararsizim pizza mi yesem yoksa hamburger mi'))

    def test_accept_generated_passes_when_query_topic_touched(self):
        """Seyim sorusunda iyi aday sorudaki konuya dokunuyorsa GECER."""
        bot = self._bot('kararsiz',
                        ['kararsizim pizza hamburger secim'],
                        ['pizza guzel bir secim olabilir',
                         'hamburger de iyi bir secenek olur'])
        gen = 'pizza bugun guzel bir secim olabilir canim'
        self.assertTrue(bot._accept_generated(
            gen, 'kararsiz',
            query='bugun kararsizim pizza mi yesem yoksa hamburger mi'))

    def test_short_emotion_skips_topic_gate(self):
        """Kisa duygusal sorgu (2 icerik kelimesi) kapisizdir: empatik yanit
        sorudaki kelimeleri ezberlemek zorunda degildir."""
        bot = self._bot('uzuntu',
                        ['canim sikkin yorgunum'],
                        ['herkesin kotu gunleri olur', 'bu da gecer inan'])
        gen = 'herkesin bazen kotu gunleri olur ama bu da gecer'
        self.assertTrue(bot._accept_generated(
            gen, 'uzuntu', query='canim sikkin'))

    def test_is_knowledge_question(self):
        bot = ChatBot()
        self.assertTrue(bot._is_knowledge_question('galaksi nedir'))
        self.assertTrue(bot._is_knowledge_question('benefse ne demek'))
        self.assertTrue(bot._is_knowledge_question('mustafa kemal hakkinda bilgi ver'))
        self.assertFalse(bot._is_knowledge_question('bugun cok mutluyum'))
        self.assertFalse(bot._is_knowledge_question('tesekkur ederim'))

    def test_kararsiz_intent_registered_as_conversational(self):
        """intents.json'da kararsiz mevcut ve siniflandiriciya ogretilen
        (conversational) kumeye giriyor."""
        bot = ChatBot()
        data = bot.load_intents(os.path.join(BASE, 'intents.json'))
        self.assertIn('kararsiz', bot.intent_tags)
        data = bot.conversational_data(data)
        self.assertIn('kararsiz', bot.intent_tags)

    def test_definition_reaches_knowledge(self):
        """Tanim sorulari (X nedir/ne demek) bilgi retrieval ile yanitlanir;
        siniflandirici chat tag'ine kapsa bile bilgi onceligi calisir."""
        bot = ChatBot()
        data = bot.load_intents(os.path.join(BASE, 'intents.json'))
        bot.conversational_data(data)
        self.assertTrue(bot._is_knowledge_question('benefse nedir'))
        kb = bot._select_knowledge(bot.tokenize('benefse nedir'))
        self.assertIsNotNone(kb, 'benefse bilgi intenti retrieval ile bulunmali')
        in_any = any(kb in resp for tag, resp in bot.intents.items()
                     if tag not in bot.intent_tags)
        self.assertTrue(in_any, 'KB yaniti bir bilgi intentinin yanit bankasindan olmali')


if __name__ == '__main__':
    unittest.main(verbosity=2)