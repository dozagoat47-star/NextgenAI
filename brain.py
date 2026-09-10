"""
Nextgen AI - Neural Network from Scratch
Built using only numpy, no TensorFlow or PyTorch.
"""

import numpy as np
import json
import math
import random
import string
import os


# Turkce islev/durak kelimeleri: anahtar kelime dikkatinde agirligi sifir.
# (Cumlede konuyu tasimazlar; "hangi, ne, mi" gibi her yerde gecerler.)
STOPWORDS = {
    'mi', 'mu', 'miyim', 'misin', 'misiniz', 'msin', 'msiniz',
    'sen', 'ben', 'bana', 'sana', 'bu', 'su', 'neyi', 'hangisi',
    'ne', 'hangi', 'kac', 'nasil', 'neden', 'nicin', 'bir', 'da', 'de',
    'ya', 'ki', 'miydi', 'midir', 'neydi', 'var', 'yok',
    # soru/template parcalari: konu tasimazlar, IDF onlari yanlis guclendirmesin
    'zam', 'kurult', 'hakk', 'bilk', 'ver', 'bilg', 'anlat', 'soyle',
    'kal', 'olur', 'olabilir', 'edebil', 'eder', 'onerr', 'oner',
}


class NeuralNetwork:
    """
    Basit bir feedforward sinir ağı.
    Sadece numpy ile sıfırdan yazılmıştır.
    """

    def __init__(self, layer_sizes):
        """
        Sinir ağını başlatır.

        Args:
            layer_sizes: Her katmandaki nöron sayısını içeren liste
                         Örn: [input_size, hidden1, hidden2, output_size]
        """
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.num_layers = len(layer_sizes)

        # Her katman için ağırlıkları ve bias'ları başlat (He initialization)
        for i in range(self.num_layers - 1):
            # He initialization - ReLU için ideal
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        """ReLU aktivasyon fonksiyonu"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """ReLU'nun türevi"""
        return (z > 0).astype(float)

    def softmax(self, z):
        """Softmax aktivasyon fonksiyonu (çıkat için)"""
        # Overflow önlemek için en büyük değeri çıkarıyoruz
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """
        İleri yayılım (forward propagation).

        Args:
            X: Giriş verisi (batch_size, input_size)

        Returns:
            Tahmin sonucu
        """
        self.z_list = []  # Her katmanın线性 çıkışı
        self.a_list = [X]  # Her katmanın aktivasyonu

        current_input = X

        for i in range(self.num_layers - 1):
            # Lineer dönüşüm: z = X * W + b
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_list.append(z)

            # Aktivasyon fonksiyonu
            if i < self.num_layers - 2:  # Gizli katmanlar için ReLU
                a = self.relu(z)
            else:  # Çıkış katmanı için Softmax
                a = self.softmax(z)

            self.a_list.append(a)
            current_input = a

        return current_input

    def compute_loss(self, y_pred, y_true):
        """
        Cross-entropy kaybı hesaplar.

        Args:
            y_pred: Tahmin edilen olasılıklar
            y_true: Gerçek etiketler (one-hot encoded)

        Returns:
            Kayıp değeri
        """
        m = y_true.shape[0]
        # Cross-entropy loss
        log_likelihood = -np.log(y_pred[np.arange(m), y_true] + 1e-8)
        return np.mean(log_likelihood)

    def backward(self, y_true, learning_rate):
        """
        Geri yayılım (backpropagation) ve ağırlık güncelleme.

        Args:
            y_true: Gerçek etiketler
            learning_rate: Öğrenme hızı
        """
        m = y_true.shape[0]
        deltas = [None] * (self.num_layers - 1)

        # Çıkış katmanı için hata
        output = self.a_list[-1]
        delta_last = output.copy()
        delta_last[np.arange(m), y_true] -= 1
        deltas[-1] = delta_last / m

        # Gizli katmanlar için geriye doğru hata hesapla
        for i in range(self.num_layers - 3, -1, -1):
            delta = np.dot(deltas[i + 1], self.weights[i + 1].T) * self.relu_derivative(self.z_list[i])
            deltas[i] = delta

        # Ağırlıkları ve bias'ları güncelle
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * np.dot(self.a_list[i].T, deltas[i])
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    def train(self, X, y, epochs=1000, learning_rate=0.01, batch_size=8, verbose=True):
        """
        Modeli eğitir.

        Args:
            X: Eğitim verileri
            y: Etiketler
            epochs: Eğitim döngüsü sayısı
            learning_rate: Öğrenme hızı
            batch_size: Batch boyutu
            verbose: Eğitim bilgisi yazdırılsın mı?
        """
        losses = []
        m = X.shape[0]

        for epoch in range(epochs):
            # Verileri karıştır
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            num_batches = 0

            # Mini-batch gradient descent
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # İleri yayılım
                output = self.forward(X_batch)

                # Kayıp hesapla
                loss = self.compute_loss(output, y_batch)
                epoch_loss += loss
                num_batches += 1

                # Geri yayılım ve güncelleme
                self.backward(y_batch, learning_rate)

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 100 == 0:
                accuracy = self.evaluate(X, y)
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2%}")

        return losses

    def predict(self, X):
        """Tahmin yapar"""
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def predict_proba(self, X):
        """Olasılık tahminleri yapar"""
        return self.forward(X)

    def evaluate(self, X, y):
        """Doğruluk oranını hesaplar"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def save(self, filepath):
        """Modeli kaydeder"""
        model_data = {
            'layer_sizes': self.layer_sizes,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
        print(f"Model saved: {filepath}")

    def load(self, filepath):
        """Modeli yükler"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        self.layer_sizes = model_data['layer_sizes']
        self.weights = [np.array(w) for w in model_data['weights']]
        self.biases = [np.array(b) for b in model_data['biases']]
        self.num_layers = len(self.layer_sizes)
        print(f"Model loaded: {filepath}")


class ChatBot:
    """
    Sohbet botu - Sinir ağı kullanarak niyetleri tahmin eder.
    """

    def __init__(self):
        self.model = None
        self.intents = {}
        self.all_patterns = []
        self.vocabulary = []
        self.intent_tags = []
        self.stem_cache = {}

    def ascii_normalize(self, text):
        """
        Turkce ozel karakterleri ASCII karsiliklarina cevirir.
        Boylece 'ogle' / 'öğle' / 'OĞLE' hepsi ayni kelimeye donusur.
        """
        turkish_to_ascii = {
            'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
            'â': 'a', 'î': 'i', 'û': 'u', 'i': 'i', 'o': 'o', 'u': 'u',
            'Ç': 'c', 'Ğ': 'g', 'İ': 'i', 'I': 'i', 'Ö': 'o', 'Ş': 's', 'Ü': 'u',
            'Â': 'a', 'Î': 'i', 'Û': 'u',
            '\u0307': '',  # Python'in 'İ'.lower() ciktisindaki kombinasyon noktasi
        }
        return text.translate(str.maketrans(turkish_to_ascii))

    def simple_stem(self, word):
        """Basit Turkce kelime koku bulma (stemming)"""
        if word in self.stem_cache:
            return self.stem_cache[word]

        original = word
        stem = word

        # onceki tek-adim yaklasimi yerine, ekleri tekrarli sekilde temizle
        suffixes = [
            'misiniz', 'miyim', 'musunuz', 'misin', 'lerimiz', 'larimiz',
            'leriniz', 'lariniz', 'lerine', 'larina', 'lerin', 'larin',
            'lari', 'leri', 'lar', 'ler',
            'den', 'dan', 'ten', 'tan', 'de', 'da', 'te', 'ta',
            'nin', 'nin', 'nun', 'nun', 'in', 'in', 'un', 'un',
            'mis', 'mis', 'mis', 'mus', 'im', 'im', 'um', 'um',
            'sin', 'sin', 'sun', 'sin', 'i', 'i', 'u', 'u',
            'ecek', 'acak', 'erek', 'arak', 'mel', 'mal', 'mek', 'mak',
            'iyor', 'uyor', 'yor', 'ir', 'ar', 'er', 'en', 'an',
            'ye', 'ya', 'yi', 'yi', 'yle', 'yla'
        ]

        while True:
            removed = False
            for suffix in suffixes:
                if len(stem) - len(suffix) >= 3 and stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    removed = True
                    break
            if not removed:
                break

        # Turkce son sessiz sedasizlasma (devoicing): yemek/yemegi -> yemek
        # 'ozellikleri' gibi koklerde g/g/k tutarliligi saglar
        if len(stem) >= 4 and stem[-1] in 'gbd':
            stem = stem[:-1] + {'g': 'k', 'b': 'p', 'd': 't'}[stem[-1]]

        self.stem_cache[original] = stem
        return stem

    def tokenize(self, text):
        """Metni kelimelere ayırır ve temizler"""
        # noktalama işaretlerini kaldır
        text = text.lower()
        text = self.ascii_normalize(text)
        for char in string.punctuation:
            text = text.replace(char, '')
        words = text.split()
        return [self.simple_stem(w) for w in words]

    def bag_of_words(self, words):
        """Bag of words vektörü oluşturur"""
        bag = [0] * len(self.vocabulary)
        for w in words:
            if w in self.vocabulary:
                bag[self.vocabulary.index(w)] = 1
        return bag

    def load_intents(self, filepath):
        """Niyet dosyasını yükler"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.intents = {intent['tag']: intent['responses'] for intent in data['intents']}
        self.all_patterns = []
        for intent in data['intents']:
            for pattern in intent['patterns']:
                self.all_patterns.append(pattern.lower())

        # Tüm kelimeleri ve etiketleri topla
        all_words = []
        tags = []

        for intent in data['intents']:
            tag = intent['tag']
            if tag not in tags:
                tags.append(tag)

            for pattern in intent['patterns']:
                words = self.tokenize(pattern)
                all_words.extend(words)

        # Benzersiz kelimeleri sırala
        self.vocabulary = sorted(list(set(all_words)))
        self.intent_tags = sorted(tags)

        # Her niyet icin normalizasyonlu anahtar kelime kumesi
        # (sohbet esnasinda niyet secimine yardimci olur)
        self.intent_kws = {}
        for tag in self.intent_tags:
            self.intent_kws[tag] = set()
        for intent in data['intents']:
            tag = intent['tag']
            kws = set()
            for pattern in intent['patterns']:
                kws.update(self.tokenize(pattern))
            if tag in self.intent_kws:
                self.intent_kws[tag].update(kws)

        self._build_keyword_weights()
        return data

    def _build_keyword_weights(self):
        """Anahtar kelime dikkati icin IDF agirliklarini hesaplar.

        Seyrek gecen kelime yuksek agirlik (ayirt edici), her intent'te
        gecen ortak kelimeler dusuk agirlik alir.
        """
        n_intents = len(self.intent_tags) or 1
        doc_count = {}
        for tag in self.intent_tags:
            for stem in self.intent_kws.get(tag, ()):
                doc_count[stem] = doc_count.get(stem, 0) + 1
        weights = {}
        for stem, count in doc_count.items():
            w = math.log(n_intents / (1.0 + count))
            weights[stem] = 0.0 if stem in STOPWORDS else w
        self.keyword_weights = weights

    def prepare_training_data(self, intents_data):
        """Eğitim verilerini hazırlar"""
        X = []
        y = []

        for intent in intents_data['intents']:
            tag = intent['tag']
            tag_index = self.intent_tags.index(tag)

            for pattern in intent['patterns']:
                words = self.tokenize(pattern)
                bag = self.bag_of_words(words)
                X.append(bag)
                y.append(tag_index)

        return np.array(X), np.array(y)

    def train_model(self, intents_filepath, epochs=1000, learning_rate=0.01):
        """Train the model"""
        print("Loading intents file...")
        data = self.load_intents(intents_filepath)

        print("Preparing training data...")
        X, y = self.prepare_training_data(data)

        input_size = len(self.vocabulary)
        output_size = len(self.intent_tags)
        hidden1 = min(64, max(16, input_size // 2))
        hidden2 = min(32, max(8, output_size * 2))

        print(f"\nModel Architecture:")
        print(f"  Input layer:  {input_size} neurons (vocabulary size)")
        print(f"  Hidden1:      {hidden1} neurons")
        print(f"  Hidden2:      {hidden2} neurons")
        print(f"  Output layer: {output_size} neurons (intent count)")
        total = input_size*hidden1 + hidden1 + hidden1*hidden2 + hidden2 + hidden2*output_size + output_size
        print(f"  Total params: {total}")
        print(f"\nTraining samples: {len(X)}")
        print(f"Total words: {input_size}")
        print(f"Total intents: {output_size}")
        print()

        self.model = NeuralNetwork([input_size, hidden1, hidden2, output_size])

        print("Training started...")
        losses = self.model.train(X, y, epochs=epochs, learning_rate=learning_rate)

        final_accuracy = self.model.evaluate(X, y)
        print(f"\nTraining completed! Accuracy: {final_accuracy:.2%}")

        return losses

    def get_probability(self, user_input):
        """User input icin en guclu intent tahminini ve olasiligini dondurur."""
        if self.model is None or not self.vocabulary:
            return None, 0.0
        words = self.tokenize(user_input)
        bag = self.bag_of_words(words)
        X = np.array([bag])
        probabilities = self.model.predict_proba(X)[0]
        best_index = int(np.argmax(probabilities))
        return self.intent_tags[best_index], float(probabilities[best_index])

    def _classify(self, user_input):
        """Input icin secilen intent etiketini, guveni ve belirsizlik bayragini dondurur.

        Keyword override dahil get_response ile birebir ayni secim mantigini kullanir.
        """
        words = self.tokenize(user_input)
        bag = self.bag_of_words(words)
        X = np.array([bag])

        probabilities = self.model.predict_proba(X)[0]
        best_index = np.argmax(probabilities)
        best_probability = float(probabilities[best_index])
        best_tag = self.intent_tags[best_index]

        # ANAHTAR KELIME DIKKATI: IDF agirlikli onem skoru.
        # Seyrek/ayirt edici kelimeler (galaksi, gazneli, yardim) yuksek,
        # her yerde gecenler (hangi, ne, mi) sifira yakin agirliktadir.
        attn = {}
        input_set = set(words)
        for tag in self.intent_tags:
            inter = input_set & self.intent_kws.get(tag, set())
            s = sum(self.keyword_weights.get(w, 0.0) for w in inter)
            if s > 0:
                attn[tag] = s
        cls_attn = sum(self.keyword_weights.get(w, 0.0)
                       for w in (input_set & self.intent_kws.get(best_tag, set())))

        if not attn and best_probability < 0.15:
            return 'Anlayamadim', best_probability, True

        chosen_tag = best_tag
        if attn:
            top_attn_tag = max(attn, key=lambda t: (attn[t], self.intent_tags.index(t)))
            top_attn = attn[top_attn_tag]
            # Ayirt edici anahtar kelime eslesmesi (>=1.5) siniflandiriciyi asar;
            # siniflandirici emin degilse (<0.85) daha zayif eslesme de yeter.
            decisive = top_attn >= 1.5 and (
                best_probability < 0.85 or top_attn >= cls_attn + 1.0)
            if decisive:
                chosen_tag = top_attn_tag
        return chosen_tag, best_probability, False

    def predict(self, user_input):
        """Disa aktarmadan tahmin kullanimi icin (Excel taramalari gibi).

        Returns:
            (etiket, guven_yuzdesi) ; anlayamazsa ('Anlayamadim', 0.0)
        """
        if self.model is None:
            return 'Model yok', 0.0
        tag, probability, unclear = self._classify(user_input)
        if unclear:
            return 'Anlayamadim', 0.0
        return tag, round(probability * 100, 2)

    def keyword_strength(self, user_input):
        """Giris icin en guclu anahtar kelime dikkat skorunu dondurur (0.0 - ~10).

        Bilgi sorusu yanlis yerel intent'e dustugunde internet fallback
        kararini vermek icin kullanilir.
        """
        words = self.tokenize(user_input)
        if not words:
            return 0.0
        input_set = set(words)
        best = 0.0
        for tag in self.intent_tags:
            s = sum(self.keyword_weights.get(w, 0.0)
                    for w in (input_set & self.intent_kws.get(tag, set())))
            if s > best:
                best = s
        return best

    def get_response(self, user_input):
        """Generate response for user input"""
        if self.model is None:
            return "Model not trained yet! Please run train.py first."

        chosen_tag, _, unclear = self._classify(user_input)
        if unclear:
            return "Anlayamadim, baska sekilde soyler misin?"

        words = self.tokenize(user_input)

        responses = self.intents.get(chosen_tag, ["Bir hata olustu."])

        if len(responses) == 1:
            return responses[0]

        # Cevabı sec: giris kelimeleriyle (normalizasyonlu) en cok oyusan
        input_words = set(words)
        best_responses = []
        best_score = -1
        for resp in responses:
            resp_words = set(self.tokenize(resp))
            score = len(input_words & resp_words)
            resp_flat = self.ascii_normalize(resp).lower()
            for w in input_words:
                if len(w) >= 3 and w in resp_flat:
                    score += 0.5
            if score > best_score:
                best_score = score
                best_responses = [resp]
            elif score == best_score:
                best_responses.append(resp)

        if best_score <= 0:
            return random.choice(responses)

        return random.choice(best_responses)

    def save_model(self, model_dir):
        """Save model and bot data"""
        os.makedirs(model_dir, exist_ok=True)

        self.model.save(os.path.join(model_dir, 'model.json'))

        bot_data = {
            'vocabulary': self.vocabulary,
            'intent_tags': self.intent_tags,
            'intents': self.intents,
            'intent_kws': {tag: sorted(list(kws)) for tag, kws in self.intent_kws.items()}
        }
        with open(os.path.join(model_dir, 'bot_data.json'), 'w', encoding='utf-8') as f:
            json.dump(bot_data, f, ensure_ascii=False, indent=2)

        print(f"Bot data saved: {model_dir}")

    def load_model(self, model_dir):
        """Load model and bot data"""
        self.model = NeuralNetwork([1])  # Temporary size
        self.model.load(os.path.join(model_dir, 'model.json'))

        with open(os.path.join(model_dir, 'bot_data.json'), 'r', encoding='utf-8') as f:
            bot_data = json.load(f)

        self.vocabulary = bot_data['vocabulary']
        self.intent_tags = bot_data['intent_tags']
        self.intents = bot_data['intents']
        self.intent_kws = {tag: set(kws) for tag, kws in bot_data.get('intent_kws', {}).items()}
        if not self.intent_kws:
            for tag in self.intent_tags:
                self.intent_kws[tag] = set()
        self._build_keyword_weights()

        print(f"Bot data loaded: {model_dir}")
        print(f"  Vocabulary size: {len(self.vocabulary)}")
        print(f"  Intent count: {len(self.intent_tags)}")
