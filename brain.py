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
import re
import unicodedata


# Latin (Turkce dahil) ozel harflerin ASCII karsiliklari. Aksanli harflerin
# cogu NFD decompose ile cikar, ama ayristirilamayan harfler (eszett, ligatur
# vb.) dogrudan eslenir. Boylesi 'jacques prevert' / 'jacques prévert' gibi
# yabanci adlari da ayni torbaya dusurur (autogrow onlari ham ekliyor).
_LATIN_TO_ASCII = {
    'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
    'â': 'a', 'î': 'i', 'û': 'u', 'i': 'i', 'o': 'o', 'u': 'u',
    'Ç': 'c', 'Ğ': 'g', 'İ': 'i', 'I': 'i', 'Ö': 'o', 'Ş': 's', 'Ü': 'u',
    'Â': 'a', 'Î': 'i', 'Û': 'u',
    'ß': 'ss', 'æ': 'ae', 'Æ': 'AE', 'œ': 'oe', 'Œ': 'OE',
    'ð': 'd', 'Ð': 'D', 'ø': 'o', 'Ø': 'O', 'ł': 'l', 'Ł': 'L',
    'þ': 'th', 'Þ': 'TH',
    '\u0307': '',  # Python'in 'İ'.lower() ciktisindaki kombinasyon noktasi
}
_LATIN_TRANSLATE = str.maketrans(_LATIN_TO_ASCII)


# Turkce islev/durak kelimeleri: anahtar kelime dikkatinde agirligi sifir.
# (Cumlede konuyu tasimazlar; "hangi, ne, mi" gibi her yerde gecerler.)
STOPWORDS = {
    'mi', 'mu', 'miyim', 'misin', 'misiniz', 'msin', 'msiniz',
    'sen', 'ben', 'bana', 'sana', 'bu', 'su', 'neyi', 'hangisi',
    'ne', 'hangi', 'kac', 'nasil', 'neden', 'nicin', 'bir', 'da', 'de',
    'ya', 'ki', 'miydi', 'midir', 'neydi', 'var', 'yok',
    # baglac/dolgu kelimeleri: konu tasimaz; 'bana baska bir mizah yap'
    # gibi uzun isteklerde eylem niyeti tanim/ansiklopedi intent'ine kaymasin.
    'baska', 'tane', 'daha',
    # soru/template parcalari: konu tasimazlar, IDF onlari yanlis guclendirmesin
    'zam', 'kurult', 'hakk', 'bilk', 'ver', 'bilg', 'anlat', 'soyle',
    'kal', 'olur', 'olabilir', 'edebil', 'eder', 'onerr', 'oner',
    # koklenmis soru/kilavuz kelimeler: nerede->nere, hangi->hank gibi.
    # Bunlarin yuksek IDF'si yanlis intent secimini guclendirir.
    'nere', 'neres', 'hank', 'hangis', 'kim', 'kimt', 'ney', 'nered',
    'kimi', 'kimin', 'neyi', 'nic', 'ned', 'neye', 'nerde', 'nerdey',
    # yardimci/edilgen fiil kokleri: konu tasimaz, OOV gibi davranmasin,
    # yapilir->yapil, edilir->edil, yapmali->yap, miyim->miy gibi.
    'edil', 'yapil', 'yapilir', 'yap', 'miy',
    # gorus/modal belirtecleri: tek basina konu tasimaz, corpus kisa-yolunu
    # yanlis tetikleyip ansiklopedi cevabi cekmesin ('sence yapmali miyim'
    # -> PSG makalesi gibi).
    'sence', 'bence',
    # yapisal/tumce baglac ve edatlar: konu tasimaz, IDF onlari guclendirmesin
    'ile', 'icin', 'gibi', 'kadar', 'cok', 'hic', 'biraz', 'az', 'en',
    'once', 'sonra', 'zaman', 'simdi', 'bugun', 'yarin', 'dun',
    'acaba', 'sanki', 'peki', 'ama', 've', 'ancak', 'cunku', 'ayrica', 'belki',
    'biz', 'siz', 'onlar', 'onun', 'bunun', 'sunun', 'birsey', 'sey',
    'miyiz', 'miyuz', 'miyim', 'muyuz', 'muyum', 'musun', 'bende',
}

# OLUMSUZLUK ALGILAMA sabitleri
# Sozcukler (olumsuz isaret): degil, yok, hayir, asla, hic(?), nefret vs.
NEGATION_SIGNALS = {
    'degil', 'degilim', 'değil', 'hayir', 'hayır', 'asla', 'hic', 'yok',
    'yoktur', 'nefret', 'nefret', 'olmaz', 'yapma', 'etme', 'gitme',
    'gelme', 'isteme', 'sevme', 'verme', 'alma', 'durma', 'konusma',
    'soyleme', 'calisma', 'izleme',
}

# raw kelime sonlarina gore olumsuzluk (egitim/ilgi eki):
# sev-m-i-yorum, yap-m-a-yacak, git-me-di-m vb.
NEGATION_SUFFIXES = (
    'meyecegim', 'mayacagim', 'miyordum', 'miyorsunuz', 'miyorsun',
    'miyorum', 'miyorlar', 'miyor', 'miyiz', 'miyim',
    'meyeceksin', 'mayacaksin', 'meyecek', 'mayacak',
    'madim', 'medim', 'madin', 'medin', 'mam', 'mem',
    'mazsin', 'mazsun', 'maz', 'mez',
    'mayayim', 'meyeyim', 'masan', 'mesin',
)

# Olumsuz icerikli secime destek yanitlari (semptatik yaklasim)
NEGATION_SUPPORT = [
    "Anlıyorum, senin için bu durum kolay olmamış. Bana ne ters gittiğini anlatmak istersen buradayım.",
    "Hiç sorun değil. Birlikte sana iyi gelecek bir konu bulabiliriz, ne dersin?",
    "Anlaman uzun sürmedi, 'hayır' demek de bir tercihtir. Peki senin isteğin ne?",
    "Tamam, buna saygı duyuyorum. Başka hangi konuda yardımcı olabilirim?",
]

# Belirli bir konuyu (topic) iceren olumsuzluk destek yanitlari
NEGATION_SUPPORT_TOPIC = [
    "Anlıyorum, {topic} senin için pek uygun değil gibi. Peki hangi konudan konuşmak istersin?",
    "{topic} hakkında farklı bir şey soruyorsun sanırım. Neyi kastediyorsun, biraz açar mısın?",
    "{topic} sana hitap etmiyor gibi; istersen başka bir konuya geçelim.",
]

# Cumleyi parcalara ayirmak icin: noktalama ve baglaclar.
# Boylece coklu anahtar kelime iceren cumlelerde her parcaya ayri bakilir.
SEGMENT_PATTERN = re.compile(
    r"[.,!?;:•]\s*|\b(?:ve|veya|ya da|yoksa|ama|fakat|ancak|lakin|fakat|"
    r"sonra|ardindan|bundan sonra)\b",
    re.IGNORECASE)


class NeuralNetwork:
    """
    Basit bir feedforward sinir ağı.
    Sadece numpy ile sıfırdan yazılmıştır.
    """

    def __init__(self, layer_sizes, dropout=0.2, use_batchnorm=True,
                 weight_decay=1e-4, seed=None, use_attention=True,
                 attn_hidden=24):
        """
        Sinir ağını başlatır.

        Args:
            layer_sizes: Her katmandaki nöron sayısını içeren liste
                         Örn: [input_size, hidden1, hidden2, output_size]
            dropout: Gizli katmanlardaki dropout orani (0.2 = %20 at).
            use_batchnorm: Gizli katmanlara batch normalization uygulansin mi?
            weight_decay: L2 duzenlilestirme (weight decay) katsayisi.
            seed: Tekrarlanabilirlik icin rastgelelik tohumu.
            use_attention: Girdiye bagli ogrenilebilir dikkat kapisi (gate).
                          Geleneksel sabit IDF agirliklarinin yerine, hangi
                          kelimelerin ONEMLI oldugunu model kendisi ogerenir.
            attn_hidden: Dikkat kapisi icin gizli birim sayisi.
        """
        if seed is not None:
            np.random.seed(seed)

        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.num_layers = len(layer_sizes)

        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.weight_decay = weight_decay
        self.max_grad_norm = 5.0
        self.training = True
        self.use_attention = use_attention
        self.attn_hidden = attn_hidden

        # Her katman için ağırlıkları ve bias'ları başlat (He initialization)
        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        # OGRENILEBILIR DIKKAT KAPISI: gate = sigmoid(ReLU(X*W1+b1)*W2+b2)
        # Kelime BASINA onem kapilari: gate boyutu (m, vocab). Baslangicta
        # gate ~0.82 (b2>0) -> istenirse model kelimeleri kendisi artirir/azaltir.
        vocab = layer_sizes[0]
        self.attn_w1 = np.random.randn(vocab, attn_hidden) * np.sqrt(2.0 / vocab)
        self.attn_b1 = np.zeros((1, attn_hidden))
        self.attn_w2 = np.random.randn(attn_hidden, vocab) * np.sqrt(2.0 / attn_hidden)
        self.attn_b2 = np.ones((1, vocab)) * 1.5

        # Adam optimizer durumlari (mutlak anlar)
        self._m_w = [np.zeros_like(w) for w in self.weights]
        self._v_w = [np.zeros_like(w) for w in self.weights]
        self._m_b = [np.zeros_like(b) for b in self.biases]
        self._v_b = [np.zeros_like(b) for b in self.biases]
        self._m_attn = {'w1': np.zeros_like(self.attn_w1), 'b1': np.zeros_like(self.attn_b1),
                        'w2': np.zeros_like(self.attn_w2), 'b2': np.zeros_like(self.attn_b2)}
        self._v_attn = {'w1': np.zeros_like(self.attn_w1), 'b1': np.zeros_like(self.attn_b1),
                        'w2': np.zeros_like(self.attn_w2), 'b2': np.zeros_like(self.attn_b2)}
        self._t = 0

        # Batch normalization durumlari (yalnizca gizli katmanlar)
        self.bn_gamma = []
        self.bn_beta = []
        self.bn_mean = []
        self.bn_var = []
        for i in range(max(0, self.num_layers - 2)):
            d = self.layer_sizes[i + 1]
            self.bn_gamma.append(np.ones((1, d)))
            self.bn_beta.append(np.zeros((1, d)))
            self.bn_mean.append(np.zeros(d))
            self.bn_var.append(np.ones(d))
        self.bn_grads = [{'gamma': None, 'beta': None} for _ in self.bn_gamma]

        # Ileri/geri propagasyon gecicileri
        self.reset_buffers()

    def _attention_gate(self, X):
        """Girdiye bagli onem kapisini hesaplar (Bahdanau-benzeri dikkat)."""
        z1 = np.dot(X, self.attn_w1) + self.attn_b1
        h1 = self.relu(z1)
        z2 = np.dot(h1, self.attn_w2) + self.attn_b2
        gate = 1.0 / (1.0 + np.exp(-np.clip(z2, -50, 50)))
        self._attn_z1 = z1
        self._attn_h1 = h1
        self._attn_z2 = z2
        self._attn_gate = gate
        return gate

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

    def reset_buffers(self):
        """Ileri gecis tamponlarini sifirlar."""
        self.z_list = []
        self.a_list = [None]
        self.dropout_masks = []
        self.bn_xhat = []
        self.bn_std = []

    def forward(self, X, apply_dropout=None):
        """
        İleri yayılım (forward propagation).

        Args:
            X: Giriş verisi (batch_size, input_size)
            apply_dropout: None ise self.training durumuna gore karar verir.

        Returns:
            Tahmin sonucu (batch_size, output_size)
        """
        if apply_dropout is None:
            apply_dropout = self.training

        # OGRENILEBILIR DIKKAT: girdi kelime torbalarini onem derecesine gore
        # agirliklandir (onemli soru kelimelerinden gez, dolgu kelimeleri kapati).
        x_attn = X
        self._attn_X = X
        if self.use_attention:
            gate = self._attention_gate(X)
            x_attn = X * gate

        self.z_list = []  # Her katmanın lineer çıkışı
        self.a_list = [x_attn]  # Her katmanın aktivasyonu
        self.dropout_masks = []
        self.bn_xhat = []
        self.bn_std = []

        current_input = x_attn

        for i in range(self.num_layers - 1):
            # Lineer dönüşüm: z = X * W + b
            z = np.dot(current_input, self.weights[i]) + self.biases[i]

            is_hidden = i < self.num_layers - 2

            # Batch normalization (yalnızca gizli katmanlar)
            if is_hidden and self.use_batchnorm:
                if apply_dropout:  # Eğitim modu -> batch istatistikleri
                    batch_mean = z.mean(axis=0, keepdims=True)
                    batch_var = z.var(axis=0, keepdims=True)
                    std = np.sqrt(batch_var + 1e-8)
                    xhat = (z - batch_mean) / std
                    ema = 0.9
                    self.bn_mean[i] = ema * self.bn_mean[i] + (1 - ema) * batch_mean.ravel()
                    self.bn_var[i] = ema * self.bn_var[i] + (1 - ema) * batch_var.ravel()
                else:  # Tahmin modu -> biriken istatistikler
                    std = np.sqrt(self.bn_var[i] + 1e-8)[None, :]
                    xhat = (z - self.bn_mean[i]) / std
                z = self.bn_gamma[i] * xhat + self.bn_beta[i]
                self.bn_xhat.append(xhat)
                self.bn_std.append(std)
            else:
                self.bn_xhat.append(None)
                self.bn_std.append(None)

            self.z_list.append(z)

            # Aktivasyon fonksiyonu
            activation = None
            if is_hidden:  # Gizli katmanlar için ReLU + dropout
                activation = self.relu(z)
                if self.dropout > 0 and apply_dropout:
                    keep = 1.0 - self.dropout
                    mask = (np.random.rand(*activation.shape) < keep) / keep
                    activation = activation * mask
                    self.dropout_masks.append(mask)
                else:
                    self.dropout_masks.append(None)
            else:  # Çıkış katmanı için Softmax
                activation = self.softmax(z)

            self.a_list.append(activation)
            current_input = activation

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
        Geri yayılım (backpropagation) ve Adam ile ağırlık güncelleme.

        Args:
            y_true: Gerçek etiketler (dizin dizisi veya one-hot)
            learning_rate: Öğrenme hızı
        """
        m = y_true.shape[0]
        self._t += 1
        t = self._t

        deltas = [None] * (self.num_layers - 1)

        # Çıkış katmanı için hata (softmax + cross-entropy birleşik gradyan)
        output = self.a_list[-1]
        if y_true.ndim > 1:  # one-hot ise indekse çevir
            y_index = np.argmax(y_true, axis=1)
        else:
            y_index = y_true
        delta_last = output.copy()
        delta_last[np.arange(m), y_index] -= 1
        deltas[-1] = delta_last / m

        # Gizli katmanlar için geriye doğru hata hesapla
        for i in range(self.num_layers - 3, -1, -1):
            delta = np.dot(deltas[i + 1], self.weights[i + 1].T)
            if self.dropout_masks[i] is not None:
                delta = delta * self.dropout_masks[i]
            delta = delta * self.relu_derivative(self.z_list[i])

            # Batch normalization zincir kuralı
            if self.use_batchnorm and self.bn_xhat[i] is not None:
                xhat = self.bn_xhat[i]
                std = self.bn_std[i]
                gamma = self.bn_gamma[i]

                dxhat = delta * gamma
                self.bn_grads[i]['gamma'] = np.sum(delta * xhat, axis=0, keepdims=True)
                self.bn_grads[i]['beta'] = np.sum(delta, axis=0, keepdims=True)

                dx = (1.0 / (m * std)) * (
                    m * dxhat
                    - np.sum(dxhat, axis=0, keepdims=True)
                    - xhat * np.sum(dxhat * xhat, axis=0, keepdims=True))
                delta = dx

            deltas[i] = delta

        # Gradyanları hesapla ve norm clipping uygula
        grad_w = [np.dot(self.a_list[i].T, deltas[i]) for i in range(self.num_layers - 1)]
        grad_b = [np.sum(deltas[i], axis=0, keepdims=True) for i in range(self.num_layers - 1)]

        # DIKKAT KAPISI gradyanlari: x_attn = X * gate; dL/dx_attn = deltas[0]*W0.T
        attn_grads = {}
        if self.use_attention:
            grad_input = np.dot(deltas[0], self.weights[0].T)  # (m, vocab)
            dg = grad_input * self._attn_X                      # dL/dgate
            gate = self._attn_gate
            dz2 = dg * gate * (1.0 - gate)  # sigmoid türevi, (m, vocab) tireci
            attn_grads['w2'] = np.dot(self._attn_h1.T, dz2)
            attn_grads['b2'] = np.sum(dz2, axis=0, keepdims=True)
            dh1 = np.dot(dz2, self.attn_w2.T) * (self._attn_z1 > 0)
            attn_grads['w1'] = np.dot(self._attn_X.T, dh1)
            attn_grads['b1'] = np.sum(dh1, axis=0, keepdims=True)

        total_norm = np.sqrt(sum(np.sum(g * g) for g in grad_w)
                             + sum(np.sum(g * g) for g in grad_b)
                             + sum(np.sum(g * g) for g in attn_grads.values()))
        scale = 1.0
        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / total_norm

        # Adam güncellemesi (momentum + RMSProp birleşimi)
        b1, b2, eps = 0.9, 0.999, 1e-8
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t

        for i in range(self.num_layers - 1):
            gW = grad_w[i] * scale
            gB = grad_b[i] * scale
            if self.weight_decay > 0:
                gW += self.weight_decay * self.weights[i]

            self._m_w[i] = b1 * self._m_w[i] + (1 - b1) * gW
            self._v_w[i] = b2 * self._v_w[i] + (1 - b2) * (gW * gW)
            m_hat = self._m_w[i] / bc1
            v_hat = self._v_w[i] / bc2
            self.weights[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            self._m_b[i] = b1 * self._m_b[i] + (1 - b1) * gB
            self._v_b[i] = b2 * self._v_b[i] + (1 - b2) * (gB * gB)
            m_hat_b = self._m_b[i] / bc1
            v_hat_b = self._v_b[i] / bc2
            self.biases[i] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + eps)

        # Dikkat kapisi parametrelerini Adam ile guncelle
        if self.use_attention and attn_grads:
            attr_map = {'w1': 'attn_w1', 'b1': 'attn_b1', 'w2': 'attn_w2', 'b2': 'attn_b2'}
            for key in ('w1', 'b1', 'w2', 'b2'):
                g = attn_grads[key] * scale
                self._m_attn[key] = b1 * self._m_attn[key] + (1 - b1) * g
                self._v_attn[key] = b2 * self._v_attn[key] + (1 - b2) * (g * g)
                m_hat = self._m_attn[key] / bc1
                v_hat = self._v_attn[key] / bc2
                param = getattr(self, attr_map[key])
                param -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        # Batch normalization parametrelerini güncelle (gamma, beta)
        if self.use_batchnorm:
            for i in range(len(self.bn_grads)):
                g = self.bn_grads[i]
                if g['gamma'] is not None:
                    self.bn_gamma[i] -= learning_rate * g['gamma'] * scale
                    self.bn_beta[i] -= learning_rate * g['beta'] * scale

    def _init_optimizer(self):
        """Adam moment durumlarını sıfırlar (yeni eğitim için)."""
        self._m_w = [np.zeros_like(w) for w in self.weights]
        self._v_w = [np.zeros_like(w) for w in self.weights]
        self._m_b = [np.zeros_like(b) for b in self.biases]
        self._v_b = [np.zeros_like(b) for b in self.biases]
        if self.use_attention:
            self._m_attn = {'w1': np.zeros_like(self.attn_w1), 'b1': np.zeros_like(self.attn_b1),
                            'w2': np.zeros_like(self.attn_w2), 'b2': np.zeros_like(self.attn_b2)}
            self._v_attn = {'w1': np.zeros_like(self.attn_w1), 'b1': np.zeros_like(self.attn_b1),
                            'w2': np.zeros_like(self.attn_w2), 'b2': np.zeros_like(self.attn_b2)}
        self._t = 0

    def train(self, X, y, epochs=1000, learning_rate=0.01, batch_size=16,
              verbose=True, lr_min_ratio=0.1, early_stop=False, patience=20):
        """
        Modeli eğitir.

        Args:
            X: Eğitim verileri
            y: Etiketler
            epochs: Eğitim döngüsü sayısı
            learning_rate: Başlangıç öğrenme hızı
            batch_size: Mini-batch boyutu
            verbose: Eğitim bilgisi yazdırılsın mı?
            lr_min_ratio: Cosinüs çizelgesinin alt sınır oranı (örn. 0.1 -> son LR = base*0.1)
            early_stop: Erken durdurma aktif mi?
            patience: Erken durdurmada sabır (epoch cinsinden)
        """
        losses = []
        m = X.shape[0]
        base_lr = learning_rate
        best_acc = 0.0
        best_state = None
        wait = 0
        self.training = True

        for epoch in range(epochs):
            # Cosinüs öğrenme hızı çizelgesi
            frac = epoch / max(epochs - 1, 1)
            lr = base_lr * (lr_min_ratio + (1 - lr_min_ratio)
                            * (0.5 * (1 + np.cos(np.pi * frac))))

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

                # İleri yayılım (dropout açık)
                output = self.forward(X_batch, apply_dropout=True)

                # Kayıp hesapla
                loss = self.compute_loss(output, y_batch)
                epoch_loss += loss
                num_batches += 1

                # Geri yayılım ve Adam güncellemesi
                self.backward(y_batch, lr)

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

            if early_stop:
                accuracy = self.evaluate(X, y)
                if accuracy > best_acc + 1e-4:
                    best_acc = accuracy
                    best_state = self.get_state()
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose:
                            print(f"Erken durdurma: epoch {epoch + 1}, en iyi accuracy {best_acc:.2%}")
                        if best_state is not None:
                            self.set_state(best_state)
                        break

            if verbose and (epoch + 1) % 50 == 0:
                accuracy = self.evaluate(X, y)
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2%}")

        return losses

    def predict(self, X):
        """Tahmin yapar (deterministik mod)"""
        self.training = False
        try:
            output = self.forward(X)
        finally:
            self.training = True
        return np.argmax(output, axis=1)

    def predict_proba(self, X):
        """Olasılık tahminleri yapar (deterministik mod)"""
        self.training = False
        try:
            output = self.forward(X)
        finally:
            self.training = True
        return output

    def evaluate(self, X, y):
        """Doğruluk oranını hesaplar"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_state(self):
        """Eğitilebilir parametrelerin kopyasını döndürür (erken durdurma için)."""
        state = {
            'weights': [w.copy() for w in self.weights],
            'biases': [b.copy() for b in self.biases],
            'bn_gamma': [g.copy() for g in self.bn_gamma],
            'bn_beta': [b.copy() for b in self.bn_beta],
            'bn_mean': [m.copy() for m in self.bn_mean],
            'bn_var': [v.copy() for v in self.bn_var],
        }
        if self.use_attention:
            state['attn'] = {
                'w1': self.attn_w1.copy(), 'b1': self.attn_b1.copy(),
                'w2': self.attn_w2.copy(), 'b2': self.attn_b2.copy(),
            }
        return state

    def set_state(self, state):
        """get_state ile kopyalanan durumu geri yükler."""
        self.weights = [w.copy() for w in state['weights']]
        self.biases = [b.copy() for b in state['biases']]
        if state.get('bn_gamma'):
            self.bn_gamma = [g.copy() for g in state['bn_gamma']]
            self.bn_beta = [b.copy() for b in state['bn_beta']]
        if state.get('bn_mean'):
            self.bn_mean = [m.copy() for m in state['bn_mean']]
            self.bn_var = [v.copy() for v in state['bn_var']]
        if state.get('attn') and getattr(self, 'use_attention', False):
            self.attn_w1 = state['attn']['w1'].copy()
            self.attn_b1 = state['attn']['b1'].copy()
            self.attn_w2 = state['attn']['w2'].copy()
            self.attn_b2 = state['attn']['b2'].copy()

    def save(self, filepath):
        """Modeli kaydeder"""
        model_data = {
            'layer_sizes': self.layer_sizes,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'dropout': self.dropout,
            'use_batchnorm': self.use_batchnorm,
            'weight_decay': self.weight_decay,
            'use_attention': self.use_attention,
            'attn_w1': self.attn_w1.tolist(),
            'attn_b1': self.attn_b1.tolist(),
            'attn_w2': self.attn_w2.tolist(),
            'attn_b2': self.attn_b2.tolist(),
            'bn_gamma': [g.tolist() for g in self.bn_gamma],
            'bn_beta': [b.tolist() for b in self.bn_beta],
            'bn_mean': [m.tolist() for m in self.bn_mean],
            'bn_var': [v.tolist() for v in self.bn_var],
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
        print(f"Model saved: {filepath}")

    def load(self, filepath):
        """Modeli yükler (geriye dönük uyumlu)"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        self.layer_sizes = model_data['layer_sizes']
        self.weights = [np.array(w) for w in model_data['weights']]
        self.biases = [np.array(b) for b in model_data['biases']]
        self.num_layers = len(self.layer_sizes)

        # Yeni hiperparametreler; eski modellerde varsayılanlara düş
        self.dropout = model_data.get('dropout', 0.0)
        self.use_batchnorm = model_data.get('use_batchnorm', False) and 'bn_gamma' in model_data
        self.weight_decay = model_data.get('weight_decay', 0.0)
        self.use_attention = model_data.get('use_attention', False) and 'attn_w1' in model_data
        self.attn_hidden = self.layer_sizes[0] if not model_data.get('attn_w1') else \
            np.array(model_data['attn_w1']).shape[1]
        self.max_grad_norm = 5.0

        self.bn_gamma = []
        self.bn_beta = []
        self.bn_mean = []
        self.bn_var = []
        self.bn_grads = []
        if self.use_batchnorm:
            self.bn_gamma = [np.array(g) for g in model_data['bn_gamma']]
            self.bn_beta = [np.array(b) for b in model_data['bn_beta']]
            self.bn_mean = [np.array(m) for m in model_data['bn_mean']]
            self.bn_var = [np.array(v) for v in model_data['bn_var']]
            self.bn_grads = [{'gamma': None, 'beta': None} for _ in self.bn_gamma]

        if self.use_attention:
            self.attn_w1 = np.array(model_data['attn_w1'])
            self.attn_b1 = np.array(model_data['attn_b1'])
            self.attn_w2 = np.array(model_data['attn_w2'])
            self.attn_b2 = np.array(model_data['attn_b2'])
        else:
            self.attn_w1 = np.zeros((self.layer_sizes[0], 1))
            self.attn_b1 = np.zeros((1, 1))
            self.attn_w2 = np.ones((1, 1))
            self.attn_b2 = np.zeros((1, 1))

        # Adam momentleri GERCEK parametre boyutlarindan sonra baslatilmali
        self._init_optimizer()

        self.training = True
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
        self._neg_suffixes = NEGATION_SUFFIXES
        self.seq = None  # SeqGen LSTM ureteci (model dosyasi varsa lazy yuklenir)
        self.seq_enabled = False  # uretici varyantlari OP-IN (latency icin)

    def is_negation_word(self, word, stem=None):
        """Tek bir sozcugun olumsuzluk tasiyip tasimadigini dondurur.

        Sozcuk kumesi (degil/yok/hayir...), olumsuz fiil ekleri
        (miyorum/mayacak/maz...) ve emir-negatifleri (yapma/gitme) kapsanir.
        """
        if word in NEGATION_SIGNALS:
            return True
        if stem and stem in NEGATION_SIGNALS:
            return True
        if len(word) >= 5:
            for suffix in self._neg_suffixes:
                if word.endswith(suffix):
                    return True
        return False

    def detect_negation(self, text):
        """Cumlede olumsuzluk olup olmadigini ve konu (content) koklerini dondurur.

        Returns:
            (olumsuz_mu: bool, konu_kokleri: set)
        """
        norm = self.ascii_normalize(text.lower())
        content = set()
        for w in norm.split():
            if len(w) < 3:
                continue
            stem = self.simple_stem(w)
            if self.is_negation_word(w, stem):
                continue
            if stem not in STOPWORDS and len(stem) >= 4:
                content.add(stem)
        negated = any(self.is_negation_word(w, self.simple_stem(w))
                      for w in norm.split() if len(w) >= 3)
        return negated, content

    def ascii_normalize(self, text):
        """
        Turkce ozel karakterleri ASCII karsiliklarina cevirir.
        Boylece 'ogle' / 'öğle' / 'OĞLE' hepsi ayni kelimeye donusur.
        Ek olarak yabanci aksanlar da sokulur: 'prevert' / 'prévert' ayni olur.
        """
        t = text.translate(_LATIN_TRANSLATE)
        if any(ord(c) > 127 for c in t):
            t = ''.join(c for c in unicodedata.normalize('NFD', t)
                        if not unicodedata.combining(c))
        return t

    def simple_stem(self, word):
        """Basit Turkce kelime koku bulma (stemming)"""
        if word in self.stem_cache:
            return self.stem_cache[word]

        original = word
        stem = word

        # onceki tek-adim yaklasimi yerine, ekleri tekrarli sekilde temizle
        # Not: 'miy' olumsuzluk infiksi korunur; 'sevmiyim' -> 'sevmiy'
        # (yine de "sev"le ayni torbaya dusmez) ve 'edil/yapil' korunur.
        suffixes = [
            'misiniz', 'miyim', 'musunuz', 'misin', 'lerimiz', 'larimiz',
            'leriniz', 'lariniz', 'lerine', 'larina', 'lerin', 'larin',
            'lari', 'leri', 'lere', 'lara', 'lar', 'ler',
            'den', 'dan', 'ten', 'tan', 'de', 'da', 'te', 'ta',
            'nin', 'nun', 'in', 'un',
            'mis', 'mus', 'im', 'um', 'izin', 'uz',
            'sin', 'sun', 'i', 'u',
            'ecek', 'acak', 'erek', 'arak', 'mel', 'mal', 'mek', 'mak',
            'iyor', 'uyor', 'yor', 'ir', 'ar', 'er', 'en', 'an',
            'ye', 'ya', 'yi', 'yle', 'yla', 'ken',
        ]

        while True:
            removed = False
            for suffix in suffixes:
                if len(stem) - len(suffix) >= 3 and stem.endswith(suffix):
                    # Olumsuzluk infiksi ('miy') ile baslayan ekleri atma:
                    # 'sevmiyim' -> 'sevmiy' kalmalı, 'sev' olmamali.
                    if 'miy' in stem and suffix.startswith('mi'):
                        continue
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
        """Metni kelimelere ayırır ve temizler.

        Tekil kelimelerin yaninda "yapay zeka", "kara delik" gibi anlamli
        2'li kelime gruplarini (bigram) da tek birim olarak dondurur.
        Her iki parcasi durak kelime olan gruplar elenir: "hakkinda bilgi"
        konu tasimaz, "yapay zeka" konu tasir.
        """
        # noktalama işaretlerini kaldır
        text = text.lower()
        text = self.ascii_normalize(text)
        for char in string.punctuation:
            text = text.replace(char, '')
        words = text.split()
        stems = [self.simple_stem(w) for w in words]
        result = list(stems)
        for i in range(len(stems) - 1):
            a, b = stems[i], stems[i + 1]
            if a and b and a not in STOPWORDS and b not in STOPWORDS:
                if len(a) >= 4 and len(b) >= 4:
                    result.append(a + '_' + b)
        return result

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

    def split_segments(self, text):
        """Cumleyi noktalama ve baglaclarla parcalara boler.

        Coklu anahtar kelime iceren cumlelerde her parcanin kendi
        anahtar kelimelerine ayri ayri dikkat edilebilmesi icin kullanilir.
        """
        parts = [p.strip() for p in SEGMENT_PATTERN.split(text) if p.strip()]
        return parts or [text]

    def _attn_of(self, words, exclude=None):
        """Kelime (stem) kumesi icin intent basina IDF agirlikli dikkat skoru.

        exclude: olumsuzluk senaryosunda konu kokleri atlanir; boylece
        "futbol sevmiyorum" spor intent'ini guclendirmez.
        """
        input_set = set(words)
        if exclude:
            input_set = input_set - set(exclude)
        attn = {}
        for tag in self.intent_tags:
            inter = input_set & self.intent_kws.get(tag, set())
            s = sum(self.keyword_weights.get(w, 0.0) for w in inter)
            if s > 0:
                attn[tag] = s
        return attn

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
        hidden1 = min(128, max(48, input_size // 2))
        hidden2 = min(64, max(24, input_size // 4))
        hidden3 = min(48, max(12, output_size * 2))

        print(f"\nModel Architecture:")
        print(f"  Input layer:  {input_size} neurons (vocabulary size)")
        print(f"  Hidden1:      {hidden1} neurons")
        print(f"  Hidden2:      {hidden2} neurons")
        print(f"  Hidden3:      {hidden3} neurons")
        print(f"  Output layer: {output_size} neurons (intent count)")
        total = (input_size*hidden1 + hidden1 + hidden1*hidden2 + hidden2
                 + hidden2*hidden3 + hidden3 + hidden3*output_size + output_size)
        print(f"  Total params: {total}")
        print(f"\nTraining samples: {len(X)}")
        print(f"Total words: {input_size}")
        print(f"Total intents: {output_size}")
        print(f"Optimizer: Adam | Dropout: 0.2 | BatchNorm: açık | L2: 1e-4")
        print()

        self.model = NeuralNetwork(
            [input_size, hidden1, hidden2, hidden3, output_size],
            dropout=0.2, use_batchnorm=True, weight_decay=1e-4, seed=42,
            use_attention=True, attn_hidden=24)

        print("Training started...")
        losses = self.model.train(X, y, epochs=epochs, learning_rate=learning_rate,
                                  batch_size=32, early_stop=True, patience=30)

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

    def _classify(self, user_input, negated_content=None):
        """Input icin secilen intent etiketini, guveni ve belirsizlik bayragini dondurur.

        Keyword override dahil get_response ile birebir ayni secim mantigini kullanir.
        negated_content: olumsuzluk senaryosunda konu kokleri; intent dikkatinden
        cikarilir ("futbol sevmiyorum" -> 'spor' onaylamaci secilmesin).
        """
        words = self.tokenize(user_input)
        bag = self.bag_of_words(words)
        X = np.array([bag])

        probabilities = self.model.predict_proba(X)[0]
        best_index = np.argmax(probabilities)
        best_probability = float(probabilities[best_index])
        best_tag = self.intent_tags[best_index]

        exclude = frozenset(negated_content or ())

        # ANAHTAR KELIME DIKKATI: IDF agirlikli onem skoru.
        # Seyrek/ayirt edici kelimeler (galaksi, gazneli, yardim) yuksek,
        # her yerde gecenler (hangi, ne, mi) sifira yakin agirliktadir.
        whole_attn = self._attn_of(words, exclude=exclude)
        cls_attn = sum(self.keyword_weights.get(w, 0.0)
                       for w in ((set(words) - exclude) & self.intent_kws.get(best_tag, set())))

        if not whole_attn and best_probability < 0.15:
            return 'Anlayamadim', best_probability, True, words

        # COKLU ANAHTAR KELIME: cumleyi parcalara bol, her parcanin kendi
        # dikkat skorunu hesapla; en guclu parcayi sec. Boylece
        # "çay öner ama bugün stresliyim" gibi cumlelerde son konu yakalanir.
        attn = whole_attn
        resp_words = words
        segments = self.split_segments(user_input)
        if len(segments) > 1:
            best_any = max(whole_attn.values(), default=0.0)
            for seg in segments:
                seg_words = self.tokenize(seg)
                if not seg_words:
                    continue
                seg_attn = self._attn_of(seg_words, exclude=exclude)
                if not seg_attn:
                    continue
                seg_top = max(seg_attn, key=lambda t: (seg_attn[t], self.intent_tags.index(t)))
                if seg_attn[seg_top] > best_any:
                    best_any = seg_attn[seg_top]
                    attn = seg_attn
                    resp_words = seg_words

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
        return chosen_tag, best_probability, False, resp_words

    def predict(self, user_input):
        """Disa aktarmadan tahmin kullanimi icin (Excel taramalari gibi).

        Returns:
            (etiket, guven_yuzdesi) ; anlayamazsa ('Anlayamadim', 0.0)
        """
        if self.model is None:
            return 'Model yok', 0.0
        tag, probability, unclear, _ = self._classify(user_input)
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

    def can_answer(self, user_input):
        """Veri kumesinden cevap vermeye guvenilir mi?

        Kural 1 (Katı Eşik): Softmax olasiligi dusukse secilen intent e
        guvenilmez; ancak guclu IDF eslesmesi (>=1.5) siniflandiriciyi
        asabilir (mevcut davranis).
        Kural 2 (Kelime Ortulusme): Sorunun anlamli kelimelerinden (stop-word
        haric) HICBIRI secilen intent kalibinda gecmiyorsa, secim tahmindir;
        donusturulmez.

        Returns:
            bool: True = dataset cevabini kullan, False = corpus/fallback.
        """
        tag, probability, unclear, _ = self._classify(user_input)
        if unclear:
            return False

        strength = self.keyword_strength(user_input)
        if strength < 0.5:
            # Soru hicbir intent ile anlamli kelime paylasmiyor: secim guvenilmez.
            return False

        if probability < 0.30 and strength < 1.5:
            return False

        # Kural 2b: Soru icinde modelin HICBIR intentinde gecmeyen bilgilendirici
        # bir kelime varsa (ozel isim/terim gibi: paris, akropol) ve secim zayif
        # bir eslesmeye dayaniyorsa dataset cevabi suphelidir -> fallback.
        if self.has_unknown_subject(user_input) and strength < 1.5:
            return False

        query_kws = {w for w in self.tokenize(user_input) if w not in STOPWORDS}
        if not query_kws:
            return probability >= 0.30 or strength >= 1.5
        overlap = query_kws & self.intent_kws.get(tag, set())
        if not overlap and strength < 1.5:
            # Secilen intent sorunun HICBIR anahtar kelimesini icermiyor.
            return False

        return True

    def has_unknown_subject(self, user_input):
        """Soruda modelin hicbir intent kalibinda gecmeyen bilgilendirici bir kelime
        var mi?

        Ozel isim ve terimler (paris, akropol, kardiyoloji) intent kaliblarinda
        olmadigi icin agirlik 0 olur; bunlar varken NN secimi tahmin olabilir,
        corpus/internet fallbak tercih edilir. Stopword kokleri zaten agirlik 0
        tutuldugundan otomatik elenir.
        """
        for w in self.tokenize(user_input):
            if len(w) < 4:
                continue
            if w in STOPWORDS:
                continue
            if '_' in w:
                # iki parcasi bilinen bigram yanlis alarm vermesin
                a, b = w.rsplit('_', 1)
                if self.keyword_weights.get(a, 0.0) > 0.0 or self.keyword_weights.get(b, 0.0) > 0.0:
                    continue
            if self.keyword_weights.get(w, 0.0) == 0.0:
                return True
        return False

    def _negation_reply(self, content):
        """Olumsuz icerikli girisler icin semptatik (destekleyici) yanit uretir."""
        topic = ""
        if content:
            topic = sorted(content, key=len, reverse=True)[0]
        if not topic:
            return random.choice(NEGATION_SUPPORT)
        return random.choice(NEGATION_SUPPORT_TOPIC).format(topic=topic)

    def get_response(self, user_input):
        """Generate response for user input"""
        if self.model is None:
            return "Model not trained yet! Please run train.py first."

        negated, neg_content = self.detect_negation(user_input)

        chosen_tag, _, unclear, resp_words = self._classify(
            user_input, negated_content=neg_content if negated else None)

        if unclear:
            if negated:
                return self._negation_reply(neg_content)
            return "Anlayamadim, baska sekilde soyler misin?"

        # OLUMSUZ ICERIK: "Futbol sevmiyorum" -> onaylamaci sport yaniti yerine
        # destekleyici yanit verilmeli.
        if negated and neg_content:
            return self._negation_reply(neg_content)

        responses = self.intents.get(chosen_tag, ["Bir hata olustu."])

        if len(responses) == 1:
            return responses[0]

        # Cevabi sec: en guclu parcanin kelimeleriyle (normalizasyonlu)
        # en cok oyusan yaniti sec. Stop-word'ler ('yap','miy' gibi) skoru
        # sulandirmasin; yoksa 'stres icin yuruYUS YAP' 'Kilo... diyet' ile
        # esit puana ulasir ve rastgele secim yanlis yanit verir.
        input_words = {w for w in resp_words if w not in STOPWORDS and len(w) >= 3}
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

        # SeqGen LSTM ureteciyle taze varyant (kalite kapisi gecmezse canned).
        gen = self._try_seq_rephrase(chosen_tag)
        if gen:
            return gen

        return random.choice(best_responses)

    def _try_seq_rephrase(self, tag):
        """SeqGen LSTM ureteciyle tag'e gore taze bir varyant uretir.

        Kalite sapagi: cok kisa/tekrariest/sozcuk dagina dokunmayan ciktilari
        reddeder (None dondurur) -> get_response guvenli sekilde canned'e donebilir.
        Model dosyasi yoksa da None (sessiz devre disi).
        """
        try:
            if not getattr(self, 'seq_enabled', False):
                return None
            if self.seq is None:
                from seqgen import load_seq
                self.seq = load_seq()
                if self.seq is None:
                    return None
            gen = self.seq.sample(tag, temperature=0.9, top_k=14)
            if not gen or len(gen) < 12 or len(gen) > 260:
                return None
            letters = [c for c in gen.lower() if c.isalpha()]
            if len(letters) < 6 or len(set(letters)) < int(len(letters) * 0.30):
                return None
            union = set()
            for r in (self.intents.get(tag) or []):
                union |= set(self.tokenize(r))
            kws = self.intent_kws.get(tag) or set()
            gen_toks = set(self.tokenize(gen))
            if not gen_toks:
                return None
            overlap = len(gen_toks & (union | kws))
            return gen if overlap / float(len(gen_toks)) >= 0.30 else None
        except Exception:
            return None

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
