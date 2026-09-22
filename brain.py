"""
Nextgen AI - Neural Network from Scratch
Built using only numpy, no TensorFlow or PyTorch.
"""

import numpy as np
import io
import json
import math
import random
import string
import os
import re

from normalize import ascii_normalize as _normalize
from bpe import turkish_lower as _tr_lower
from transformer import TransformerNN


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
    'yoktur', 'nefret', 'olmaz', 'yapma', 'etme', 'gitme',
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
    r"[.,!?;:•]\s*|\b(?:ve|veya|ya da|yoksa|ama|fakat|ancak|lakin|"
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
        # None = henuz yoklanmadi; ilk get_response'da seq_model.json var mi diye
        # bakilir, varsa True olur (latency: sadece bir kez import/load).
        self.seq_enabled = None
        # Seq2Seq encoder-decoder ureteci (model/seq2seq_model.json). None ise
        # dosya yok demektir; seq2seq.sample -> quality gate -> LSTM fallback.
        self.seq2 = None
        # Decoder-only LLM ureteci (model/llm_model.json). llm.sample ->
        # kalite kapisi -> seq2seq -> LSTM fallbacki. None ise yuklenmedi ya da
        # dosya yok; llm_enabled dosya yoklugunu bir kez tespit edip biter.
        self.llm = None
        self.llm_enabled = None
        # Bilgi (retrieval) hatti: knowledge_map desen->parca eslesmesi ve
        # corpus fallback'i lazy yuklenir; LLM'e verilecek bilgi parcasini
        # zenginlestirmektedir (yoksa canned bilgi yaniti yeter).
        self._kb_map = None
        # Uretilen ASCII model ciktisini gercek Turkce imlaya ceviren sozluk
        # (lazy: ilk deasciify cagrisinda canned yanitlardan kurulur).
        self._deascii_lex = None
        # Transformer girdisi icin token -> indeks eslemesi
        self.vocab_to_idx = {}
        self.pad_idx = 0
        self.max_seq_len = 24
        # Iki katmanli mimari esikleri:
        #  confidence_threshold : sohbet sinif seciminin guven esigi. Altinda
        #    kalan secimler once bilgi intelletimine (knowledge retrieval)
        #    sorulur; bilgi bulunamazsa kullaniciya yeniden sorulur.
        #  knowledge_threshold  : bilgi intent'ine ait anahtar kelime dikkatinin
        #    ayirt edici sayilmasi icin gereken min IDF agirlikli skor.
        self.confidence_threshold = 0.40
        self.knowledge_threshold = 1.5
        # LLM bilgi koullandirmasinda 'konu cekimi' (knowledge_bias): bilgi
        # parcasinda gecen icerik kelimelerinin logit bonusu -> uretim bilgiye
        # daha cok dokunur (baslangicta guclu, sona dogru sonecek sekilde).
        # 0.0 = kapali (eski davranis).
        self.knowledge_bias = 1.2

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
        return _normalize(text)

    def _build_deascii_lex(self):
        """canned yanitlardan ASCII(key) -> Turkce(value) sozlugu kurar.

        Uretici modeller (LLM/seq2seq/seqgen) ASCII uzayinda konustugundan
        'nasilsin' gibi ciktigi gercek Turkce imla takip etmez; kayitli
        yanitlardaki dogru imlayi ezberleyip ters yonde esleriz. Key
        ascii_normalize(lower), value ise `turkish_lower` ile dogru kucuk
        harfli Turkce sozcuktur (İ/I/ı ayrimi korunur).
        """
        lex = {}
        acronym = set()
        for resps in (self.intents or {}).values():
            for r in resps:
                for w in re.findall(r"[^\W_]+", r):
                    key = self.ascii_normalize(w).lower()
                    if not key:
                        continue
                    if len(w) >= 2 and w.isupper():
                        acronym.add(key)    # kisaltma (AI, NATO) -> imla degismez
                        continue
                    low = _tr_lower(w)
                    if key != low and key not in lex:
                        if len(key) < 2:
                            continue        # tek harf eslemeleri guvenilmez
                        lex[key] = low
        for key in acronym:
            lex.pop(key, None)
        self._deascii_lex = lex

    def deasciify(self, text):
        """ASCII model ciktisini sozlukten gercek Turkce imlaya cevirir.

        Bilinmeyen sozcukler oldugu gibi kalir (ozel isim 'Nextgen' bozulmaz);
        buyuk harfle baslayan sozcuk hedefli imlada da buyuk harfle baslar.
        Noktalama/ayraclar korunur. Sözlük yoksa ilk cagrida kurulur (lazy).
        """
        if not text:
            return text
        if self._deascii_lex is None:
            self._build_deascii_lex()
        out = []
        for token in re.split(r'(\W+)', text):
            if not token or not token.isalpha():
                out.append(token)
                continue
            mapped = (self._deascii_lex or {}).get(self.ascii_normalize(token).lower())
            if mapped is None:
                out.append(token)
                continue
            if token[0].isupper():
                mapped = mapped[0].upper() + mapped[1:]
            out.append(mapped)
        return ''.join(out)

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

    def text_to_indices(self, text, max_seq_len=None):
        """Metni transformer girdisine (vasatılmis token indeks dizisi) cevirir.

        Tokenler vocab_to_idx üzerinden indekslenir; taninmayanlar atlanir;
        dizi max_seq_len'a PAD ile hizalanir (PAD = vocab boyutu).
        """
        if max_seq_len is None:
            max_seq_len = self.max_seq_len
        seq = []
        for w in self.tokenize(text):
            if w in self.vocab_to_idx:
                seq.append(self.vocab_to_idx[w])
            if len(seq) >= max_seq_len:
                break
        if len(seq) < max_seq_len:
            seq = seq + [self.pad_idx] * (max_seq_len - len(seq))
        return seq

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
        self.vocab_to_idx = {w: i for i, w in enumerate(self.vocabulary)}
        self.pad_idx = len(self.vocabulary)

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

    def conversational_data(self, data):
        """Veriden sohbet intent'lerini ayirir ve intent_tags'i bunlara indirir.

        Bilgi intent'leri Wikipedia sablonundan uretilmis 6 desenli
        ("{konu} nedir", "{konu} hakkinda bilgi" gibi); sohbet intent'leri ise
        daha zengin (8-34 desen). Bu yuzden desen sayisi >6 olanlar sohbet
        intentidir, geri kalanlar bilgi intenti olarak taranmaya devam eder.

        Kucuk veri kumeleri (test/itibari dosyalar) filtreden tamamen
        elenirse orijinal veri aynen dondurulur (bozulma yok).
        """
        conv = [it for it in data['intents'] if len(it['patterns']) > 6]
        if not conv:
            return data
        self.intent_tags = sorted(it['tag'] for it in conv)
        return {'intents': conv}

    @property
    def knowledge_intents(self):
        """Sohbet siniflarina (intent_tags) dahil olmayan bilgi intentleri.

        Anahtar kelime retrieval'i bu kume uzerinde calisir; yeni bot_data
        'intent_tags' (sohbet siniflari, sayi veriden gudumludur) + 'intents'
        (tam kume) tutar, boylece bilgi kumesi fark olarak turetilir (sema
        degismez).
        """
        excl = set(self.intent_tags)
        return {t: r for t, r in self.intents.items() if t not in excl}

    def _build_keyword_weights(self):
        """Anahtar kelime dikkati icin IDF agirliklarini hesaplar.

        Seyrek gecen kelime yuksek agirlik (ayirt edici), her intent'te
        gecen ortak kelimeler dusuk agirlik alir.
        """
        n_intents = len(self.intent_kws) or 1
        doc_count = {}
        for tag in self.intent_kws:
            for stem in self.intent_kws.get(tag, ()):
                doc_count[stem] = doc_count.get(stem, 0) + 1
        weights = {}
        for stem, count in doc_count.items():
            w = 0.0 if stem in STOPWORDS else math.log(n_intents / (1.0 + count))
            weights[stem] = w
        self.keyword_weights = weights

        # Ters dizin (stem -> {tag: agirlik}) : sorgu basina intent taramak
        # yerine giris kelimeleri kadar is yapilur. O(791 intent) yerine
        # O(kelime sayisi x intent-basina-ortak-kelime) -> etkilesim icin yeterli.
        inv = {}
        for tag in self.intent_kws:
            for stem in self.intent_kws[tag]:
                w = weights.get(stem, 0.0)
                if w > 0.0:
                    inv.setdefault(stem, {})[tag] = w
        self._inv_kws = inv

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
        inv = getattr(self, '_inv_kws', None)
        if not inv:
            self._build_keyword_weights()
            inv = self._inv_kws
        attn = {}
        for w in input_set:
            for tag, wgt in inv.get(w, {}).items():
                attn[tag] = attn.get(tag, 0.0) + wgt
        return attn

    def _tag_rank(self, tag):
        """Tie-break sirasi: sohbet intent'leri kendi sirasinda, bilgi
        intent'leri en sonda (sohbet eslesmesi oncelikli olsun).
        """
        try:
            return self.intent_tags.index(tag)
        except ValueError:
            return len(self.intent_tags)

    def prepare_training_data(self, intents_data):
        """Eğitim verilerini hazırlar.

        Transformer mimarisi icin girdi, PAD-hizalanmis token indeks
        dizileridir (BoW degil). Donus: (X_seq, y) ; y = intent indeksleri.
        """
        X = []
        y = []
        max_len = 0

        for intent in intents_data['intents']:
            tag = intent['tag']
            tag_index = self.intent_tags.index(tag)

            for pattern in intent['patterns']:
                seq = self.text_to_indices(pattern)
                X.append(seq)
                y.append(tag_index)
                ln = len(self.tokenize(pattern))
                if ln > max_len:
                    max_len = ln

        self.max_seq_len = max(1, min(32, max_len))
        # Sekanslari yeni max_seq_len'a hizala (kisa olanlar PAD ile dolar)
        X = [seq[:self.max_seq_len] + [self.pad_idx] *
             max(0, self.max_seq_len - len(seq)) for seq in X]
        return np.array(X), np.array(y)

    def train_model(self, intents_filepath, epochs=500, learning_rate=0.001,
                    val_ratio=0.1, conversational_only=True):
        """Transformeri egitir (tok-girdi, multi-head self-attention).

        Egitim setinin son %val_ratio'luk kismi sabit seed ile DOGRULAMA
        setine ayrilir; erken durdurma ve raporlanan dogruluk val setinde
        olculur (egitim-seti dogrulugu ezberi yansitir, yanilticidir).

        conversational_only=True iken yalnizca sohbet intent'leri (deseni >6)
        siniflandirciya ogretilir; bilgi intent'leri (791'lik gercek veride
        753 adet, Wikipedia sablonlu) anahtar kelime retrieval ile cevaplanir.
        """
        print("Loading intents file...")
        data = self.load_intents(intents_filepath)

        if conversational_only:
            data = self.conversational_data(data)
            print(f"Conversational-only: {len(self.intent_tags)} sohbet intent'i "
                  f"+ {len(self.knowledge_intents)} bilgi intent'i (retrieval).")

        print("Preparing training data...")
        X, y = self.prepare_training_data(data)

        # Sabit tohumla tekrarlanabilir train/val ayrimi
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(X))
        n_val = max(1, int(len(X) * val_ratio))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        X_val, y_val = X[val_idx], y[val_idx]
        X, y = X[train_idx], y[train_idx]

        input_size = len(self.vocabulary)
        output_size = len(self.intent_tags)

        # Mimari: decoder-only LLM'e kiyasla kucuk ama classifier icin guclu.
        # Colab/Lightning AI egitimi ile ayni degerler kullanilmali (parity).
        arch = dict(d_model=128, num_blocks=4, num_heads=4, ff_mult=4)

        print(f"\nTransformer Architecture:")
        print(f"  Vocab size:   {input_size}")
        print(f"  Max seq len:  {self.max_seq_len}")
        print(f"  Embed dim:    {arch['d_model']} "
              f"({arch['num_heads']} kafa, {arch['num_blocks']} blok)")
        print(f"  Output:       {output_size} intents")
        print(f"\nTraining samples: {len(X)} (val: {len(X_val)})")
        print(f"Optimizer: AdamW | GELU | Pre-LN | Dropout 0.10 | L2: 1e-4")
        print()

        self.model = TransformerNN(
            vocab_size=input_size,
            num_intents=output_size,
            max_seq_len=self.max_seq_len,
            d_model=arch['d_model'], num_blocks=arch['num_blocks'],
            num_heads=arch['num_heads'], ff_mult=arch['ff_mult'],
            dropout=0.1, attn_dropout=0.05, weight_decay=1e-4,
            seed=42)

        print("Training started...")
        losses = self.model.train(X, y, epochs=epochs, learning_rate=learning_rate,
                                  batch_size=32, early_stop=True, patience=30,
                                  X_val=X_val, y_val=y_val)

        train_accuracy = self.model.evaluate(X, y)
        val_accuracy = self.model.evaluate(X_val, y_val)
        print(f"\nTraining completed!")
        print(f"  Egitim dogrulugu: {train_accuracy:.2%}")
        print(f"  Dogrulama dogrulugu: {val_accuracy:.2%}")

        return losses

    def get_probability(self, user_input):
        """User input icin en guclu intent tahminini ve olasiligini dondurur."""
        if self.model is None or not self.vocabulary:
            return None, 0.0
        X = np.array([self.text_to_indices(user_input)])
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
        X = np.array([self.text_to_indices(user_input)])

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
                seg_top = max(seg_attn, key=lambda t: (seg_attn[t], self._tag_rank(t)))
                if seg_attn[seg_top] > best_any:
                    best_any = seg_attn[seg_top]
                    attn = seg_attn
                    resp_words = seg_words

        chosen_tag = best_tag
        if attn:
            top_attn_tag = max(attn, key=lambda t: (attn[t], self._tag_rank(t)))
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
        inv = getattr(self, '_inv_kws', None)
        if not inv:
            self._build_keyword_weights()
            inv = self._inv_kws
        tag_scores = {}
        for w in input_set:
            for tag, wgt in inv.get(w, {}).items():
                tag_scores[tag] = tag_scores.get(tag, 0.0) + wgt
        return max(tag_scores.values()) if tag_scores else 0.0

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
        if strength < 0.5 and probability < 0.95:
            # Soru hicbir intent ile anlamli kelime paylasmiyor: secim guvenilmez.
            # Cok yuksek guven (>=0.95, ornek 'ne yapabilirsin') stem
            # boslugunu asar; yoksa fallback'e birak.
            return False

        if probability < 0.30 and strength < 1.5:
            return False

        # Kural 2b: Soru icinde modelin HICBIR intentinde gecmeyen bilgilendirici
        # bir kelime varsa (ozel isim/terim gibi: paris, akropol) ve secim zayif
        # bir eslesmeye dayaniyorsa dataset cevabi suphelidir -> fallback.
        if self.has_unknown_subject(user_input) and strength < 1.5 and probability < 0.95:
            return False

        query_kws = {w for w in self.tokenize(user_input) if w not in STOPWORDS}
        if not query_kws:
            return probability >= 0.30 or strength >= 1.5
        overlap = query_kws & self.intent_kws.get(tag, set())
        if len(overlap) == 1:
            # Tek stem eslesmesi cok kisa ise (bak/don gibi) konular arasi
            # karisim olabilir; yuuksek guven bile olsa fallback'e birak.
            (only,) = overlap
            if len(only) <= 3:
                return False
        if not overlap:
            return probability >= 0.95 or strength >= 1.5

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

    def _is_knowledge_question(self, text):
        """Tanim/olgu sorusu mu? ("X nedir", "X ne demek", "X hakkinda bilgi",
        "X kimdir", "X nerede"...) Bilgi intent'leri bu sablonlarla uretilir;
        siniflandirici chat tag'ine kaptirdiginda bile bilgi oncelik kazanir."""
        t = self.ascii_normalize(text).lower()
        return any(m in t for m in (
            'nedir', 'ne demek', 'ne demektir', 'hakkinda bilgi', 'hakkinda bil',
            'kimdir', 'kimlerdir', 'nerede', 'neredir', 'ne zaman', 'anlami nedir',
            'acilimi', 'tarihcesi', 'tarihi nedir', 'konusu nedir'))


    def _negation_reply(self, content):
        """Olumsuz icerikli girisler icin semptatik (destekleyici) yanit uretir."""
        topic = ""
        if content:
            topic = sorted(content, key=len, reverse=True)[0]
        if not topic:
            return random.choice(NEGATION_SUPPORT)
        return random.choice(NEGATION_SUPPORT_TOPIC).format(topic=topic)

    def _select_response(self, tag, resp_words):
        """Bir intent'in yanitlari arasindan girisle en cok ortusenini secer.

        Stop-word'ler ('yap','miy' gibi) skoru sulandirmasin; yoksa
        'stres icin yuruYUS YAP' 'Kilo... diyet' ile esit puana ulasir ve
        rastgele secim yanlis yanit verir. SeqGen cagrisi arayan taraftadir.
        """
        responses = self.intents.get(tag, ["Bir hata olustu."])

        if len(responses) == 1:
            return responses[0]

        input_words = {w for w in resp_words if w not in STOPWORDS and len(w) >= 3}
        best_responses = []
        best_score = -1
        for resp in responses:
            resp_set = set(self.tokenize(resp))
            score = len(input_words & resp_set)
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

    def _select_knowledge(self, words, exclude=None):
        """Bilgi intent'leri arasinda en guclu IDF eslesmesini arar.

        Sohbet siniflandiricisi bilgi sorularina (wayne rooney kimdir) guven
        vermez; bu metot bilgi intent'lerinin anahtar kelime dikkatini tarar
        ve yeterince ayirt edici (> knowledge_threshold) bir eslesme varsa
        o bilgi intent'inin en iyi yanitini dondurur, yoksa None.
        Ters dizin kullanildigi icin maliyet giris kelimesi kadardir.
        """
        if not self.knowledge_intents:
            return None
        input_set = set(words)
        if exclude:
            input_set = input_set - set(exclude)
        inv = getattr(self, '_inv_kws', None)
        if not inv:
            self._build_keyword_weights()
            inv = self._inv_kws
        kb = self.knowledge_intents
        scores = {}
        for w in input_set:
            for tag, wgt in inv.get(w, {}).items():
                if tag in kb:
                    scores[tag] = scores.get(tag, 0.0) + wgt
        if not scores:
            return None
        top_tag = max(scores, key=lambda t: (scores[t], self._tag_rank(t)))
        if scores[top_tag] < self.knowledge_threshold:
            return None
        return self._select_response(top_tag, words)

    def get_response(self, user_input):
        """Generate response for user input"""
        if self.model is None:
            return "Model not trained yet! Please run train.py first."

        negated, neg_content = self.detect_negation(user_input)

        chosen_tag, probability, unclear, resp_words = self._classify(
            user_input, negated_content=neg_content if negated else None)

        # OLUMSUZ ICERIK: "Futbol sevmiyorum" -> onaylamaci sport yaniti yerine
        # destekleyici yanit verilmeli.
        if negated and neg_content:
            return self._negation_reply(neg_content)

        # BILGI ONCELIGI: tanim/olgu sorusu ("X nedir", "X hakkinda bilgi", "X
        # kimdir"...) ise once bilgi retrieval denenir. Sınıflandırıcı bilgi
        # sorusunu chat tag'ine kaptirdiginda bile ("galaksi nedir" -> teknoloji)
        # ansiklopedik yanit ezilmeden doner. Retrieval bos donerse normal akis
        # (sohbet siniflandirmasi) devam eder.
        if self._is_knowledge_question(user_input):
            kbt = self._select_knowledge(
                resp_words, exclude=neg_content if negated else None)
            if kbt:
                return self._try_kb_rephrase(user_input, kbt)

        # ANAHTAR KELIME OVERRIDE bilgi intentine ulasti: canned bilgi yaniti
        # (LLM varsa bilgi parcasindan yeniden kurulur -> kopya degil).
        if chosen_tag not in self.intent_tags:
            kb = self._select_response(chosen_tag, resp_words)
            return self._try_kb_rephrase(user_input, kb)

        # GUVENSIZ SECIM: sohbet siniflarina guvenilmiyorsa once bilgi
        # intentlerine sor; eslesme varsa bilgi yaniti (LLM ile yeniden
        # kurulur), yoksa kullanicidan netlestirme iste.
        if unclear or probability < self.confidence_threshold:
            kb = self._select_knowledge(
                resp_words, exclude=neg_content if negated else None)
            if kb:
                return self._try_kb_rephrase(user_input, kb)
            if unclear:
                return "Anlayamadim, baska sekilde soyler misin?"

        best = self._select_response(chosen_tag, resp_words)

        # Uretici hatti: LLM -> Seq2Seq -> SeqGen LSTM. Kalite kapisi
        # gecmezse ekranda eski yanit (canned) doner (guvenli fallback).
        gen = self._try_seq_rephrase(chosen_tag, user_input)
        if gen:
            return gen

        return best

    def _try_seq_rephrase(self, tag, query=None):
        """Once Decoder-only LLM, sonra Seq2Seq encoder-decoder (char), en
        sonunda SeqGen LSTM ile taze bir varyant uretir.

        Ornekleme kosulu: asil query (varsa), yoksa tag. Seklendirme modelleri
        bu kosula egitilmis olmalidir (colab notebook'lari sorgu-kosullu egitiyor).
        Kalite sapagi: cok kisa/tekrariest/sozcuk dagina dokunmayan ya da
        KAYITLI YANITIN KOPYASI olan ciktilari reddeder -> akil yurutme: model
        canned'i ezberlemek yerine konuya yapisik OZGUN cumle kurabilir. Model
        dosyalari yoksa da None (sessiz devre disi). Hangi uretec olursa olsun
        ayni kapidan gecer: llm -> seq2seq -> seqgen fallbacki.
        """
        try:
            ctx = query or tag
            if self._ensure_llm():
                gen = self._best_of_llm(ctx, tag)
                if gen:
                    return self.deasciify(gen)
            if self.seq2 is None:
                from seq2seq import load_seq2seq
                self.seq2 = load_seq2seq()
            if self.seq2 is not None:
                gen = self.seq2.sample(ctx, temperature=0.6, top_k=6)
                if self._accept_generated(gen, tag, query=ctx):
                    return self.deasciify(gen)
            if self.seq_enabled is None:
                self.seq_enabled = False
                from seqgen import load_seq
                self.seq = load_seq()
                self.seq_enabled = self.seq is not None
            if not self.seq_enabled:
                return None
            gen = self.seq.sample(ctx, temperature=0.7, top_k=10)
            if self._accept_generated(gen, tag, query=ctx):
                return self.deasciify(gen)
            return None
        except Exception:
            return None

    def _best_of_llm(self, query, tag, tries=3):
        """LLM ile best-of-N sohbet/yanit adayi uretir.

        Ilk-token entropisi yanlis baslangica kaydiginda tek deneme usually
        tutmaz; birden fazla numune alinir, kalite kapisindan gecen adaylardan
        en uzun/anlamli olani secilir. Gecen yoksa None (ust katman seq2seq/
        seqgen/canned fallbackina duser).
        """
        best, best_len = None, 0
        for _ in range(max(1, int(tries))):
            gen = self.llm.sample(query, temperature=0.6, top_k=6,
                                  rep_penalty=0.4)
            if not self._accept_generated(gen, tag, query=query):
                continue
            if best is None or len(gen) > best_len:
                best, best_len = gen, len(gen)
        return best

    def _ensure_llm(self):
        """LLM'i bir kez yukler (dosya yoksa kalici olarak devre disi)."""
        if self.llm_enabled is None:
            self.llm_enabled = False
            try:
                from llm import load_llm
                self.llm = load_llm()
                self.llm_enabled = self.llm is not None
            except Exception:
                self.llm = None
        return self.llm_enabled and self.llm is not None

    def _external_knowledge(self, query, tag=None):
        """Bilgi sorgusu icin corpus'tan en alakali parcayi dondurur.

        Sira: (1) knowledge_map.jsonl desen->parca eslesmesi (deterministik,
        egitimle ayni harita; once `tag`'in desenleri, sonra ham sorgu),
        (2) canli corpus.search fallback. Ikisi de yoksa None (canned bilgi
        yanitina duser). Corpus, egitimdekiyle ayni kaynak oldugu icin LLM
        koullandirmasi train/val ile tutarli olur.
        """
        try:
            if self._kb_map is None:
                from seqgen import clean_chars
                kmap = {}
                p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'knowledge_map.jsonl')
                if os.path.exists(p):
                    with io.open(p, encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            row = json.loads(line)
                            if row.get('ctx') and row.get('text'):
                                kmap[row['ctx']] = row['text']
                self._kb_map = kmap or {}
                # dosya anahtarlari Turkce imlali uzayda; LLM context'i ise
                # ASCII+kesik (clean_chars). Kullanicinin ASCII yazisini da
                # bilgiye baglamak icin ayni uzayda ikinci lut olustur.
                self._kb_ascii = {clean_chars(c, 64): t
                                  for c, t in (kmap or {}).items()}
            if self._kb_map:
                qn = query.strip().lower()
                # 1a) hedef bilgi intent'inin desenleriyle esles
                if qn in self._kb_map:
                    return self._kb_map[qn]
                # 1a') ASCII/kesim uzayinda esles ("nasilsin" ~ "nasılsın")
                aq = clean_chars(qn, 64)
                if aq and aq in self._kb_ascii:
                    return self._kb_ascii[aq]
                # 1b) desenlerin normallesmis haliyle kesis (cok desenli tutarli)
                best, bs = None, 0.0
                qset = set(qn.split()) - {'nedir', 'kimdir', 'kactir', 'nerede',
                                          'hakkinda', 'bilgi', 'ver', 'anlat',
                                          'ne', 'demek', 'bana', 'mi', 'mu'}
                for ctx, text in self._kb_map.items():
                    cset = set(ctx.split()) - {'nedir', 'kimdir', 'kactir',
                                               'nerede', 'hakkinda', 'bilgi',
                                               'ver', 'anlat', 'ne', 'demek',
                                               'bana', 'mi', 'mu'}
                    score = len(qset & cset)
                    if score > bs:
                        bs, best = score, text
                if bs >= 2:
                    return best
        except Exception:
            pass
        return None

    def _try_kb_rephrase(self, query, kb, tries=3):
        """Bilgi (retrieval) yanitini ozetler/yeniden kurar: kopyala-yapistir
        yerine bilgi parcasindan yola cikip kisa, ozgun bir aciklama uretir.

        LLM yoksa ya da ozet kalite kapisindan gecmezse ham `kb` doner
        (guvenli fallback). _accept_generated ile ayni mantik ama konu kumesi
        bilgi metninin kendisidir (tag yoktur).

        `tries`: best-of-N ornekleme. Ilk token entropisi yanlis basa
        kactiginda (tek denemede tutmama) birden fazla aday uretilir, kalite
        kapisindan gecen ilk/EN_OTORITATIF aday secilir; gecen yoksa kb.
        """
        if not query or not kb:
            return kb
        if not self._ensure_llm():
            return kb
        ext = None
        try:
            ext = self._external_knowledge(query)
        except Exception:
            ext = None
        try:
            knowledge = (ext + '\n' + kb) if ext else kb
            best, best_len = None, 0
            for _ in range(max(1, int(tries))):
                gen = self.llm.sample(query, temperature=0.7, top_k=10,
                                      knowledge=knowledge[:500],
                                      rep_penalty=0.4,
                                      knowledge_bias=self.knowledge_bias)
                if not self._accept_kb_rephrase(gen, kb):
                    continue
                if best is None or len(gen) > best_len:
                    best, best_len = gen, len(gen)
            if best:
                return self.deasciify(best)
        except Exception:
            pass
        return kb

    def _accept_generated(self, gen, tag, query=None):
        """Uretilen varyanti kalite kapisindan gecirir (ozgunluk odakli).

        Rule-check:
          - Kisa/tekrar/sozcuk dagi denetimi (eski),
          - konu orusu: token'larin en az %25'i intent sozcuk dagine dokunmali,
          - OZGUNLUK: token'larin en az %15'i kayitli yanitlarda YOK olmali
            (kopya reddedilir). Kayitli kanonlar sadece boyle asilir -> akil
            yurutme/yeni cumle kurmaya alan acilir.
          - yapisan tekrar (her yeni token ayni) elenir.
          - SORGU KONUSU: query 3+ icerik kelimesi tasiyorsa uretim o kelimelerden
            en az birini gecmeli. Siniflandirici yanlis intent'e dustuğünde bile
            ("bugun kararsizim pizza mi yesem..." -> hava_durumu) model konu-disi
            metne ziplamasin; bu sart saglanmazsa aday reddedilir (guvenli canned).
            Kisa/duygusal sorgular (2 icerik kelimesi) kapisiz kalir: empatik
            yanitlar sorudaki kelimeleri tekrar etmek zorunda degildir.
        """
        if not gen or len(gen) < 12 or len(gen) > 260:
            return False
        letters = [c for c in gen.lower() if c.isalpha()]
        if len(letters) < 6 or len(set(letters)) < int(len(letters) * 0.30):
            return False
        canned = set()
        for r in (self.intents.get(tag) or []):
            canned |= set(self.tokenize(r))
        kws = self.intent_kws.get(tag) or set()
        cand = self.tokenize(gen)
        if len(cand) < 3:
            return False
        gen_set = set(cand)
        known = canned | kws
        inter = gen_set & known
        # konu orusu: en az 2 TANIDIK sozcuk (tek "tatli icin" ile konudan
        # sapan kisa karmasik uretim kalite kapisini gecmesin).
        if len(inter) < 2:
            return False
        overlap = len(inter) / float(len(gen_set))
        if overlap < 0.25:
            return False
        # SORGU KONUSU: 3+ icerik kelimeli soruda uretim sorudaki konuya
        # dokunmali. Siniflandirici kaymasi durumunda bile konu-disi aday
        # reddedilir ve guvenli canned fallback'e dusulur.
        if query is not None:
            qkws = {t for t in self.tokenize(query) if t not in STOPWORDS}
            if len(qkws) >= 3 and not (gen_set & qkws):
                return False
        # ozgunluk: canned disinda en az %15 yeni sozcuk (kopyaya hayir)
        novel = gen_set - canned
        if len(novel) / float(len(gen_set)) < 0.15:
            return False
        # yapiskan tekrar: ayni sozcugun yan yana tekrari dominat olmamali
        if len(cand) >= 4:
            dup = sum(1 for a, b in zip(cand, cand[1:]) if a == b)
            if dup / float(len(cand) - 1) > 0.5:
                return False
        return True

    def _accept_kb_rephrase(self, gen, kb):
        """Bilgi yeniden-kurumu icin ozel kapida: konu bilgi parcasinda,
        ozgunluk %15, kisa ve akici. Başarisizsa None -> ham kb doner."""
        if not gen or len(gen) < 12 or len(gen) > 260:
            return False
        letters = [c for c in gen.lower() if c.isalpha()]
        if len(letters) < 6 or len(set(letters)) < int(len(letters) * 0.30):
            return False
        kb_set = set(self.tokenize(kb))
        if not kb_set:
            return False
        cand = self.tokenize(gen)
        if len(cand) < 3:
            return False
        gen_set = set(cand)
        kb_inter = gen_set & kb_set
        # konu kumesi: en az 2 tanidik kelime (tek kelimelik kesisle ilgisiz
        # uretim kalite kapisini gecmesin).
        if len(kb_inter) < 2:
            return False
        overlap = len(kb_inter) / float(len(gen_set))
        if overlap < 0.20:
            return False
        novel = gen_set - kb_set
        if len(novel) / float(len(gen_set)) < 0.15:
            return False
        if len(cand) >= 4:
            dup = sum(1 for a, b in zip(cand, cand[1:]) if a == b)
            if dup / float(len(cand) - 1) > 0.5:
                return False
        return True

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
        """Load model and bot data (transformer / eski feedforward otomatik tepk)."""
        model_path = os.path.join(model_dir, 'model.json')
        with open(model_path, 'r', encoding='utf-8') as f:
            head = json.load(f)

        if head.get('arch') == 'transformer':
            self.model = TransformerNN(vocab_size=1, num_intents=1, max_seq_len=1)
            self.model.load(model_path)
            # LoRA adaptörü varsa taban ağırlıkların üzerine uygula (kolonları
            # yeni sözcük/sınıf eklemeden model.json'a dokunulmaz)
            lora_path = os.path.join(model_dir, 'lora.json')
            if os.path.exists(lora_path):
                with open(lora_path, 'r', encoding='utf-8') as f:
                    lora = json.load(f)
                self.model.apply_lora(lora)
                print(f"LoRA adaptor yuklendi: {lora_path} "
                      f"(+{lora.get('vocab_added', 0)} vocab, "
                      f"+{lora.get('head_added', 0)} intent)")
        else:
            self.model = NeuralNetwork([1])  # Eski feedforward uyumlulugu
            self.model.load(model_path)

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

        # Transformer girdi eslemesi (farkli dille yeniden kurulur)
        self.vocab_to_idx = {w: i for i, w in enumerate(self.vocabulary)}
        self.pad_idx = len(self.vocabulary)
        if hasattr(self.model, 'max_seq_len') and self.model.max_seq_len:
            self.max_seq_len = self.model.max_seq_len

        print(f"Bot data loaded: {model_dir}")
        print(f"  Vocabulary size: {len(self.vocabulary)}")
        print(f"  Intent count: {len(self.intent_tags)}")
        print(f"  Knowledge intents (retrieval): {len(self.knowledge_intents)}")
