# Olcum araclari

Bunlar karar araclari, uretim kodu degil. Hepsi **olcer**, hicbiri veri
dosyalarina yazmaz. `eval_llm.py` ve `train_llm.py` modelin kalitesini
olcer; buradakiler **uretim yolunun** ne yaptigini olcer.

Tum raporlar `olcum_raporlari/` altina yazilir (`.gitignore`'da).

## Iki sey ayri tutulur

`eval_llm.py` **capraz dogrulama** setiyle olcer: modelin egitimde hic
gormedigi sorular, altin cevapla karsilastirilir. Guclu ama yapay.

Buradaki araclar **uretim yolunu** olcer: gercek `app.load_bot()` +
`bot.get_response()`, gercek kalite kapisi, gercek best-of-3 ornekleme.

29.09'da ikisi zit sonuc verdi: eval yeni modeli alakada belirgin
iyi bulurken, uretim yolunda yeni model daha az metin geciriyordu.
Nedeni kapinin harf-cesitliligi hatasiydi; duzeltilince fark ortustu.
Ders: **uretim yolunu sadece eval ile olcmek yaniltir.**

## uretim_olc.py - uretim yolunu olcer (ana arac)

```
python tools/uretim_olc.py <etiket> [cikti.json]
```

50 soruyu **`soru_listesi.json`'dan** (sabit liste) okur, her soru icin
uretim denemelerini (best-of-3) kaydeder:

- **deneme kabul orani** - kapidan gecen uretim / toplam deneme
- **ekrana uretim gecen soru** - kullanicinin gercekten modelin cumlesini
  gordugu oran (kalanlar canned'a duser)
- **sadakat** - uretilen metnin icerik kelimelerinden bilgi parcasinda
  gercekten bulunanlarin orani (uydurma olcumu)
- **ret nedenleri** - hangi kapi sarti ihlal edildi, dagilimiyla

### `soru_listesi.json` - NEDEN SABIT (01.10.2026)

Sorular once `knowledge_map.jsonl`'den satir adimiyla (`i % 600`) seciliyordu.
Ama `knowledge_map.jsonl` CI'da (`kbmap.yml`) her gun **yeniden uretiliyor**:
satir sayisi degismese bile satir icerigi degisiyor, yani ayni adim **farkli
sorulari** seciyor. Olculdu: 29.09 raporu ile 01.10 raporunda **3/50 ortak
soru** vardi. Yani "ekrana cikan metin %52 -> %76" gibi **zaman serisi
iddialari gecersizdi** - 47/50 soru degismisti.

Simdi sorular `tools/soru_listesi.json`'dan okunur; dosya yoksa eski yolla
uretilip **sabitlenir** ve uyari basar. Dosya repodadir, veri dosyasi DEGILDIR
(olcum tanimidir) ve **elle degistirilmemelidir**: degistirilirse yeni rapor
eskisiyle kiyaslanamaz.

### Sadakat normalizasyonu duzeltildi (01.10.2026)

`icerik_kelimeler` once Turkce harfleri **siliyordu**:

```
"sarkinin"  -> ['sarkinin']    (ASCII metin bozulmadan)
"sarkinin"  -> ['ark']         (s ve noktali i SILINDI)
```

Model ciktisi ASCII, bilgi metni (`kb`) Turkce oldugu icin iki taraf **hic
eslesemiyordu**; sadakat yapay olarak dusuk olculuyordu. Artik projenin kanonik
katlamasi `normalize.ascii_normalize` kullaniliyor.

**Olculen etki** (01.10 modeli, ayni 50 soru, ayni tohum, ayni uretimler):
sadakat **%33,3 -> %57,1**.

> `uretim_karsilastir.py` rapor JSON'undaki `sadakat` alanini okudugu icin
> **01.10.2026 oncesi uretilmis raporlarla** calistirilirsa eski (bozuk)
> sayilari kullanir. Karsilastirmadan once `baza_0110.json` sonrasi uretilmis
> rapor kullan.

`np.random.seed(7)` her soruda tekrar kurulur ve `_external_knowledge`
yereldir (ag yok), yani **iki kosu birebir ayni** sonucu verir. Iki modeli
karsilastirmak icin bu belirleyicilik sarttir.

## uretim_karsilastir.py - iki uretim raporunu paired karsilastirir

```
python tools/uretim_karsilastir.py <A.json> <B.json>
```

Fark **A - B** yazilir (eval_llm.py ile ayni yon). Dort eksen: ekrana
uretim gecen soru (ikili + McNemar), sadakat (paired t), deneme basi
kabul, ret nedenleri farki.

## sohbet_olc.py - SOHBET yolunu olcer (taban olcumu, 02.10.2026)

```
python tools/sohbet_olc.py <etiket> [cikti.json]
```

`uretim_olc.py` **bilgi** yolunu olcer (`brain._try_kb_rephrase`, sadakat).
Sohbet tamamen farkli bir yoldur (`_classify` -> chat sinifi ->
`_try_seq_rephrase`), metrikleri de farklidir. **Sohbet hic olculmemisti**:
sabit 50 sorunun 50'si de bilgi sorusuydu, tek sohbet sorusu yoktu — yani
"sohbet kalitesi %72" denilen her sayi aslinda bilgi-only idi.

Sorular `soru_listesi_sohbet.json`'dan (sabit, **40/40 sohbet sinifinin
tamami** + 5 kenar soru). Olctukleri:

- **KULLANILABILIR ISABET** (ana olcu) - dogru tahmin **ve** o sinif
  gercekten var. Sadece tahmin dogruluğu degil: `brain.py:1746`
  `chosen_tag in self.intent_tags` kapisi acilmazsa sohbet yolu hic
  calismaz, dogru tahmin yine de ise yaramaz.
- **SINIF TASI** - beklenen etiket var ama siniflandirici sinifinda **degil**.
  Bu sorular yapisal olarak kazanilamaz; olcmek icin varlar.
- **ETIKET OLMAYAN TAHMIN** - `predict()` intents.json'da **olmayan** bir
  metin dondurdu (`ari hjelm`, `an giang` gibi). brain.py tarafindan uretilmis,
  olcum aracindan degil.
- canned orani (donen metin intents.json'daki hazir bir yanitin birebir kopyasi mi),
- tekrarli yanit, bos yanit, LLM yolunun acilma/donme orani.

> **Aracin kendi hatasi (02.10, duzeltildi):** "sinif tasi" filtresi once
> *tahmin edilen* etikete bakiyordu; 6 yanlis tahmini sinif tasi saydi.
> Dogru kriter **beklenen** etiketin sinifta olmasi.
>
> **Daha onemli: olcum kendini kirletiyordu.** Tek geciste her soru icin once
> `predict()` sonra `get_response()` cagriliyordu; bir onceki sorunun
> `get_response`'i sonraki sorunun `predict`'ini degistiriyordu. Dogrulama:
> `predict()` iki kez arka arkaya 45/45 ayni (deterministik), yani kirlilik
> yalnizca `get_response -> predict` yonunde. Cozum: **iki ayri gecis** -
> (1) siniflandirma, hicbir `get_response` cagrisi yapmadan; (2) uretim.
> Bu duzeltmeden sonra "etiket olmayan bozuk tahmin" iddiasi TAMAMEN DUSTU
> (45 sorunun 44'u gecerli etiket, tek istisna `Anlayamadim` sentinel'i).
> 01.10 tarihli ilk sohbet raporlarindaki sinif-tasi ve bozuk-etiket
> sayilari **gecersizdir**.

### `soru_listesi_sohbet.json` - ASCII-only, bilgi listesiyle ayni sozlesme

Turkce harf iceren soru **yok**: `normalize.py:16-19` etkisi ayri konudur,
ikisini ayni olcumde karistirmamak icin. Sorular desenlerin kopyasi degil
(kullanici bunu boyle soyleyebilir), isabet orani ezber degil genelleme olcer.

`kapsam_disi` grubundaki sorularda `kabul: "sohbet_sinifi_DEGIL"` vardir:
olcutulen sey etiket degil, **sohbet sinifina girmemesi**. Bilgi yoluna
yonlendirmek kabul sayilir - modelin ciktisina gore ayarlanmis bir esik
degil.

## kapi_ab.py - OZGUNLUK ESIGI taramasi (esik 0.15 -> 0.00)

Kapinin `ozgunluk >= %15` kuralini 8 esikte **paired** olarak tarar ve her
esikte ekrana cikan metnin sadakatini olcer. Ciktida "yeni kabul" = 0.15'te
reddedilip o esikte kabul edilen adaylar ve onlarin sadakati gorunur.

**Neden tek kosu yeterli:** uretim yolu 3 adayi her zaman uretir ve gecenler
arasindan en uzun olani secer (`brain.py:2033-2042`). Aday metinleri ve secim
kurali esikten bagimsiz oldugu icin **tek kosuda** kaydedip esik degistirerek
yeniden hesaplamak ayni sonucu verir (8 kosunun 1'i maliyetinde).

Kapinin kopyasi once **dogrulanir**: eldeki rapordaki 150 adayda asil
`brain._accept_kb_rephrase` karariyla **150/150 ayni** olmazsa tarama gecersiz
sayilir. (01.10'da dogrulandi.)

Onceki surumu harf-cesitliligi kuralini kiyasiyordu; o kural 25.09'da
matematiksel olarak imkansiz bulundu ve `min(12, ...)` ile duzeltildi
(`brain.py:2081-2098`), sorun kapandi.

## model_ab.py - "hangi modeli kuralim" SORUSUNU OLCMEYLE cevaplar

01.10.2026 23:26 egitimi val loss'u %4,4 iyilestirdi ama 50 sabit
sorumada **ekrana cikan metin %72 -> %52 dustu** (deterministik
olculdugu dogrulandi). Yani "metriglere bakin, model iyi" yaniltici.

```
python tools/model_ab.py model/nde_irma_0110_2326/llm_model.json yeni
python tools/model_ab.py <yeni.json> yeni --rapor olcum_raporlari/uretim_eski_0110c.json
```

Yaptigi:

1. **Determinizm kontrolu** - ayni modeli iki kez olcer, 5 alanin
   (`uretilildi`, `yanit`, `sadakat`, `kb`, `bos_kova`) 50/50 ayni
   oldugunu dogrular. Degilse sonuc gecersiz.
2. **Asil olcum** - aday modelle bir kez daha olcer.
3. **Veri degisti mi** - iki kosu arasinda uretimi etkileyen veri
   dosyalarinin (intents/knowledge_map/corpus/chatgrow_*) izini karsilastirir.
4. **Karar** - ekrana cikan metin sayisini karsilastirir, hangisinin
   kurulacagini yazar.

**Arac modeli KURMAZ** (yedekleme `model_kur.py`'nin isi). Amaci:
karar gozle degil olcumle verilsin.

### `uretim_olc.py` raporuna damga yazar
Her rapor artik `veri_damgasi`, `model_damgasi` ve
`soru_listesi_damgasi` tasir (dosya `boyut` + `mtime`). 01.10'da iki rapor
"paired" sanildi ama aralarinda saatler icinde `chatgrow_*.jsonl`
degismisti; artik rapor kendi kendini savunur.

## yedek_olc.py - kapi reddedince ekrana ne cikiyor?

Kapı 3 adayı reddedince `_try_kb_rephrase` **ham kb metnini** döndürüyor
(`brain.py:2047`). Bu metin çoğu zaman küçük harfle başlıyor, yarım
cümle olabiliyor ve doğallaştırma dolguları ("bir bakıma", "kısa ca",
"ayrıca") taşıyor.

**Kapının kuralına dokunmadan** yedekten kaçınmanın yollarını ölçer:
`tries` 3→6 ve `temperature` 0,7→0,9 (daha çok aday / daha cesur aday →
kapıdan geçen bulunma şansı). 4 varyantı aynı 50 sabit soruda karşılaştırır.

Nesnel yedek işaretleri (metin okumadan sayılır): küçük harfle başlıyor mu,
noktalama ile bitiyor mu, dolgu ifadesi var mı, karakter sayısı.

`brain.py:2031-2047` yalnızca parametreli **kopyalanır**; üretim kodu
değişmez.

**Ölçülen sonuç (01.10.2026, `olcum_raporlari/yedek_olc_0110.json`):**
sezgi **yanlış çıktı** — ham kb modelin metninden **daha temiz**:

| işaret | modelin metni (n=160) | ham kb yedeği (n=40) |
|---|---|---|
| dolgu | %21,9 | **%7,5** |
| yarım cümle | %8,1 | **%0,0** |
| küçük harfle başlıyor | %0,0 | %7,5 |

Kaldıraç **kapı değil `tries`**: `tries 3→6` ekran oranını %72→%86 çıkarıyor
(+7 soru) kapı kuralına dokunmadan; sıcaklık 0,7→0,9 tek başına sadece +2.

**Süre bedeli 01.10–02.10'da ÖLÇÜLDÜ** (`yedek_olc 0110b`, damgalı
`olcum_raporlari/yedek_olc_0110b.json`) — önceki kayıt "süre ölçülmedi" diyordu:

| varyant | ekran oranı | soru/sn | süre | sadakat |
|---|---|---|---|---|
| bugun t3_t07 (taban) | %72 | 3,28 | ×1,00 | taban |
| **t6_t07** | **%86 (+16 puan)** | 7,17 | **×2,18** | **−0,4 puan** |
| t3_t09 | %76 (+0) | 3,12 | ×0,95 | −1,2 puan |
| t6_t09 | %86 (+12 puan) | 7,73 | ×2,36 | −3,4 puan |

Sıcaklık 0,9 **elendi**: `tries`'i 6'ya çıkarmadan ekran oranını artırmıyor
(+0 puan) ve zaten sadakati düşürüyor. `tries 3→6` gerçek bir kaldıraç:
**+16 puan karşılığında ×2,18 süre ve −0,4 puan sadakat.** Karar kullanıcıya.


## sure_olc.py - zaman tavanini kaggle_train.txt'ten OLÇER

`train_llm.sure_ve_hesapla()`'nın tavanının **gerçekte ne kadar yanlış**
olduğunu logdan ölçer; `train_llm.py`'ye dokunmaz.

**Bulduğu 1. hata (01.10.2026, §6.16):** formül `MS_PER_PAIR`'ı kullanıyordu
— bu bir **eğitim çifti** (post-expansion) maliyeti — ama dönen sayı
`coz_max_pairs` → `MAX_PAIRS` zincirinde **ham çift** (pre-expansion)
biriminde kullanılıyor. 01.10'da ölçülen expansion **4,7711** → birim hatası
tam o kadar. Sonuç: tavan gerçek sınırın **5,34 KAT** uzaktaydı ve 29.09'dan
beri **hiç bağlamıyordu**.

**Bulduğu 2. hata (01.10–02.10.2026, §6.18a) — araç KENDİSİNİN hatası:**
`MS_HAM` hesaplanırken encode de içine katılıyordu
(`EPOCH_SN + ENCODE_SN`), ama `sure_ve_hesapla` encode'u
`kalan_dk`'dan **zaten** düşüyor → **encode iki kez sayılıyordu**.
Sabit %9,45 yüksek → tavan 234.207 yerine doğru değer **256.333**.
Ders: sabiti türeten arac ile onu kullanan kod arasında **hiçbir muhasebe
kalemi karşılıklı sayılmamalı.** `MS_HAM` artık encode içermez
(88,7127 ms/ham, 12 epoch); encode içeren değer yalnızca rapor için
`MS_HAM_ENCODE_DAHIL` olarak yazılır.

Arac üç seçeneğin tavanını yan yana basar ve **hangisinin yetersiz olduğunu
gösterir**: (A) sadece `MS_PER_PAIR`'ı düzeltmek yalnızca %12,1 düşürür →
**yetersiz**; (B) ham çift başına ölçülen sabit → **256.333** → **doğru**.

Araç ayrıca **kapsama kaybını canlı `intents.json`'dan ölçer** (varsayım
yazmaz) ve "düzeltme uygulanmış mı" durumunu `DURUM:` bloğunda kendisi
söyler — elle bakılmaz.

Yeni sabiti `MS_PER_HAM_CIFT_EPOCH` olarak `train_llm.py`'ye yazdıktan sonra
**yeni koşu olçtüğünde bu araç tekrar çalıştırılıp sabit güncellenir.**

```
python tools/sure_olc.py [kaggle_train.txt]
```


## kendi_cumlesi.py - model kendi cumlesini mi kuriyor

Her yanit icin yolu ayirt eder: canned (yapistirma) mi, uretilmis mi?
Asil soru buydu; `uretim_olc.py` bunun daha genis hali.

## kalite_olc.py - gecikme, yarim kelime, yapistirma

Dort seyi sayiya cevirir: yanit suresi ve asamasi, uretimin `max_len`'de
kesilip kesilmedigi (token sayisi == max_len ise kesilmis demektir),
yapistirma orani.

## token_olc.py - TOKEN_PER_PAIR'i bagimsiz olcer

`train_llm.py` icindeki TOKEN_PER_PAIR sabitinin gercekten ortalama
token sayisi olup olmadigini dogrular. Bu arac 29.09'daki dongusel
olcum hatasini yakaladi: log `n * TOKEN_PER_PAIR` yaziyordu, yani
"olcum" kendi varsayimini tekrarliyordu.

## Kurallar

- Bu araclar **veri dosyalarina dokunmaz** (`intents.json`,
  `knowledge_map.jsonl`, `corpus.jsonl`, `corpus_ids.jsonl`).
  `app.load_bot()` corpusu yukler ama yazmaz; yazan yol
  `fallback_answer`'in ogrenme yoludur ve bu araclar onu **cagirmaz**.
- `eval_llm.py` `train_llm.py`'yi import eder: olcum surerken
  `train_llm.py`'ye dokunma, A/B'yi kirletir.
- Karsi lastirmada `--compare-key` sadece t-testinin olculdugu metrigi
  secer; farkli decoding ayari olan raporlar uyariyla reddedilir.
