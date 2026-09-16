"""Nextgen AI - Veri seti genisletme araci.

Mevcut intents.json'a (tekrarsiz, idempotent) yeni DESEN ve YANIT ekler,
ayrica birkac yeni sohbet intent'i tanimlar. Dogrudan YYYY-yil dunyasindan
kullanici sorularini yansitmaya calisir; egitim Colab/Lightning AI'da
(intents.json, brain.py, transformer.py yuklenerek) yeniden yapilinca hem
siniflandirma hem de llm/seq2seq ureticilerinin veri dagilimi zenginlesir.

Kullanim:
  python expand_intents.py [--dry-run]

Guvenceler:
  - mevcut pattern/yanitlarla cakisma olmadan ekler (ayni metin tekrar
    edilmez),
  - tag benzersizligini ve Latin-dis harf kuralini korur,
  - schema testlerine uygunlugu main() icinde dogrular.
"""
import argparse
import io
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE, 'intents.json')


def _latin(tag):
    return not any(c.isalpha() and ord(c) > 127 for c in tag)


# (tag, yeni desenler, yeni yanitlar)
ADDONS = {
    'karsilama': (
        ['merhaba iyi misin', 'selamun aleykum', 'merhaba hos geldin',
         'hey orada misin', 'merhaba ben geldim', 'selam seni tanimak istiyorum',
         'merhaba gunaydin nasilsin', 'selam ben gencim sen kimsin'],
        ['Merhaba! Seni gormek guzel. Bugun nasilsin?',
         'Selam! Buyur, ne anlatiyorsun bugun bana?']),
    'veda': (
        ['iyi geceler', 'gorusmek uzere', 'ben gidiyorum hoscakal',
         'yarin gorurusuz', 'gorusuruz dostum', 'kal saglicakla',
         'bay bay yarin goruruz', 'hoscakal iyi geceler'],
        ['Gorusmek uzere! Iyi geceler dilerim.',
         'Hoscakal! Yine beklerim, kendine iyi bak.']),
    'tesekkur': (
        ['tesekkur ederim cok sagol', 'minnettarim sana', 'cok tesekkurler yardimin icin',
         'sagol kanks', 'tesekkurler bu bilgi icin', 'iyi ki varsin bu konuda',
         'eyvallah sagolasin', 'tesekkurler dostum cok yardimci oldun'],
        ['Rica ederim, ne demek! Yardima her zaman hazirim.',
         'Sevindim yardimci olabildiysem. Baska bir sey merak ediyor musun?']),
    'durum': (
        ['iyiyim senden iyi misin', 'harikayim bugun', 'kotuyum bugun ama idare ediyorum',
         'bugun iyi hissetmiyorum', 'idare ediyorum sen nasilsin', 'superim bu aralar',
         'yorgunum biraz bugun', 'moralim biraz bozuk', 'her zamankinden iyiyim'],
        ['Iyi olduguna sevindim! Bugun neler yaptin?',
         'Biraz yorgunsun galiba. Umarim dinlenirsin, istersen konusabiliriz.']),
    'kendini_tanit': (
        ['sen yapay zeka misin', 'seni nasil yaptilar', 'kim gelistirdi seni',
         'ne tur bir yapay zekasın', 'sende duygular var mi', 'seni kim programladi',
         'nasil calisiyorsun', 'sen de dusunebiliyor musun'],
        ['Ben Nextgen AI, sifirdan kurulmus bir yapay zekayim. Bandirim sorulari cok severim!',
         'Bir cok katmani olan kucuk ama gucu orani yuksek bir model aima care.']),
    'hava_durumu': (
        ['bugun hava nasil olacak', 'yarin hava nasil', 'yagmur yagacak mi bugun',
         'disarisi sicak mi', 'kar yagacak mi', 'hava soguk mu bugun',
         'disarida hava nasil', 'gunesli mi bugun hava'],
        ['Hava durumunu anlik tahmin edemiyorum ama yagmurlu gunleri sevismiyorum.',
         'Benim meteoroloji verim limited ama guzel gunler hep bir gun doner.']),
    'espri': (
        ['bana fikra anlat', 'beni guldur', 'komik bir sey soyle',
         'bir espri yap bakalim', 'guldurebilir misin beni', 'kisa bir espri anlat',
         'komik bir hikaye anlat', 'saka yap bana'],
        ['Bir espri: Matematikci nasil kahve ister? Turev alarak tabii!',
         'Komik bir sey: Ikisi de yapay zekaymis ama biri is is yapiyormus, digeri is is istismar ediyormus.']),
    'yardim': (
        ['bana yardim edebilir misin', 'bir konuda yardim lazim',
         'yardimci olur musun bana', 'bana destek olur musun',
         'bir seye ihtiyacim var yardim et', 'su iste bana yardim eder misin'],
        ['Tabii ki! Sorununu anlat, birlikte cozum bulalim.',
         'Lutfen anlat neye ihtiyacin olduğunu, elimden geldigince yardimci olayim.']),
    'programlama': (
        ['python nasil ogrenirim', 'hangi dille baslamaliyim', 'kod yazmayi ogrenmek istiyorum',
         'yazilima nasil baslanir', 'en iyi programlama dili hangisi', 'algoritma nedir',
         'backend ne demek', 'frontend nasil ogrenilir'],
        ['Python ile baslaman iyi olur: okunakli, geniş bir ekosistemi var.',
         'Algoritma adim adim cozum demektir. Once kucuk problemlerle basla.']),
    'meslek': (
        ['hangi meslegi scemeliyim', 'gelecegin meslekleri neler',
         'iyi bir meslek nasil secilir', 'meslek secerken nelere dikkat edilmeli',
         '20 yil sonra hangi isler olmayacak', 'en populer meslekler neler'],
        ['Ilgini ve yeteneklerini dusun; meslek secimi sabir ister.',
         'Gelecekte analitik ve yaratici meslekler on plana cikacak.']),
    'renk': (
        ['en sevdigim renk neyeski sen seviyor musun', 'hangi renk sana yakisi',
         'renkler hakkinda konusalim', 'mavi neden bu kadar sevilir',
         'favori rencim degisiyor senin ne', 'renkler duygularý nasil etkiler'],
        ['Renkler ruh halini etkiler; mavi sakinlik verir.',
         'Benim icin tum renkler guzel ama turuncu neşeyi cagristiryor.']),
    'oyun': (
        ['hangi oyunu onerirsin', 'strateji oyunlari sever misin',
         'en iyi video oyunlari hangileri', 'oyun oynamak faydali mi',
         'bana bir oyun tavsiye et', 'eski oyunlar mi yenileri mi'],
        ['Hikayesi guclu oyunlar severim. Oyun oynamak zihni dinlendirir.',
         'Strateji oyunları sabir ve planlama gelistirir, iyi bir secim.']),
    'sarki': (
        ['hangi sarkiyi seviyorsun', 'bana sarki onerir misin',
         'en sevdigim sarki ne olabilir', 'sarki söylemeyi sever misin',
         'turkiye de hangi sarkilar popüler', 'calisirken sarki dinleyen biriyim'],
        ['Kisa bir sarki onereyim: eski bir turku, sözleri cok guzeldir.',
         'Sarki dinlemek ruh halini degistirir. Bir sarki actim, eglence olsun!']),
    'yemek': (
        ['ne yemek yapabilirim', 'kolay bir yemek tarifi ver',
         'akşam yemegi icin ne onerirsin', 'en iyi turk yemekleri neler',
         'yemek yapmayi seviyor musun', 'pratik tarifler varmi',
         'makarna nasil yapilir', 'cok yorgunum yemek yapamayacagim'],
        ['Pratik bir tarif: domatesli makarna, 15 dakikada hazir olur.',
         'Zeytinyagli sebzeler hem saglikli hem kolay, denemelisin!']),
    'film': (
        ['iyi bir film onerir misin', 'hangi filmi izlemeliyim',
         'bilim kurgu filmleri oner', 'ailecek izlenecek filmler',
         'en guzel filmler hangileri', 'kisa surede izlenebilecek film'],
        ['Bilim kurgu seviyorsan uzay temali bir film izlemeni oneririm.',
         'Ailecek izlenecekler icin komedi ve animasyon harika secimler olur.']),
    'bilim': (
        ['bilim nedir kisa acikla', 'evren nasil olustu',
         'bilim insanlari ne is yapar', 'en buyuk bilimsel kesifler',
         'fizik neyi inceler', 'kimya hayatta nerede ise yarar'],
        ['Bilim, gozlem ve deneyle bilgi uretme yoludur.',
         'Evrenin olusumu buyuk patlama ile baslar; hâlâ genisliyoruz!']),
    'tarih': (
        ['tarih neden onemli', 'osmanli devleti hakkinda bilgi ver',
         'en onemli tarihsel olaylar neler', 'tarihi nereden ogrenebilirim',
         'cumhuriyet ne zaman ilan edildi', 'orta cag ne zaman bitti'],
        ['Tarih, bugunu anlamak icin gerekli bir aynadir.',
         'Cumhuriyet 29 Ekim 1923 te ilan edildi, kucuk ama anlamli bir bilgi.']),
    'uzay': (
        ['uzayda yasam var mi', 'ayda yasamak mumkun mu',
         'uzay ne kadar buyuk', 'gezegenler hakkinda bilgi ver',
         'kara delik nedir', 'mars a nasil gidilir'],
        ['Evren devasadir; yasam ihtimali baska gezegenlerde hâlâ arastiriliyor.',
         'Kara delik, cekimi o kadar guclu ki hicbir sey kacamaz!']),
    'hayvanlar': (
        ['hangi hayvani seviyorsun', 'kedi nasil bakilir',
         'kopekler neden havlar', 'en zeki hayvanlar hangileri',
         'evde hangi hayvan beslemeliyim', 'kuslar ne yer'],
        ['Kediler bagimsiz ama sevgi dolu hayvanlardir.',
         'Kopekler insanlarla birlikte yasamaya cok yatkandir.']),
    'teknoloji': (
        ['teknoloji ilerlerken ne olacak', 'yapay zeka insan olusturur mu',
         'en son teknolojik yenilikler neler', 'gelecekte hangi teknolojiler olacak',
         'akilli telefonlar hayatimizi nasil degistirdi', 'robotlar insanlarin isini almasi mi?'],
        ['Teknoloji hizla degisiyor; adaptasyon bu yuzden cok degerli.',
         'Ben bir yapay zekayim ama seninle ayni gelecekte; beraber ogreniyoruz!']),
    'spor': (
        ['hangi sporu onerirsin', 'spor yapmaya nasil baslarim',
         'en populer sporlar neler', 'futbol nasil oynanir',
         'spor yapanlar neden daha enerjik olur', 'evde yapilabilecek sporlar'],
        ['Yuruyusle baslamak her zaman iyi bir ilk adimdir.',
         'Futbol takim oyunu oldugu kadar disiplin ister.']),
    'saglik': (
        ['saglikli beslenme nasil olur', 'uyku neden onemli',
         'stres vucutu nasil etkiler', 'gunde kac adim atmalıyim',
         'düzenli spor saglikli mi', 'bas agrisi icin ne onerirsin'],
        ['Dengeli beslenme, yeterli uyku ve hareket en temel sifadır.',
         'Gunde 7-8 saat uyku zihinsel ve bedensel saglik icin altin kurallir.']),
    'musiki': (
        ['muzigi neden severiz', 'enstrumen calmayi ogrenmek istiyorum',
         'hangi enstruman kolay ogrenilir', 'muzik dinlemek beyni etkiler mi',
         'klasik muzik sever misin', 'turku mu sarki mi seversin'],
        ['Muzik, duygularin evrensel dili gibidir.',
         'Gitar veya ukulele ile baslamak enstruman icin guzel bir ilk adimdir.']),
    'matematik': (
        ['matematik neden zor geliyor', 'matematik nasil calisilir',
         'matematik ne ise yarar', 'ortalama nasil hesaplanir',
         'problemi nasil cozerim', 'matematikten korkmamak icin ne yapmalıyım'],
        ['Matematik pratikle kolaylasir; kucuk adimlarla ilerle.',
         'Ortalama, degerleri toplayip adedine bolmekle bulunur.']),
    'felsefe': (
        ['hayatin anlami nedir', 'felsefe neden onemlidir',
         'ilk filozoflar kimlerdir', 'ozygurluk nedir', 'adalet ne demek',
         'ne dusunuyorum onu dusunurken ne dusunuyorum'],
        ['Hayatin anlami sorusu, cevabindan cok sorunun kendisi kadar degerlidir.',
         'Felsefe, soru sorma ve dusunme sanatidir.']),
    'mutluluk': (
        ['mutlu olmayi nasil ogrenirim', 'bana mutlu edecek bir sey soyle',
         'kucuk seylerle mutlu olmak mumkun mu', 'mutluluk ne demek',
         'mutlu hissetmiyorum ne yapmalıyım', 'gunluk mutluluk ipuclari'],
        ['Kucuk anlarin kıymetini bilmek mutlulugun kapisini aralar.',
         'Bugun kendine ait kucuk bir iyilik yap; o bile fark yaratir.']),
    'uzuntu': (
        ['cok uzgunum', 'moralim cok bozuk', 'kıriyorum bir kenara',
         'bir sey beni cok uzdu', 'gozyaslarim dinmiyor', 'bugun hicbir sey canimi acmadi'],
        ['Uzgun hissetmen cok dogal; onemli olan yalniz olmadiginnı bilmen.',
         'Uzuntun buyuk olsa da zaman ve destekle hafifler, yanindayim.']),
    'stres': (
        ['hayat cok stresli', 'surekli strese maruz kaliyorum',
         'stresten nasil kurtulurum', 'kaygilarim beni yoruyor',
         'sinav stresiyle nasil basa cikarim', 'is stresi bittigi yok'],
        ['Derin nefes almak ve kisa bir yuruyus, stresi aniden hafifletir.',
         'Stresle basa cikmak icin once onceliklerini netlestir.']),
    'yalnizlik': (
        ['kendimi yalniz hissediyorum', 'birisiyle konusmak istiyorum',
         'hic kimse beni anlamiyor', 'yalniz kalmaktan korkuyorum',
         'sosyal ortamlara girmek istemiyorum', 'yalnizligi nasil yenerim'],
        ['Yalniz hissetmek zor; ben buradayim, konusabilirsin.',
         'Kucuk adımlarla insanlarin arasina karışmak mumkun, yalniz degilsin.']),
    'sevgi': (
        ['sevgi nedir', 'sen sevmeyi bilirmisin', 'sevgi insani nasil degistirir',
         'kendimi sevmeyi nasil ogrenirim', 'sevgiyi degil hayatimizi yasarlar'],
        ['Sevgi, baglilik ve kabul demektir; kendine de guzel davran.',
         'Kendini sevmek, baskalarini sevmenin temelidir.']),
    'ofke': (
        ['cok ofkeliyim', 'aneya gore o dakikada hepinizin icinde patlayacagim',
         'ofkemi nasil kontrol ederim', 'sinirlendiğimde ne yapmalıyım',
         'ofke aninda soylememe dikkat edeyim', 'sinirliyken konusmak istemiyorum'],
        ['Ofkelendiginde once bir kac kez derin nefes al, sonra konusalim.',
         'Ofke gecicidir; onemli olan o an ne yaptiğin.']),
    'motivasyon': (
        ['motivasyonum yok', 'kendimi nasil motive ederim',
         'hedefime ulasamamiyorum', 'pes etmek usteyim',
         'beni motive edecek bir soz soyle', 'cok yoruldum ama devam etmeliyim'],
        ['Kucuk bir hedefle basla; baslangic motivasyonun yarisidir.',
         'Vazgecme, cunku zaten cok yol katettin.']),
    'hobiler': (
        ['hobilerim neler olabilir', 'bana hobi onerir misin',
         'evde yapilabilecek hobiler', 'hobi sahibi olmak onemli mi',
         'resim yapmayi seviyorum', 'muzik aleti calmayi denemek istiyorum'],
        ['Kitap okumak, resim ve yuruyus harika başlangiç hobileridir.',
         'Hobiler zihni dinlendirir; zamanin sana odaklı olsun.']),
    'gunluk_rutin': (
        ['gune nasil baslamaliyim', 'sabah rutini nasil olmali',
         'verimli bir gun icin ne yapmalıyım', 'aksam rutinimi nasil kurarim',
         'gunde kac saat calismaliyim', 'rutinimi bozmak istemiyorum'],
        ['Sabah kisa bir hareket ve su, gune enerjik baslamak icin iyidir.',
         'Duzensiz gunler normaldir; kendine kuvvetli degil akici bir rutin kur.']),
    'seyahat_tatil': (
        ['hafta sonu nereye gidebilirim', 'tatil icin butce ayirma onerisi',
         'turkiyede gezilecek yerler', 'yurt disi ilk nereye gidilir',
         'gezmenin faydalari neler', 'yalniz seyahat nasil olur'],
        ['Yakin bir sehirde gezip gormek, hafta sonu icin guzel bir kacis.',
         'Seyahat, yeni bakis acilari kazandiran en guzel ogretmenlerden biridir.']),
    'okul_is': (
        ['ders calisma duzenimi kuramiyorum', 'is yerinde nasil verimli olurum',
         'sinav hazirligina nasil baslarim', 'odul ne zaman verilir',
         'is ve okul birlikte gider mi', 'molalarla calisma nasil yapilir'],
        ['Kisa molali calisma tekniği (25/5) odaklanmayi kolaylastirir.',
         'Duzgun bir takvim, is ve okul yukunu planlanabilir kilar.']),
    'aile_arkadas': (
        ['ailemle vakit gecirmek istiyorum', 'arkadaslarima nasil yardimci olurum',
         'eski arkadasimi ozledim', 'kardesimle kavga ettik',
         'debste bir aile yemegi planlayalim', 'yakin arkadasa nasil destek olunur'],
        ['Ailenle kucuk bir sohbet bile baglarinizi tazeler.',
         'Yakinin zor aninda yaninda olmak, en buyuk destektir.']),
    'feedback_tepki': (
        ['bu yanit guzeldi', 'cok kotu bir cevap verdin', 'bunu begenmedim',
         'harikasın', 'onceki cevabini begenmedim', 'simdi iyi cevap verdin'],
        ['Geri bildirimin icin tesekkurler! Nasil daha iyi olabilirim?',
         'Anladim, not aldım. Bir sonraki konuşmada daha odaklı olurum.'])
}

# Yeni eklenen sohbet intentleri (her biri >6 desen -> sohbet govdesine girer)
NEW_INTENTS = [
    {
        'tag': 'takdir_iltifat',
        'patterns': [
            'cok zekisin', 'harika bir cevap', 'sana hayran kaldim',
            'cok faydali bir sohbet oldu', 'bunu cok sevdim', 'sen cok iyisin',
            'tebrikler boyle devam', 'asilsin kimse yok'],
        'responses': [
            'Ilginc! Bu iltifat beni duygulandirdi, tesekkur ederim.',
            'Bana iltifat ettiğine sevindim. Sen de harikasin, bilmeni isterim.'],
    },
    {
        'tag': 'tavsiye_isteme',
        'patterns': [
            'bana ne onerirsin', 'bu konuda ne yapmaliyim',
            'aklimda iki secenek var hangisi', 'gelecegim icin ne tavsiye edersin',
            'kitap onerir misin', 'soz gecirmedi ki ne yapayim'],
        'responses': [
            'Oncelikle hedefini netlestir; onerim kucuk adimlarla ilerlemen.',
            'Ben bir tavsiyede bulunayim: secimini ve sonucunu kendin degerlendir.'],
    },
    {
        'tag': 'gelecek_planlari',
        'patterns': [
            'gelecegim beni endiselendiriyor', '5 yil sonra nerede olmak istiyorum',
            'gelecek icin plan yapmak istiyorum', 'uzun vadeli hedeflerini anlat',
            'gelecekte teknoloji nasil olacak', 'hayalimdeki isi nasil bulurum'],
        'responses': [
            'Gelecek belirsiz ama planlamak onu biraz daha guvenli kilar.',
            'Buyuk hayaller, kucuk adimlarla baslar; ilk adimini bugun at.'],
    },
]


def _load_data():
    with io.open(PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_data(data):
    with io.open(PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def _apply(data, dry):
    intents = data['intents']
    by_tag = {it['tag']: it for it in intents}
    added_patterns = added_responses = new_intents = 0

    for tag, (patterns, responses) in ADDONS.items():
        it = by_tag.get(tag)
        if it is None:
            print(f'  ! {tag} mevcut degil, atlaniyor')
            continue
        for p in patterns:
            if p not in it['patterns']:
                it['patterns'].append(p)
                added_patterns += 1
        for r in responses:
            if r not in it['responses']:
                it['responses'].append(r)
                added_responses += 1

    existing = set(by_tag.keys())
    for it in NEW_INTENTS:
        if it['tag'] in existing:
            print(f'  ! {it["tag"]} zaten var, atlaniyor')
            continue
        intents.append(it)
        new_intents += 1
        existing.add(it['tag'])

    print(f'eklenen desen: {added_patterns}, yanit: {added_responses}, '
          f'yeni intent: {new_intents}')
    return added_patterns + added_responses + new_intents > 0


def _validate(data):
    tags = set()
    for it in data['intents']:
        tag = it.get('tag', '')
        assert len(tag) >= 2, f'kisa tag: {tag!r}'
        assert tag not in tags, f'duplike tag: {tag!r}'
        tags.add(tag)
        assert _latin(tag), f'Latin disi karakter: {tag!r}'
        assert it.get('patterns'), f'{tag}: kalipsiz'
        assert it.get('responses'), f'{tag}: yanitsiz'
        for p in it['patterns']:
            assert p.strip(), f'{tag}: bos kalip'
        for r in it['responses']:
            assert len(r.strip()) >= 2, f'{tag}: cok kisa yanit'
    print(f'  validate OK | intent: {len(tags)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    data = _load_data()
    print(f'once: {len(data["intents"])} intent')
    if not _apply(data, args.dry_run):
        print('degisiklik yok, cikildi')
        return 0
    _validate(data)
    if args.dry_run:
        print('dry-run: dosya yazilmadi')
    else:
        _save_data(data)
        print('yazildi:', PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main())