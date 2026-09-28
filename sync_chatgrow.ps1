# sync_chatgrow.ps1 — yerelde veri tazeleme (Kitap + Hugging Face)
#
# Kaggle notebook'u her açılışta `chatgrow_*.jsonl` glob'uyla repo kökündeki
# continuation çiftlerini otomatik toplar; HF/kitap fetch'i INTERNET'e bağlıdır
# ve Kaggle'da ağ yavaş/kesik olabilir. Bu script yerelde (Task Scheduler ile
# günde 3-4 kez, deterministik) her iki kaynağı indirir, JSONL'leri repo köküne
# yazar, commit+push eder. Böylece Kaggle'da internet bağımlılığı kalmaz.
#
# Kullanım:
#   powershell -ExecutionPolicy Bypass -File sync_chatgrow.ps1
#   powershell -ExecutionPolicy Bypass -File sync_chatgrow.ps1 -Seed 7
#
# Her koşuda HF için YENİ dosya üretilir (tohum koşudan türetilir, aşağıya
# bakınız). ESKİ JSONL'ler silinmez: Kaggle glob farklı adlardaki tüm
# chatgrow_*.jsonl'i yakalar, Task Scheduler günde 3-4 kez çalıştırır ve
# Kaggle en taze dosyaları glob'la alır.
#
# İSTİSNA: kitap dosyası SABİT addır (chatgrow_kitap.jsonl) ve tohumsuz
# çalışır — build_book_pairs.py deterministiktir, her koşu aynı 609 çifti
# verir. Zaman damgalı ad, üretici her değiştiğinde yeni dosya biriktiriyordu.

param(
    # 0 (varsayilan) = tohumu KOSUDAN TURET, yani her calistirmada farkli.
    # 29.09 OLCUMU: tohum sabit 7 iken script 19 kez kosmus, 16'sinda HF
    # ciktisi mevcut dosyayla byte-byte ayni olup ATILMIS. Yani gunlerce
    # no-op: betigin "Kaggle en taze dosyayi glob'la alir" amaci gerceklesmiyor.
    # Pozitif deger verirsen elle sabitlersin (test icin).
    [int]$Seed = 0
)

# SYSTEM hesabiyla calisabilmesi icin tam yol; PATH'e guvenme (kullaniciya ozel)
$Python = 'C:\Users\cxc\AppData\Local\Programs\Python\Python312\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Tohum 0 ise kosudan turet: gunun toplam dakikasi (0..1439) + 1. Gun
# 03:00 -> 181, 15:00 -> 901 gibi. Gunde 3-4 kosu oldugu icin ayni
# gun iki kosu ayni tohumu alabilir; o durumda byte-byte kopya cikar ve
# mevcut koruma onu atar (veri kaybi olmaz, sadece o kosu bos gecer).
# AAyari (gun) eklenerek tamamen kaza olmasi engellendi.
if ($Seed -le 0) {
    $Seed = ((Get-Date).Day * 1440) + ((Get-Date).Hour * 60) + (Get-Date).Minute
}

# Yeni uretim, mevcut bir dosyayla ayni icerikteyse repo'ya 0-degerli kopya
# sokmamak icin atilir (HF/kitap kaynaklari deterministik -> her koşu ayni).
$script:dedup_count = 0
function Test-OzdesIcerik {
    param([string]$Path, [string]$Desen)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $yen = (Get-FileHash -LiteralPath $Path -Algorithm MD5).Hash
    foreach ($eski in (Get-ChildItem -LiteralPath $root -Filter $Desen -File -ErrorAction SilentlyContinue)) {
        if ($eski.FullName -eq $Path) { continue }
        if ((Get-FileHash -LiteralPath $eski.FullName -Algorithm MD5).Hash -eq $yen) {
            return $true
        }
    }
    return $false
}

# Task Scheduler SYSTEM kullaniciyla calisir -> ciktiyi log dosyasina yaz
$log = Join-Path $root 'chatgrow_sync.log'
Start-Transcript -Path $log -Append -Force | Out-Null
Write-Host "== calisti: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =="

# --- 1) Hugging Face tr-turkish continuation ciftleri -----------------------
$hf_out = Join-Path $root "chatgrow_hf_$(Get-Date -Format 'yyyyMMdd_HHmm').jsonl"
$hf_ok = $false
Write-Host "[1/4] HF cekiliyor: $hf_out"
try {
    & $Python -X utf8 -u fetch_hf_turkish.py --out $hf_out --max-pairs 1200 --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "fetch_hf_turkish.py rc=$LASTEXITCODE"
    }
    $hf_ok = $true
    Write-Host "  OK: $hf_out"
    if (Test-OzdesIcerik -Path $hf_out -Desen 'chatgrow_hf_*.jsonl') {
        Remove-Item -LiteralPath $hf_out -Force
        $hf_ok = $false
        $script:dedup_count++
        Write-Host "  .. icerik mevcut bir HF dosyasiyla ayni; yeni dosya atildi" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  !! HF basarisiz: $($_.Exception.Message)" -ForegroundColor Red
}

# --- 2) Wikisource tr kitap/masal continuation ciftleri ----------------------
# 29.09: ZAMAN DAMGALI AD KALDIRILDI -> SABIT AD.
# Neden: build_book_pairs.py deterministik (SEED sabit) ve Wikisource
# katalogu tukenmis; her kosuda ayni 609 cifti veriyor. Zaman damgali
# ad, uretici HER degistiginde yeni dosya biriktiriyordu. Koruma
# (Test-OzdesIcerik) byte-byte aynilik ariyordu, ama uretici duzeltmesi
# icerigi kismen degistirdigi icin koruma gecmedi: son kosuda 609
# ciftin yalniz 29'u benzersizdi (%95 tekrar).
# Artik: sabit isim + icerik karsilastirmasi. Ayni gelirse dosyaya
# dokunulmaz (commit yok), uretici iyilesince dosya kendiliginden
# yenilenir. HF yarisi seed'li oldugu icin her kosuda yeni veri verir;
# kitap yarisi SEED'sIZ calisir, yani sabit -> commit gurultusu de yok.
$book_out = Join-Path $root 'chatgrow_kitap.jsonl'
$book_tmp = Join-Path $root 'chatgrow_kitap.tmp.jsonl'
$book_ok = $false
Write-Host "[2/4] kitap cekiliyor: $book_out"
try {
    # --ctx-len/--resp-len VERILMEZ: ureticinin varsayilani artik loader
    # butcesine esit (64/204, testle kilitli). 29.09'da burada 300/300
    # veriliyordu; loader 204'te SERT kestiği icin hasar ureticiye hic
    # girmiyor, dosyaya gomuluyordu (48/609 = %7,88 hasar).
    & $Python -X utf8 -u build_book_pairs.py --out $book_tmp --max-pairs 1200 `
        --per-cat 1600
    if ($LASTEXITCODE -ne 0) {
        throw "build_book_pairs.py rc=$LASTEXITCODE"
    }
    $eski = if (Test-Path -LiteralPath $book_out) {
        (Get-FileHash -LiteralPath $book_out -Algorithm MD5).Hash } else { '' }
    $yeni = (Get-FileHash -LiteralPath $book_tmp -Algorithm MD5).Hash
    if ($yeni -eq $eski) {
        Remove-Item -LiteralPath $book_tmp -Force
        $script:dedup_count++
        Write-Host "  .. icerik mevcut kitap dosyasiyla ayni; dosyaya dokunulmadi" -ForegroundColor Yellow
    } else {
        Move-Item -LiteralPath $book_tmp -Destination $book_out -Force
        $book_ok = $true
        Write-Host "  OK: $book_out (yenilendi)"
    }
} catch {
    Write-Host "  !! kitap basarisiz: $($_.Exception.Message)" -ForegroundColor Red
    if (Test-Path -LiteralPath $book_tmp) { Remove-Item -LiteralPath $book_tmp -Force }
}

# --- 3) doğrulama: her başarılı JSONL en az 1 çift içermeli ------------------
$new_rows = [ordered]@{}
if ($hf_ok -and (Test-Path -LiteralPath $hf_out)) {
    $rows = (Get-Content $hf_out | Measure-Object -Line).Lines
    if ($rows -lt 1) { throw "HF JSONL bos: $hf_out" }
    $new_rows[$hf_out] = $rows
}
if ($book_ok -and (Test-Path -LiteralPath $book_out)) {
    $rows = (Get-Content $book_out | Measure-Object -Line).Lines
    if ($rows -lt 1) { throw "kitap JSONL bos: $book_out" }
    $new_rows[$book_out] = $rows
}
foreach ($f in $new_rows.Keys) {
    Write-Host "[3/4] satir: $f = $($new_rows[$f])"
}
if ($new_rows.Count -eq 0) {
    if ($script:dedup_count -gt 0) {
        Write-Host "  !! uretilen ciktilarin tamami mevcut icerikle ayni; commit atlaniyor (guncel)" -ForegroundColor Yellow
    } else {
        Write-Host "  !! hic yeni JSONL uretilemedi; commit atlaniyor" -ForegroundColor Red
        throw "chatgrow sync: HF ve kitap uretimi basarisiz"
    }
}

# --- 4) git add+commit+push --------------------------------------------------
if (Get-Command git -ErrorAction SilentlyContinue) {
    & git add chatgrow_*.jsonl
    # Yalnizca gercekten degisen/eklenen dosya varsa commit (bos commit onlenir).
    if (& git diff --cached --quiet) {
        Write-Host "[4/4] degisiklik yok; push atlaniyor"
    } else {
        & git commit -m "chatgrow: HF+kitap continuation verileri tazele (HH:MM $Seed)" 2>&1 |
            Out-Null
        & git push origin HEAD 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            # Remote main ilerlemis olabilir: rebase alip yeniden dene.
            Write-Host "  push redirecti (rc=$LASTEXITCODE); fetch + rebase + yeniden push..." -ForegroundColor Yellow
            & git fetch origin 2>&1 | Out-Null
            & git rebase origin/main 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                & git rebase --abort 2>&1 | Out-Null
                Write-Host "  !! rebase cakisti; push atlandi (sonraki kosu halleder)" -ForegroundColor Red
            } else {
                & git push origin HEAD 2>&1 | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "[4/4] !! push basarisiz (rc=$LASTEXITCODE); commit yerelde" -ForegroundColor Red
                } else {
                    Write-Host "[4/4] commit+push tamam"
                }
            }
        } else {
            Write-Host "[4/4] commit+push tamam"
        }
    }
} else {
    Write-Host "[4/4] git yok; JSONL hazir (push atlaniyor)"
}
Write-Host "SYNC TAMAM"
