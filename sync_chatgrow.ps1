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
# Her çalıştırmada yeni dosyalar üretilir (fetch_hf_turkish + build_book_pairs),
# ESKİ JSONL'ler silinmez (glob farklı adlardaki tüm chatgrow_*.jsonl'i yakalar);
# Task Scheduler bunu günde 3-4 kez çalıştırır; Kaggle en taze dosyayı glob'la alır.

param(
    [int]$Seed = 7
)

# SYSTEM hesabiyla calisabilmesi icin tam yol; PATH'e guvenme (kullaniciya ozel)
$Python = 'C:\Users\cxc\AppData\Local\Programs\Python\Python312\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Task Scheduler SYSTEM kullaniciyla calisir -> ciktiyi log dosyasina yaz
$log = Join-Path $root 'chatgrow_sync.log'
Start-Transcript -Path $log -Append -Force | Out-Null
Write-Host "== calisti: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =="

# --- 1) Hugging Face tr-turkish continuation ciftleri -----------------------
$hf_out = Join-Path $root "chatgrow_hf_$(Get-Date -Format 'yyyyMMdd_HHmm').jsonl"
Write-Host "[1/4] HF cekiliyor: $hf_out"
& $Python -X utf8 -u fetch_hf_turkish.py --out $hf_out --max-pairs 1200 --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "fetch_hf_turkish.py hata: $LASTEXITCODE" }

# --- 2) Wikisource tr kitap/masal continuation ciftleri ----------------------
$book_out = Join-Path $root "chatgrow_kitap_$(Get-Date -Format 'yyyyMMdd_HHmm').jsonl"
Write-Host "[2/4] kitap cekiliyor: $book_out"
& $Python -X utf8 -u build_book_pairs.py --out $book_out --max-pairs 1200 `
    --per-cat 1600 --ctx-len 300 --resp-len 300 --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "build_book_pairs.py hata: $LASTEXITCODE" }

# --- 3) doğrulama: her iki JSONL en az 1 cift içermeli ----------------------
$rows_hf  = (Get-Content $hf_out   | Measure-Object -Line).Lines
$rows_book = (Get-Content $book_out | Measure-Object -Line).Lines
Write-Host "[3/4] satirlar: HF=$rows_hf kitap=$rows_book"
if ($rows_hf -lt 1 -or $rows_book -lt 1) { throw "JSONL'ler bos: $hf_out / $book_out" }

# --- 4) git add+commit+push --------------------------------------------------
if (Get-Command git -ErrorAction SilentlyContinue) {
    & git add chatgrow_*.jsonl
    & git commit -m "chatgrow: HF+kitap continuation verileri tazele (HH:MM $Seed)" 2>&1 |
        Out-Null
    & git push origin HEAD 2>&1 | Out-Null
    Write-Host "[4/4] commit+push tamam"
} else {
    Write-Host "[4/4] git yok; JSONL hazir (push atlanıyor): $hf_out, $book_out"
}
Write-Host "SYNC TAMAM"
