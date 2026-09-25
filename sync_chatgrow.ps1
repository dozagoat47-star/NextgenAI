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
$book_out = Join-Path $root "chatgrow_kitap_$(Get-Date -Format 'yyyyMMdd_HHmm').jsonl"
$book_ok = $false
Write-Host "[2/4] kitap cekiliyor: $book_out"
try {
    & $Python -X utf8 -u build_book_pairs.py --out $book_out --max-pairs 1200 `
        --per-cat 1600 --ctx-len 300 --resp-len 300 --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "build_book_pairs.py rc=$LASTEXITCODE"
    }
    $book_ok = $true
    Write-Host "  OK: $book_out"
    if (Test-OzdesIcerik -Path $book_out -Desen 'chatgrow_kitap_*.jsonl') {
        Remove-Item -LiteralPath $book_out -Force
        $book_ok = $false
        $script:dedup_count++
        Write-Host "  .. icerik mevcut bir kitap dosyasiyla ayni; yeni dosya atildi" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  !! kitap basarisiz: $($_.Exception.Message)" -ForegroundColor Red
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
