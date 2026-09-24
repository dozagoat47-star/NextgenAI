# sync_chatgrow.ps1 -- chatgrow kapasite verilerini tazele (Task Scheduler)
#
# Kaggle notebook her acilista `chatgrow_*.jsonl` glob'unu repo kokunden alir.
# Ilk tazeleme BURADA (yerel) yapilir: HF + Wikisource tr kitap ikisi birden
# cekilir, JSONL'ler deterministik (--seed) uretilir, git ile commit+push
# edilir. Kaggle boylece guncel veriyi her oturumda glob ile alir (internete
# bagimli kalmaz).
#
# Cagiran: Task Scheduler "chatgrow-veri-tazeleme" (gunde 4 kez)
#   schtasks /Create /XML chatgrow-task.xml /F  (yonetici PowerShell'de)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path   # .../scripts
$Root      = Split-Path -Parent $ScriptDir                     # .../Nextgen_API
$Python = 'python'

Write-Host "[1/4] HF tr cekiliyor ($(Get-Date))" -ForegroundColor Cyan
& $Python -X utf8 -u fetch_hf_turkish.py --out (Join-Path $Root 'chatgrow_hf_turkish.jsonl') --max-pairs 20000 --seed 7
if ($LASTEXITCODE -ne 0) { throw "fetch_hf_turkish.py basarisiz: $LASTEXITCODE" }

Write-Host "[2/4] Wikisource tr kitap cekiliyor" -ForegroundColor Cyan
& $Python -X utf8 -u build_book_pairs.py --out (Join-Path $Root 'chatgrow_kitap.jsonl') --max-pairs 8000 --per-cat 1600 --ctx-len 48 --resp-len 140 --seed 5
if ($LASTEXITCODE -ne 0) { throw "build_book_pairs.py basarisiz: $LASTEXITCODE" }

Write-Host "[3/4] JSONL boyutlari" -ForegroundColor Cyan
Get-ChildItem -LiteralPath $Root -Filter 'chatgrow_*.jsonl' |
    Select-Object Name, @{n='KB';e={[math]::Round($_.Length/1KB,1)}} |
    Format-Table -AutoSize

Write-Host "[4/4] git commit+push" -ForegroundColor Cyan
if (Test-Path -LiteralPath (Join-Path $Root '.git')) {
    Push-Location $Root
    git add chatgrow_hf_turkish.jsonl chatgrow_kitap.jsonl
    git commit -m "chatgrow: HF+kitap veri tazeleme ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))" -q
    git push origin HEAD -q
    Pop-Location
}
Write-Host "DONE: chatgrow verileri tazelendi" -ForegroundColor Green
