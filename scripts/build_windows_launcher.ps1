$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Wait-AfterError {
    Write-Host ""
    Read-Host "Нажмите Enter, чтобы закрыть окно"
}

try {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $projectRoot = Split-Path -Parent $scriptDir
    Set-Location -LiteralPath $projectRoot

    Write-Host "Сборка лёгкого Windows launcher-exe"
    Write-Host "Это запускатель, а не полноценный автономный exe со всем научным приложением внутри."
    Write-Host "Он ожидает рядом файлы проекта, requirements.txt и обычно .venv."
    Write-Host ""

    $pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        Write-Host "Виртуальное окружение .venv не найдено. Создаём его..."
        & py -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            throw "Не удалось создать .venv через py -m venv .venv"
        }
    }

    & $pythonPath -c "import PyInstaller" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "PyInstaller не найден в .venv. Устанавливаем его только в локальное окружение..."
        & $pythonPath -m pip install pyinstaller
        if ($LASTEXITCODE -ne 0) {
            throw "Не удалось установить PyInstaller"
        }
    }

    Write-Host "Собираем dist\ChargedTrapLauncher.exe..."
    & $pythonPath -m PyInstaller --onefile --name ChargedTrapLauncher launcher.py
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller завершился с ошибкой"
    }

    Write-Host ""
    Write-Host "Готово: dist\ChargedTrapLauncher.exe"
    Write-Host "Напоминание: exe запускает локальный проект, но не содержит всё приложение внутри себя."
}
catch {
    Write-Host ""
    Write-Host "Сборка launcher-exe не удалась." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Wait-AfterError
    exit 1
}

