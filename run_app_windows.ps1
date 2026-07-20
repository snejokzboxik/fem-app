$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Wait-AfterError {
    Write-Host ""
    Read-Host "Нажмите Enter, чтобы закрыть окно"
}

try {
    Set-Location -LiteralPath $PSScriptRoot

    Write-Host "Запуск проекта charged_particle_trap"
    Write-Host "Папка проекта: $(Get-Location)"
    Write-Host ""

    $pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

    if (-not (Test-Path -LiteralPath $pythonPath)) {
        Write-Host "Виртуальное окружение .venv не найдено. Создаём его..."
        & py -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            throw "Не удалось создать .venv через py -m venv .venv"
        }
    }

    Write-Host "Устанавливаем или обновляем зависимости из requirements.txt..."
    & $pythonPath -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось установить зависимости из requirements.txt"
    }

    Write-Host ""
    Write-Host "Запускаем Streamlit."
    Write-Host "Локальный адрес: http://localhost:8501"
    Write-Host "Если браузер не открылся автоматически, откройте этот адрес вручную."
    Write-Host ""

    & $pythonPath -m streamlit run app.py --server.headless=false
    if ($LASTEXITCODE -ne 0) {
        throw "Streamlit завершился с ошибкой, код: $LASTEXITCODE"
    }
}
catch {
    Write-Host ""
    Write-Host "Произошла ошибка запуска." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Wait-AfterError
    exit 1
}

