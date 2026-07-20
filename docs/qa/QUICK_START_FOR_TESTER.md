# Быстрый старт для тестировщика

## 1. Открыть папку проекта

В Windows PowerShell:

```powershell
cd "C:\Users\arosl\Desktop\projects py\3d fem\charged_particle_trap"
```

## Самый простой запуск на Windows

Если нужно быстро открыть приложение без ручного ввода команд, используйте launcher из корня проекта:

1. Дважды нажмите `run_app_windows.bat`.
2. Или откройте PowerShell в папке проекта и выполните:

```powershell
.\run_app_windows.ps1
```

Если уже собран exe launcher, можно запустить:

```powershell
.\dist\ChargedTrapLauncher.exe
```

Это запускатель, а не полная автономная сборка научного приложения. Если SmartScreen ругается на exe, используйте `.bat` или `.ps1`.

Если `.bat` показывает странные ошибки кодировки или запуска, используйте PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_app_windows.ps1
```

## 2. Запустить автоматические тесты

Рекомендуемая команда:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Универсальная команда, если Python настроен в системе:

```powershell
python -m pytest
```

Если тесты прошли, переходите к Streamlit. Если тесты упали, сделайте скриншот
консоли и скопируйте первые строки ошибки.

## Optional: headless regression scenarios

Если Ярослав просит, можно также запустить короткие headless-сценарии без
Streamlit и без браузера:

```powershell
python scripts/run_regression_scenarios.py
```

Они сохраняют сводку в `results/regression/regression_summary.json`. Обычно
ручному тестировщику это не нужно без отдельной просьбы.

## 3. Запустить Streamlit-приложение

Рекомендуемая команда:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Универсальная команда:

```powershell
streamlit run app.py
```

Браузер должен открыться автоматически. Если браузер не открылся, скопируйте
локальный URL из консоли Streamlit, например:

```text
http://localhost:8501
```

и вставьте его в браузер вручную.

## 4. Если не хватает зависимостей

Если появляется ошибка про `streamlit`, `streamlit-drawable-canvas`, `PIL`,
`numpy`, `scipy` или другой пакет, выполните:

```powershell
pip install -r requirements.txt
```

Если используете виртуальное окружение проекта:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 5. Если приложение падает

Сделайте:

1. скриншот красной ошибки в браузере;
2. скриншот или копию текста ошибки из PowerShell;
3. запишите режим, в котором произошла ошибка;
4. запишите точные шаги воспроизведения;
5. оформите баг по шаблону:

```text
docs/qa/BUG_REPORT_TEMPLATE.md
```

## 6. Быстрый ручной маршрут проверки

1. Откройте Streamlit.
2. Нажмите быстрый demo run на главной странице, если кнопка доступна.
3. Проверьте, что появились графики и метрики.
4. Откройте вкладку `Эксперименты`.
5. Скачайте `experiment_config.json`.
6. Скачайте `report.md`.
7. Скачайте ZIP-пакет.
8. Проверьте режим `Рисунок электродов` с картинкой:

```text
examples/qa_assets/mask_four_rectangles.png
```

9. Проверьте режим `Электроды функциями` с пресетами.
10. Проверьте режим `Нарисовать электроды` или PNG fallback, если canvas не
    доступен.
