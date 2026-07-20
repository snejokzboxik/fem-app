"""Lightweight launcher for the local Streamlit app."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
import webbrowser


LOCAL_URL = "http://localhost:8501"


def project_root() -> Path:
    """Return the directory containing this launcher."""

    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        if executable_dir.name.lower() == "dist" and (executable_dir.parent / "app.py").exists():
            return executable_dir.parent
        return executable_dir
    return Path(__file__).resolve().parent


def venv_python(root: Path) -> Path:
    """Return the expected Python executable inside the local .venv."""

    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def print_missing_venv_help(root: Path, python_path: Path) -> None:
    """Print Russian setup guidance when the virtual environment is missing."""

    print("Виртуальное окружение .venv не найдено")
    print()
    print("Создайте окружение и установите зависимости:")
    print("python -m venv .venv")
    if os.name == "nt":
        print(r".\.venv\Scripts\python.exe -m pip install -r requirements.txt")
    else:
        print("./.venv/bin/python -m pip install -r requirements.txt")
    print()
    print(f"Ожидался Python: {python_path}")
    print(f"Папка проекта: {root}")


def wait_for_enter() -> None:
    """Keep the console open after an error."""

    try:
        input("Нажмите Enter, чтобы закрыть окно...")
    except EOFError:
        pass


def main() -> int:
    """Start Streamlit from the local project virtual environment."""

    root = project_root()
    python_path = venv_python(root)

    if not python_path.exists():
        print_missing_venv_help(root, python_path)
        wait_for_enter()
        return 1

    app_path = root / "app.py"
    if not app_path.exists():
        print(f"Файл app.py не найден: {app_path}")
        wait_for_enter()
        return 1

    print("Запускаем Streamlit-приложение...")
    print(f"Папка проекта: {root}")
    print(f"Локальный адрес: {LOCAL_URL}")
    print()

    command = [
        str(python_path),
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=false",
    ]

    try:
        process = subprocess.Popen(command, cwd=root)
    except OSError as exc:
        print("Не удалось запустить Streamlit.")
        print(f"Ошибка: {exc}")
        wait_for_enter()
        return 1

    time.sleep(2.0)
    try:
        webbrowser.open(LOCAL_URL)
    except Exception as exc:
        print(f"Не удалось открыть браузер автоматически: {exc}")
        print(f"Откройте адрес вручную: {LOCAL_URL}")

    try:
        return_code = int(process.wait())
    except KeyboardInterrupt:
        print()
        print("Остановка по Ctrl+C...")
        process.terminate()
        return 130

    if return_code != 0:
        print()
        print(f"Streamlit завершился с ошибкой, код: {return_code}")
        wait_for_enter()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
