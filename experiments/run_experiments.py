#!/usr/bin/env python3
"""
Quick launcher for UMC experiments.

Запуск экспериментов УМС в разных режимах.
"""

import argparse
import sys
import os

def run_camera_display():
    """Запуск эксперимента камера-дисплей."""
    print("🎥 Запуск эксперимента Камера-Дисплей...")
    import sys
    import os
    experiments_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(experiments_path)
    sys.path.insert(0, project_root)

    from experiments.camera_display.app import CameraDisplayApp

    app = CameraDisplayApp()
    app.launch(server_port=7861, share=False)

def run_news_stream():
    """Запуск эксперимента новостной поток."""
    print("📰 Запуск эксперимента Новостной поток...")
    import sys
    import os
    experiments_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(experiments_path)
    sys.path.insert(0, project_root)

    from experiments.news_stream.app import NewsStreamApp

    app = NewsStreamApp()
    app.create_streamlit_app()

def run_trading():
    """Запуск эксперимента трейдинг."""
    print("📈 Запуск эксперимента Эмоциональный трейдинг...")
    import sys
    import os
    experiments_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(experiments_path)
    sys.path.insert(0, project_root)

    from experiments.trading.app import TradingExperimentApp

    app = TradingExperimentApp()
    app.create_streamlit_app()

def main():
    parser = argparse.ArgumentParser(description="UMC Experiments Launcher")
    parser.add_argument(
        'experiment',
        choices=['camera', 'news', 'trading', 'all'],
        help='Какой эксперимент запустить'
    )
    parser.add_argument(
        '--port', type=int, default=8501,
        help='Порт для Streamlit приложений'
    )

    args = parser.parse_args()

    # Установка пути
    experiments_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(experiments_path)
    sys.path.insert(0, project_root)

    if args.experiment == 'camera':
        run_camera_display()
    elif args.experiment == 'news':
        os.environ['STREAMLIT_SERVER_PORT'] = str(args.port)
        run_news_stream()
    elif args.experiment == 'trading':
        os.environ['STREAMLIT_SERVER_PORT'] = str(args.port + 1)
        run_trading()
    elif args.experiment == 'all':
        print("🚀 Запуск всех экспериментов...")
        print("1. Камера-Дисплей: http://localhost:7861")
        print("2. Новости: http://localhost:8501")
        print("3. Трейдинг: http://localhost:8502")
        print("\nИспользуйте Ctrl+C для остановки")

        import threading
        import time

        # Запуск в отдельных потоках
        threads = []

        # Камера-дисплей
        def start_camera():
            try:
                run_camera_display()
            except KeyboardInterrupt:
                pass

        camera_thread = threading.Thread(target=start_camera, daemon=True)
        threads.append(camera_thread)

        # Новости
        def start_news():
            time.sleep(2)  # Небольшая задержка
            os.environ['STREAMLIT_SERVER_PORT'] = '8501'
            try:
                run_news_stream()
            except KeyboardInterrupt:
                pass

        news_thread = threading.Thread(target=start_news, daemon=True)
        threads.append(news_thread)

        # Трейдинг
        def start_trading():
            time.sleep(4)  # Задержка
            os.environ['STREAMLIT_SERVER_PORT'] = '8502'
            try:
                run_trading()
            except KeyboardInterrupt:
                pass

        trading_thread = threading.Thread(target=start_trading, daemon=True)
        threads.append(trading_thread)

        # Запуск всех потоков
        for thread in threads:
            thread.start()

        # Ожидание
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            print("\n🛑 Остановка всех экспериментов...")

if __name__ == "__main__":
    main()