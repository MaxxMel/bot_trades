from telegram_handlers import start_bot
from db import create_table

def main():
    create_table()  # Создаем таблицу БД, если не существует
    start_bot()     # Запускаем Telegram-бота (polling)

if __name__ == "__main__":
    main()
