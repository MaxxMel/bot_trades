import telebot
from telebot import types
from config import TOKEN
from db import clear_table, load_data, save_data_to_db
from data_fetch import fetch_historical_data, interval_to_milliseconds
from preprocess import preprocess_data, create_binary_target, remove_correlated_features
from plotting import send_plot, plot_correlation_matrix_matplotlib, plot_price, plot_factors, plot_predictions, plot_capital_curve
from models import (normalize_data, do_smote, feature_selection_rfe, train_basic_models, 
                    grid_search_random_forest, train_voting_ensemble)
import asyncio
import websockets
import json
import io
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

bot = telebot.TeleBot(TOKEN)

df = None
scaler = None
chosen_factors = []
features_all = []
selected_features_for_model = []
models = {}
model_scores = {}
target_column = 'close'
parsing_state = {}
socket_tasks = {}
shift_n_value = 1
nan_fill_method = None

def show_instructions(chat_id):
    instruction_text = ("Шаг 1: Сбор данных\n"
                        " - Собираем исторические данные с Bybit ...\n"
                        "Шаг 2: Предобработка данных\n"
                        " - Обработка пропусков и конвертация timestamp...\n"
                        # Продолжение инструкции...
                        )
    bot.send_message(chat_id, f"<pre>{instruction_text}</pre>", parse_mode='HTML')

def show_main_menu(chat_id):
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    # Создаем кнопки меню
    instructions_button = types.KeyboardButton('Инструкция 📚')
    btn_reset        = types.KeyboardButton('Обнулить модель ❌')
    btn_parse        = types.KeyboardButton('Спарсить данные 🔎')
    btn_load         = types.KeyboardButton('Загрузить данные 📂')
    # ... добавьте остальные кнопки
    markup.add(instructions_button, btn_reset, btn_parse, btn_load)
    bot.send_message(chat_id, "Главное меню:", reply_markup=markup)

@bot.message_handler(commands=['start'])
def start_command(message):
    show_main_menu(message.chat.id)

@bot.message_handler(func=lambda msg: True)
def handle_text(message):
    # Обработка текстовых сообщений
    chat_id = message.chat.id
    text = message.text.strip()
    # Пример:
    if text == 'Инструкция 📚':
        show_instructions(chat_id)
    else:
        bot.send_message(chat_id, "Неизвестная команда. Нажмите /start для меню.")

# Добавьте обработчики inline callback'ов аналогично
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    # Обработка inline callback'ов
    chat_id = call.message.chat.id
    data = call.data
    # Логика callback'ов (например, для парсинга, выбора факторов и т.д.)
    bot.send_message(chat_id, f"Вы нажали: {data}")
    # Дополните логику по необходимости

def start_bot():
    bot.polling(none_stop=True)
