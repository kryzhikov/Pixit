import os

import telebot
from telebot import types
from loguru import logger

from pixilizer.image import Pixilizer
from pixilizer.utils import *

telebot.apihelper.READ_TIMEOUT = 5

token = os.environ['TELEGRAM_TOKEN']
bot = telebot.TeleBot(token)
Px = Pixilizer(pixel_relative_size=1)

logger.add("overall.log")
logger.info("Bot initialized")


def get_like_markup():
    buttons = [
        types.InlineKeyboardButton(text="❤️", callback_data="pass"),
        # types.InlineKeyboardButton(text="💔", callback_data="dislike"),
    ]
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    keyboard.add(*buttons)
    return keyboard


def get_dislike_markup():
    buttons = [
        # types.InlineKeyboardButton(text="❤️", callback_data="like"),
        types.InlineKeyboardButton(text="💔", callback_data="pass"),
    ]
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    keyboard.add(*buttons)
    return keyboard


@bot.message_handler(commands=['start', 'help'])
def handle_start_help(message):
    logger.info(f"New user {message.chat.id}")
    bot.send_message(
        message.chat.id,
        "Hi, i can turn your photo to pixel art. Try i sending me some pic!")


@bot.message_handler(content_types=["photo"])
def handle_message_photo(message):
    logger.info(f"Got photo from {message.chat.id}")
    markup = types.InlineKeyboardMarkup()
    buttons = [
        types.InlineKeyboardButton(text="❤️", callback_data="like"),
        types.InlineKeyboardButton(text="💔", callback_data="dislike"),
    ]
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    keyboard.add(*buttons)
    raw = message.photo[-1].file_id
    file_info = bot.get_file(raw)
    downloaded_file = bot.download_file(file_info.file_path)
    if not os.path.exists(f"./{message.chat.id}/"):
        os.mkdir(f"./{message.chat.id}/")
    path = f"./{message.chat.id}/user_pic" + ".jpg"
    with open(path, "wb") as new_file:  # save file
        new_file.write(downloaded_file)
    # Read image
    image = read_source_image(path)
    # Perform pixilising
    rez, _ = Px.forward(image, True, 4, 10)
    # Transform to PIL image and save
    pil_rez = np_to_pil(rez, 1.2)
    pil_rez.save(path)
    # Load binary
    bin_image = open(path, 'rb')
    bot.send_photo(message.chat.id, bin_image, reply_markup=keyboard)
    logger.info(f"Sent photo to {message.chat.id}")


@bot.callback_query_handler(func=lambda call: call.data == 'like')
def log_like(call: types.CallbackQuery):
    logger.info(f"LIKE from {call.from_user.id}")
    bot.edit_message_reply_markup(call.message.chat.id,
                                  call.message.message_id,
                                  reply_markup=get_like_markup())


@bot.callback_query_handler(func=lambda call: call.data == 'dislike')
def log_dislike(call: types.CallbackQuery):
    logger.info(f"DISLIKE from {call.from_user.id}")
    bot.edit_message_reply_markup(call.message.chat.id,
                                  call.message.message_id,
                                  reply_markup=get_dislike_markup())


@bot.callback_query_handler(func=lambda call: call.data == 'pass')
def log_dislike(call: types.CallbackQuery):
    pass


bot.polling()
