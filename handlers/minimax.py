import time
from typing import Any

from anthropic import Anthropic
from expiringdict import ExpiringDict
from openai import OpenAI
from telebot import TeleBot
from telebot.types import Message

from config import settings

from ._utils import bot_reply_first, bot_reply_markdown, enrich_text_with_urls, logger

MINIMAX_MODEL = settings.minimax_model
OPENAI_BASE_URL = "https://api.minimax.io/v1"
ANTHROPIC_BASE_URL = "https://api.minimax.io/anthropic"


def _create_client() -> OpenAI | Anthropic | None:
    if not settings.minimax_api_key:
        return None
    if settings.minimax_api_type == "anthropic":
        return Anthropic(
            api_key=settings.minimax_api_key,
            base_url=settings.minimax_base_url or ANTHROPIC_BASE_URL,
        )
    return OpenAI(
        api_key=settings.minimax_api_key,
        base_url=settings.minimax_base_url or OPENAI_BASE_URL,
    )


client = _create_client()

# Global history cache
minimax_player_dict = ExpiringDict(max_len=1000, max_age_seconds=600)
minimax_pro_player_dict = ExpiringDict(max_len=1000, max_age_seconds=600)


def _anthropic_text(content: list[Any]) -> str:
    return "".join(
        block.text for block in content if getattr(block, "type", None) == "text"
    )


def _complete(messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if client is None:
        raise RuntimeError("MiniMax API key is not configured")
    if settings.minimax_api_type == "anthropic":
        response = client.messages.create(
            messages=messages,
            max_tokens=4096,
            model=MINIMAX_MODEL,
        )
        return _anthropic_text(response.content), {
            "role": response.role,
            "content": response.content,
        }

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=4096,
        model=MINIMAX_MODEL,
    )
    content = response.choices[0].message.content or ""
    return content, {"role": "assistant", "content": content}


def _stream(
    messages: list[dict[str, Any]], reply_id: Message, who: str, bot: TeleBot
) -> tuple[str, dict[str, Any]]:
    if client is None:
        raise RuntimeError("MiniMax API key is not configured")

    text = ""
    last_update = time.time()
    if settings.minimax_api_type == "anthropic":
        with client.messages.stream(
            messages=messages,
            max_tokens=8192,
            model=MINIMAX_MODEL,
        ) as stream:
            for delta in stream.text_stream:
                text += delta
                if time.time() - last_update > 1.7:
                    last_update = time.time()
                    bot_reply_markdown(reply_id, who, text, bot, split_text=False)
            response = stream.get_final_message()
        return text, {"role": response.role, "content": response.content}

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=8192,
        model=MINIMAX_MODEL,
        stream=True,
    )
    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if not delta:
            continue
        text += delta
        if time.time() - last_update > 1.7:
            last_update = time.time()
            bot_reply_markdown(reply_id, who, text, bot, split_text=False)
    return text, {"role": "assistant", "content": text}


def minimax_handler(message: Message, bot: TeleBot) -> None:
    """minimax : /minimax <question>"""
    prompt = message.text.strip()
    user_id = str(message.from_user.id)
    messages = minimax_player_dict.setdefault(user_id, [])

    if prompt == "clear":
        bot.reply_to(message, "just clear your minimax messages history")
        messages.clear()
        return
    if prompt[:4].lower() == "new ":
        prompt = prompt[4:].strip()
        messages.clear()
    prompt = enrich_text_with_urls(prompt)

    who = "MiniMax"
    reply_id = bot_reply_first(message, who, bot)
    messages.append({"role": "user", "content": prompt})
    if len(messages) > 10:
        del messages[:2]

    try:
        text, assistant_message = _complete(messages)
        if not text:
            messages.pop()
            text = f"{who} did not answer."
        else:
            messages.append(assistant_message)
    except Exception:
        logger.exception("MiniMax handler error")
        bot.reply_to(message, "answer wrong maybe up to the max token")
        messages.pop()
        return

    bot_reply_markdown(reply_id, who, text, bot)


def minimax_pro_handler(message: Message, bot: TeleBot) -> None:
    """minimax_pro : /minimax_pro <question>"""
    prompt = message.text.strip()
    user_id = str(message.from_user.id)
    messages = minimax_pro_player_dict.setdefault(user_id, [])

    if prompt == "clear":
        bot.reply_to(message, "just clear your minimax messages history")
        messages.clear()
        return
    if prompt[:4].lower() == "new ":
        prompt = prompt[4:].strip()
        messages.clear()
    prompt = enrich_text_with_urls(prompt)

    who = "MiniMax Pro"
    reply_id = bot_reply_first(message, who, bot)
    messages.append({"role": "user", "content": prompt})
    if len(messages) > 10:
        del messages[:2]

    try:
        text, assistant_message = _stream(messages, reply_id, who, bot)
        if not text:
            messages.pop()
            bot_reply_markdown(reply_id, who, f"{who} did not answer.", bot)
            return
        if not bot_reply_markdown(reply_id, who, text, bot):
            messages.clear()
            return
        messages.append(assistant_message)
    except Exception:
        logger.exception("MiniMax Pro handler error")
        bot.reply_to(message, "answer wrong maybe up to the max token")
        messages.clear()


if client is not None:

    def register(bot: TeleBot) -> None:
        bot.register_message_handler(
            minimax_handler, commands=["minimax"], pass_bot=True
        )
        bot.register_message_handler(minimax_handler, regexp="^minimax:", pass_bot=True)
        bot.register_message_handler(
            minimax_pro_handler, commands=["minimax_pro"], pass_bot=True
        )
        bot.register_message_handler(
            minimax_pro_handler, regexp="^minimax_pro:", pass_bot=True
        )
