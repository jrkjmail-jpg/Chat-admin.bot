import asyncio
import base64
import json
import logging
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import CallbackQuery, ChatMemberUpdated, InlineKeyboardButton, InlineKeyboardMarkup, Message
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pypdf import PdfReader

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FALLBACK_TO_ADMIN = "Передаю ваш вопрос администратору."
ACTIVE_PARENT_STATUS = "✅ Активен как родительский чат"
MODERATION_ONLY_STATUS = "🛡 Активен только как модератор"
IGNORED_STATUS = "⏸ Не подключён"


@dataclass(frozen=True)
class Config:
    bot_token: str
    openai_api_key: str
    admin_ids: set[int]
    service_chat_id: int | None
    database_path: str
    openai_model: str
    embedding_model: str
    default_mode: str
    timezone: str
    studio_name: str
    studio_aliases: str
    admin_names: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_int_set(value: str | None) -> set[int]:
    result: set[int] = set()
    if not value:
        return result
    for item in value.split(","):
        item = item.strip()
        if item and item.lstrip("-").isdigit():
            result.add(int(item))
    return result


def load_config() -> Config:
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("BOT_TOKEN is required")
    return Config(
        bot_token=token,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        admin_ids=parse_int_set(os.getenv("ADMIN_IDS")),
        service_chat_id=int(os.getenv("SERVICE_CHAT_ID")) if os.getenv("SERVICE_CHAT_ID") else None,
        database_path=os.getenv("DATABASE_PATH", "/app/data/studio_admin.db"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        default_mode=os.getenv("BOT_DEFAULT_MODE", "outside_working_hours"),
        timezone=os.getenv("STUDIO_TIMEZONE", "Europe/Moscow"),
        studio_name=os.getenv("STUDIO_NAME", "Тодес Рязанский проспект").strip() or "Тодес Рязанский проспект",
        studio_aliases=os.getenv("STUDIO_ALIASES", "Тодес Рязанский проспект,TODES Рязанский проспект,Рязанский проспект,Тодес Рязанка,TODES Рязанка").strip(),
        admin_names=os.getenv("ADMIN_NAMES", "Даша,Дарья,Дарья Сергеевна").strip(),
    )


class Database:
    def __init__(self, path: str, default_mode: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.default_mode = default_mode
        self.init_schema()
        logger.info("Database connected: %s", self.path)

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chats (chat_id INTEGER PRIMARY KEY, type TEXT NOT NULL, title TEXT, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS knowledge_items (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, content TEXT NOT NULL, source_chat_id INTEGER, source_message_id INTEGER, embedding TEXT, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS pending_knowledge (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, content TEXT NOT NULL, source_chat_id INTEGER, source_message_id INTEGER, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS questions (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL, user_id INTEGER, message_id INTEGER, question TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS moderation_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL, user_id INTEGER, message_id INTEGER, reason TEXT NOT NULL, text TEXT, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS working_hours (weekday INTEGER PRIMARY KEY, start_time TEXT, end_time TEXT, enabled INTEGER NOT NULL DEFAULT 1);
            """
        )
        self.conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES('bot_mode', ?)", (self.default_mode,))
        defaults = {0:("16:00","22:00",1),1:("16:00","22:00",1),2:("16:00","22:00",1),3:("16:00","22:00",1),4:("16:00","22:00",1),5:("10:00","18:00",1),6:(None,None,0)}
        for weekday, values in defaults.items():
            self.conn.execute("INSERT OR IGNORE INTO working_hours(weekday, start_time, end_time, enabled) VALUES(?, ?, ?, ?)", (weekday, *values))
        self.conn.commit()

    def get_setting(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        self.conn.execute("INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
        self.conn.commit()

    def add_chat(self, chat_id: int, chat_type: str, title: str | None) -> None:
        self.conn.execute("INSERT INTO chats(chat_id, type, title, created_at) VALUES(?, ?, ?, ?) ON CONFLICT(chat_id) DO UPDATE SET type=excluded.type, title=excluded.title", (chat_id, chat_type, title, utc_now()))
        self.conn.commit()

    def get_chat_type(self, chat_id: int) -> str | None:
        row = self.conn.execute("SELECT type FROM chats WHERE chat_id = ?", (chat_id,)).fetchone()
        return row["type"] if row else None

    def list_active_chats(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM chats WHERE type IN ('parent', 'moderation') ORDER BY title").fetchall()

    def add_pending_knowledge(self, title: str, content: str, chat_id: int | None, message_id: int | None) -> int:
        cur = self.conn.execute("INSERT INTO pending_knowledge(title, content, source_chat_id, source_message_id, created_at) VALUES(?, ?, ?, ?, ?)", (title, content, chat_id, message_id, utc_now()))
        self.conn.commit()
        return int(cur.lastrowid)

    def get_pending_knowledge(self, item_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM pending_knowledge WHERE id = ?", (item_id,)).fetchone()

    def approve_knowledge(self, item_id: int, embedding: list[float] | None) -> bool:
        row = self.get_pending_knowledge(item_id)
        if not row:
            return False
        self.conn.execute("INSERT INTO knowledge_items(title, content, source_chat_id, source_message_id, embedding, created_at) VALUES(?, ?, ?, ?, ?, ?)", (row["title"], row["content"], row["source_chat_id"], row["source_message_id"], json.dumps(embedding) if embedding else None, utc_now()))
        self.conn.execute("DELETE FROM pending_knowledge WHERE id = ?", (item_id,))
        self.conn.commit()
        return True

    def reject_knowledge(self, item_id: int) -> None:
        self.conn.execute("DELETE FROM pending_knowledge WHERE id = ?", (item_id,))
        self.conn.commit()

    def list_knowledge_with_embeddings(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM knowledge_items WHERE embedding IS NOT NULL ORDER BY id DESC").fetchall()

    def list_recent_knowledge(self, limit: int = 8) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM knowledge_items ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    def save_question(self, chat_id: int, user_id: int | None, message_id: int, question: str, status: str) -> None:
        self.conn.execute("INSERT INTO questions(chat_id, user_id, message_id, question, status, created_at) VALUES(?, ?, ?, ?, ?, ?)", (chat_id, user_id, message_id, question, status, utc_now()))
        self.conn.commit()

    def save_moderation_log(self, chat_id: int, user_id: int | None, message_id: int, reason: str, text: str | None) -> None:
        self.conn.execute("INSERT INTO moderation_logs(chat_id, user_id, message_id, reason, text, created_at) VALUES(?, ?, ?, ?, ?, ?)", (chat_id, user_id, message_id, reason, text, utc_now()))
        self.conn.commit()

    def get_working_hours_text(self) -> str:
        names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
        rows = self.conn.execute("SELECT * FROM working_hours ORDER BY weekday").fetchall()
        return "\n".join(f"{names[r['weekday']]}: {r['start_time']}-{r['end_time']}" if r["enabled"] else f"{names[r['weekday']]}: выходной" for r in rows)

    def is_studio_open_now(self, tz_name: str) -> bool:
        now = datetime.now(ZoneInfo(tz_name))
        row = self.conn.execute("SELECT * FROM working_hours WHERE weekday = ?", (now.weekday(),)).fetchone()
        if not row or not row["enabled"] or not row["start_time"] or not row["end_time"]:
            return False
        sh, sm = map(int, row["start_time"].split(":")); eh, em = map(int, row["end_time"].split(":"))
        return time(sh, sm) <= now.time() <= time(eh, em)


class OpenAIService:
    def __init__(self, config: Config) -> None:
        self.enabled = bool(config.openai_api_key)
        self.client = AsyncOpenAI(api_key=config.openai_api_key) if config.openai_api_key else None
        self.config = config

    def now_text(self) -> str:
        now = datetime.now(ZoneInfo(self.config.timezone))
        weekdays = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
        return f"Сейчас: {now.strftime('%d.%m.%Y %H:%M')}, {weekdays[now.weekday()]}, часовой пояс {self.config.timezone}."

    async def embedding(self, text: str) -> list[float] | None:
        if not self.enabled or not self.client:
            return None
        response = await self.client.embeddings.create(model=self.config.embedding_model, input=text[:6000])
        return response.data[0].embedding

    async def summarize_knowledge(self, text: str) -> str:
        cleaned = normalize_text(text)
        if not self.enabled or not self.client:
            return cleaned[:3500]
        prompt = (
            f"Ты готовишь базу знаний только для студии {self.config.studio_name}. "
            f"Алиасы нужной студии: {self.config.studio_aliases}. {self.now_text()} "
            "Извлекай только факты, которые относятся к этой студии или к Рязанскому проспекту. "
            "Другие филиалы, студии и площадки игнорируй. Сохраняй даты, время, год, день недели, адреса, группы, форму, сборы, репетиции, оплату и правила. "
            "Если год в документе не указан, не выдумывай его. Не добавляй фактов, которых нет."
        )
        response = await self.client.chat.completions.create(model=self.config.openai_model, messages=[{"role":"system","content":prompt},{"role":"user","content":cleaned[:12000]}], temperature=0.1)
        return response.choices[0].message.content or cleaned[:3500]

    async def image_to_text(self, image_path: str) -> str:
        if not self.enabled or not self.client:
            return ""
        data = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
        response = await self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=[{"role":"user","content":[{"type":"text","text":"Извлеки весь полезный текст с изображения для базы знаний студии. Сохрани даты, время, группы, адреса, правила. Не додумывай."},{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{data}"}}]}],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    async def answer(self, question: str, context: str, chat_title: str | None) -> str:
        if not self.enabled or not self.client:
            return FALLBACK_TO_ADMIN
        system = (
            f"Ты отвечаешь от имени студии {self.config.studio_name}. Алиасы: {self.config.studio_aliases}. "
            f"Администраторы: {self.config.admin_names}. Текущий чат/группа: {chat_title or 'без названия'}. {self.now_text()} "
            "Отвечай только по этой студии. Другие студии, филиалы и площадки полностью игнорируй, даже если они есть в контексте. "
            "Никогда не предлагай выбрать другую студию. Если вопрос про дату без года, используй текущий год только как ориентир, но не утверждай его, если в базе нет года. "
            "Если в базе несколько групп нужной студии, выбери по названию текущего чата. "
            "Сначала максимально попробуй ответить сам по базе. Передавай администратору только если данных по нужной студии нет, они противоречивы или нужен личный ответ. "
            f"Если точного ответа нет, ответь ровно: {FALLBACK_TO_ADMIN}"
        )
        user = f"КОНТЕКСТ:\n{context}\n\nВОПРОС РОДИТЕЛЯ:\n{question}"
        response = await self.client.chat.completions.create(model=self.config.openai_model, messages=[{"role":"system","content":system},{"role":"user","content":user}], temperature=0.1)
        return (response.choices[0].message.content or "").strip() or FALLBACK_TO_ADMIN


BAD_WORDS = {"дурак", "идиот", "тупой", "тупая", "дебил", "лох"}
AD_PATTERNS = [re.compile(r"https?://", re.I), re.compile(r"t\.me/", re.I), re.compile(r"@\w{4,}", re.I), re.compile(r"скидк[аи]|акци[яи]|купите|заработок|подработка", re.I)]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_admin(user_id: int | None, config: Config) -> bool:
    return bool(user_id and user_id in config.admin_ids)


def should_moderate(text: str, user_is_admin: bool) -> str | None:
    lower = text.lower()
    if any(word in lower for word in BAD_WORDS): return "оскорбление"
    if not user_is_admin and any(pattern.search(text) for pattern in AD_PATTERNS): return "возможная реклама или ссылка"
    return None


def is_likely_question(text: str) -> bool:
    lower = text.lower().strip()
    words = ("когда", "где", "куда", "во сколько", "сколько", "можно", "надо", "нужно", "какая", "какой", "какие", "что", "как", "почему", "даша", "дарья")
    return "?" in lower or lower.startswith(words)


def knowledge_keyboard(item_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="✅ Добавить в базу", callback_data=f"kb:approve:{item_id}")],[InlineKeyboardButton(text="❌ Не добавлять", callback_data=f"kb:reject:{item_id}")]])


def chat_control_keyboard(chat_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🛡 Только модерация", callback_data=f"chat:moderation:{chat_id}")],[InlineKeyboardButton(text="⏸ Отключить чат", callback_data=f"chat:ignored:{chat_id}")],[InlineKeyboardButton(text="✅ Родительский чат", callback_data=f"chat:parent:{chat_id}")]])


async def download_to_temp(bot: Bot, file_id: str, suffix: str) -> str:
    tg_file = await bot.get_file(file_id)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        temp_path = tmp.name
    await bot.download_file(tg_file.file_path, destination=temp_path)
    return temp_path


async def extract_pdf_text(bot: Bot, file_id: str) -> str:
    temp_path = await download_to_temp(bot, file_id, ".pdf")
    try:
        reader = PdfReader(temp_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def find_relevant_context(db: Database, query_embedding: list[float] | None, limit: int = 6) -> str:
    recent = list(db.list_recent_knowledge(4))
    if not query_embedding:
        return "\n\n".join(row["content"] for row in recent)
    query = np.array(query_embedding, dtype=np.float32)
    scored = []
    for row in db.list_knowledge_with_embeddings():
        try:
            emb = np.array(json.loads(row["embedding"]), dtype=np.float32)
            score = float(np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb)))
            scored.append((score, row))
        except Exception as exc:
            logger.warning("Bad embedding for item %s: %s", row["id"], exc)
    scored.sort(reverse=True, key=lambda item: item[0])
    selected = [row for score, row in scored[:limit] if score >= 0.15]
    merged = []
    seen = set()
    for row in selected + recent:
        if row["id"] not in seen:
            seen.add(row["id"]); merged.append(row["content"])
    return "\n\n".join(merged)


def bot_is_active(db: Database, config: Config) -> bool:
    mode = db.get_setting("bot_mode", config.default_mode)
    if mode == "always": return True
    if mode == "off": return False
    studio_open = db.is_studio_open_now(config.timezone)
    return (mode == "outside_working_hours" and not studio_open) or (mode == "working_hours_only" and studio_open)


async def notify_admins(bot: Bot, config: Config, text: str, reply_markup: InlineKeyboardMarkup | None = None) -> None:
    for admin_id in config.admin_ids:
        try: await bot.send_message(admin_id, text, reply_markup=reply_markup)
        except Exception as exc: logger.warning("Cannot notify admin %s: %s", admin_id, exc)


async def send_startup_report(bot: Bot, db: Database, config: Config) -> None:
    chats = db.list_active_chats()
    if chats:
        lines = ["Бот перезапущен и подключён к сохранённым чатам:", ""]
        for row in chats:
            status = ACTIVE_PARENT_STATUS if row["type"] == "parent" else MODERATION_ONLY_STATUS
            lines.append(f"{status}: {row['title'] or row['chat_id']} ({row['chat_id']})")
        await notify_admins(bot, config, "\n".join(lines))
    else:
        await notify_admins(bot, config, "Бот перезапущен. Сохранённых родительских чатов пока нет. При первом сообщении из группы чат будет активирован автоматически.")


async def main() -> None:
    config = load_config(); db = Database(config.database_path, config.default_mode); ai = OpenAIService(config)
    bot = Bot(config.bot_token); me = await bot.get_me(); dp = Dispatcher(); router = Router()

    @router.message(Command("start"))
    async def start(message: Message) -> None:
        await message.answer(f"Здравствуйте! Я AI-администратор студии {config.studio_name}.")

    @router.my_chat_member()
    async def bot_added(event: ChatMemberUpdated) -> None:
        if event.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}: return
        if event.new_chat_member.status not in {"member", "administrator"}: return
        db.add_chat(event.chat.id, "parent", event.chat.title or "Без названия")
        await notify_admins(bot, config, f"Чат автоматически активирован.\n\nНазвание: {event.chat.title or 'Без названия'}\nChat ID: {event.chat.id}\nСтатус: {ACTIVE_PARENT_STATUS}", chat_control_keyboard(event.chat.id))

    @router.message(F.new_chat_members)
    async def delete_join_message(message: Message) -> None:
        if message.new_chat_members and any(member.id == me.id for member in message.new_chat_members):
            db.add_chat(message.chat.id, "parent", message.chat.title)
            try: await message.delete()
            except TelegramBadRequest: pass

    @router.message(Command("set_service_chat"))
    async def set_service_chat(message: Message) -> None:
        if is_admin(message.from_user.id if message.from_user else None, config):
            db.add_chat(message.chat.id, "service", message.chat.title); db.set_setting("service_chat_id", str(message.chat.id)); await message.answer("Этот чат назначен сервисным чатом базы знаний.")

    @router.message(Command("add_parent_chat"))
    async def add_parent_chat(message: Message) -> None:
        if is_admin(message.from_user.id if message.from_user else None, config): db.add_chat(message.chat.id, "parent", message.chat.title); await message.answer("Этот чат назначен родительским чатом студии.")

    @router.message(Command("mode"))
    async def set_mode(message: Message) -> None:
        if not is_admin(message.from_user.id if message.from_user else None, config): return
        parts = (message.text or "").split(maxsplit=1); allowed = {"always", "outside_working_hours", "working_hours_only", "off"}
        if len(parts) != 2 or parts[1] not in allowed: await message.answer("Используйте: /mode always|outside_working_hours|working_hours_only|off"); return
        db.set_setting("bot_mode", parts[1]); await message.answer(f"Режим работы бота изменён: {parts[1]}")

    @router.message(Command("hours"))
    async def hours(message: Message) -> None:
        if is_admin(message.from_user.id if message.from_user else None, config): await message.answer("Рабочее время студии:\n" + db.get_working_hours_text())

    @router.callback_query(F.data.startswith("chat:"))
    async def chat_callback(callback: CallbackQuery) -> None:
        if not is_admin(callback.from_user.id, config): await callback.answer("Недостаточно прав", show_alert=True); return
        _, action, raw_chat_id = callback.data.split(":"); chat_id = int(raw_chat_id)
        try: chat = await bot.get_chat(chat_id); title = chat.title or "Без названия"
        except Exception: title = "Без названия"
        chat_type = "parent" if action == "parent" else "moderation" if action == "moderation" else "ignored"
        status = ACTIVE_PARENT_STATUS if chat_type == "parent" else MODERATION_ONLY_STATUS if chat_type == "moderation" else IGNORED_STATUS
        db.add_chat(chat_id, chat_type, title); await callback.message.edit_text(f"Чат: {title}\nChat ID: {chat_id}\nСтатус: {status}"); await callback.answer("Сохранено")

    @router.callback_query(F.data.startswith("kb:"))
    async def knowledge_callback(callback: CallbackQuery) -> None:
        if not is_admin(callback.from_user.id, config): await callback.answer("Недостаточно прав", show_alert=True); return
        _, action, raw_id = callback.data.split(":"); item_id = int(raw_id)
        if action == "approve":
            row = db.get_pending_knowledge(item_id)
            if not row: await callback.answer("Материал уже обработан", show_alert=True); return
            db.approve_knowledge(item_id, await ai.embedding(row["content"])); await callback.message.edit_text("✅ Материал добавлен в базу знаний."); await callback.answer("Добавлено")
        else:
            db.reject_knowledge(item_id); await callback.message.edit_text("❌ Материал не добавлен."); await callback.answer("Отклонено")

    async def handle_knowledge_upload(message: Message) -> bool:
        user_id = message.from_user.id if message.from_user else None
        service_chat_id = db.get_setting("service_chat_id", "")
        is_private_admin = message.chat.type == ChatType.PRIVATE and is_admin(user_id, config)
        is_service = db.get_chat_type(message.chat.id) == "service" or str(message.chat.id) == service_chat_id or message.chat.id == config.service_chat_id
        if not is_private_admin and not is_service: return False
        raw_text = ""; title = "Материал"
        if message.text and not message.text.startswith("/"): raw_text = message.text; title = raw_text[:80]
        elif message.document and message.document.mime_type == "application/pdf": raw_text = await extract_pdf_text(bot, message.document.file_id); title = message.document.file_name or "PDF-документ"
        elif message.photo:
            path = await download_to_temp(bot, message.photo[-1].file_id, ".jpg")
            try: raw_text = await ai.image_to_text(path); title = "Скриншот/фото"
            finally: Path(path).unlink(missing_ok=True)
        else: await message.answer("Пока я могу добавлять в базу текст, PDF и изображения."); return True
        if not normalize_text(raw_text): await message.answer("Не удалось извлечь текст из материала."); return True
        summary = await ai.summarize_knowledge(raw_text)
        item_id = db.add_pending_knowledge(title, summary, message.chat.id, message.message_id)
        await message.answer(f"Я подготовил материал для базы знаний:\n\n{summary[:2500]}\n\nДобавить это в базу?", reply_markup=knowledge_keyboard(item_id))
        return True

    async def moderate_if_needed(message: Message) -> bool:
        text = message.text or message.caption or ""; user_id = message.from_user.id if message.from_user else None
        reason = should_moderate(text, is_admin(user_id, config)) if text else None
        if not reason: return False
        db.save_moderation_log(message.chat.id, user_id, message.message_id, reason, text)
        try: await message.delete()
        except TelegramBadRequest: pass
        await notify_admins(bot, config, f"Модерация: {reason}\nЧат: {message.chat.title or message.chat.id}\nТекст: {text[:500]}")
        return True

    @router.message()
    async def all_messages(message: Message) -> None:
        if await handle_knowledge_upload(message): return
        if message.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}: return
        chat_type = db.get_chat_type(message.chat.id)
        if chat_type is None:
            db.add_chat(message.chat.id, "parent", message.chat.title); chat_type = "parent"
            await notify_admins(bot, config, f"Чат автоматически активирован по первому сообщению.\n\nНазвание: {message.chat.title or 'Без названия'}\nChat ID: {message.chat.id}\nСтатус: {ACTIVE_PARENT_STATUS}", chat_control_keyboard(message.chat.id))
        if chat_type in {"parent", "moderation"} and await moderate_if_needed(message): return
        if chat_type != "parent": return
        text = message.text or message.caption or ""
        if not text or text.startswith("/") or not is_likely_question(text) or not bot_is_active(db, config): return
        search_text = f"{config.studio_name}\n{config.studio_aliases}\n{message.chat.title or ''}\n{text}\n{ai.now_text()}"
        context = find_relevant_context(db, await ai.embedding(search_text))
        if not context:
            db.save_question(message.chat.id, message.from_user.id if message.from_user else None, message.message_id, text, "sent_to_admin")
            await message.reply(FALLBACK_TO_ADMIN); await notify_admins(bot, config, f"Вопрос родителя:\n{text}\n\nЧат: {message.chat.title or message.chat.id}"); return
        answer = await ai.answer(text, context, message.chat.title)
        db.save_question(message.chat.id, message.from_user.id if message.from_user else None, message.message_id, text, "sent_to_admin" if answer == FALLBACK_TO_ADMIN else "answered")
        if answer == FALLBACK_TO_ADMIN: await notify_admins(bot, config, f"Вопрос родителя:\n{text}\n\nЧат: {message.chat.title or message.chat.id}")
        await message.reply(answer)

    dp.include_router(router); await send_startup_report(bot, db, config); await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
