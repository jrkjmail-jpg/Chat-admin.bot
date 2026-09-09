import asyncio
import base64
import json
import logging
import os
import re
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import xlrd
from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import BotCommand, CallbackQuery, ChatMemberUpdated, ForceReply, InlineKeyboardButton, InlineKeyboardMarkup, Message
from docx import Document as WordDocument
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openpyxl import load_workbook
from pypdf import PdfReader

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FALLBACK_TO_ADMIN = "Передаю ваш вопрос администратору."
ACTIVE_PARENT_STATUS = "✅ Активен как родительский чат"
MODERATION_ONLY_STATUS = "🛡 Активен только как модератор"
IGNORED_STATUS = "⏸ Не подключён"
MESSAGE_BUFFER_SECONDS = float(os.getenv("MESSAGE_BUFFER_SECONDS", "4"))
MAX_KNOWLEDGE_FILE_BYTES = int(os.getenv("MAX_KNOWLEDGE_FILE_MB", "10")) * 1024 * 1024
MAX_EXTRACTED_TEXT_CHARS = int(os.getenv("MAX_EXTRACTED_TEXT_CHARS", "50000"))
MAX_OFFICE_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MODE_LABELS = {
    "always": "🟢 Включён вручную",
    "outside_working_hours": "🕒 Автоматически после смены администратора",
    "working_hours_only": "☀️ Только во время смены администратора",
    "off": "🔴 Выключен вручную",
}
WEEKDAY_LABELS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
HOURS_PROMPT_PREFIX = "Настройка графика администратора — "


@dataclass(frozen=True)
class Config:
    bot_token: str
    openai_api_key: str
    admin_ids: set[int]
    service_chat_id: int | None
    database_path: str
    openai_model: str
    embedding_model: str
    transcription_model: str
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
    token = (os.getenv("TG_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("TG_BOT_TOKEN or BOT_TOKEN is required")
    return Config(
        bot_token=token,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        admin_ids=parse_int_set(os.getenv("ADMIN_IDS")),
        service_chat_id=int(os.getenv("SERVICE_CHAT_ID")) if os.getenv("SERVICE_CHAT_ID") else None,
        database_path=os.getenv("DATABASE_PATH", "/app/data/studio_admin.db"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        transcription_model=os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"),
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
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS chats (chat_id INTEGER PRIMARY KEY, type TEXT NOT NULL, title TEXT, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS knowledge_items (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, content TEXT NOT NULL, source_chat_id INTEGER, source_message_id INTEGER, embedding TEXT, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS pending_knowledge (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, content TEXT NOT NULL, source_chat_id INTEGER, source_message_id INTEGER, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS questions (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL, user_id INTEGER, message_id INTEGER, question TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS moderation_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL, user_id INTEGER, message_id INTEGER, reason TEXT NOT NULL, text TEXT, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS working_hours (weekday INTEGER PRIMARY KEY, start_time TEXT, end_time TEXT, enabled INTEGER NOT NULL DEFAULT 1);
        """)
        self.conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES('bot_mode', ?)", (self.default_mode,))
        defaults = {0: ("16:00", "22:00", 1), 1: ("16:00", "22:00", 1), 2: ("16:00", "22:00", 1), 3: ("16:00", "22:00", 1), 4: ("16:00", "22:00", 1), 5: ("10:00", "18:00", 1), 6: (None, None, 0)}
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

    def list_working_hours(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM working_hours ORDER BY weekday").fetchall()

    def get_working_hours_text(self) -> str:
        rows = self.list_working_hours()
        return "\n".join(
            f"{WEEKDAY_LABELS[row['weekday']]}: {row['start_time']}-{row['end_time']}"
            if row["enabled"]
            else f"{WEEKDAY_LABELS[row['weekday']]}: выходной"
            for row in rows
        )

    def set_working_hours(self, weekday: int, start_time: str | None, end_time: str | None, enabled: bool) -> None:
        self.conn.execute(
            "UPDATE working_hours SET start_time = ?, end_time = ?, enabled = ? WHERE weekday = ?",
            (start_time, end_time, int(enabled), weekday),
        )
        self.conn.commit()

    def is_admin_working_now(self, tz_name: str) -> bool:
        now = datetime.now(ZoneInfo(tz_name))
        row = self.conn.execute("SELECT * FROM working_hours WHERE weekday = ?", (now.weekday(),)).fetchone()
        if not row or not row["enabled"] or not row["start_time"] or not row["end_time"]:
            return False
        sh, sm = map(int, row["start_time"].split(":"))
        eh, em = map(int, row["end_time"].split(":"))
        return time(sh, sm) <= now.time() < time(eh, em)


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

    async def transcribe_audio(self, audio_path: str) -> str:
        if not self.enabled or not self.client:
            return ""
        with open(audio_path, "rb") as audio_file:
            response = await self.client.audio.transcriptions.create(model=self.config.transcription_model, file=audio_file, language="ru")
        return getattr(response, "text", "") or ""

    async def summarize_knowledge(self, text: str) -> str:
        cleaned = normalize_text(text)
        if not self.enabled or not self.client:
            return cleaned[:3500]
        system = f"Ты готовишь базу знаний только для студии {self.config.studio_name}. Сохраняй точные даты, время, адреса, группы, форму, оплату, правила. Не добавляй фактов."
        response = await self.client.chat.completions.create(model=self.config.openai_model, messages=[{"role": "system", "content": system}, {"role": "user", "content": cleaned[:12000]}], temperature=0.1)
        return response.choices[0].message.content or cleaned[:3500]

    async def image_to_text(self, image_path: str) -> str:
        if not self.enabled or not self.client:
            return ""
        data = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
        response = await self.client.chat.completions.create(model=self.config.openai_model, messages=[{"role": "user", "content": [{"type": "text", "text": "Извлеки видимый текст с изображения для базы знаний. Не додумывай."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}]}], temperature=0.1)
        return response.choices[0].message.content or ""

    async def answer_from_context(self, question: str, context: str, chat_title: str | None) -> str:
        if not self.enabled or not self.client:
            return FALLBACK_TO_ADMIN
        system = f"Ты отвечаешь от имени студии {self.config.studio_name}. {self.now_text()} Отвечай только на поставленный вопрос. Не рассказывай всё, что знаешь. Другие филиалы игнорируй. Если точного ответа нет, ответь ровно: {FALLBACK_TO_ADMIN}"
        user = f"КОНТЕКСТ:\n{context}\n\nВОПРОС РОДИТЕЛЯ:\n{question}"
        response = await self.client.chat.completions.create(model=self.config.openai_model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.1)
        return (response.choices[0].message.content or "").strip() or FALLBACK_TO_ADMIN


AD_PATTERNS = [re.compile(r"https?://", re.I), re.compile(r"t\.me/", re.I), re.compile(r"@\w{4,}", re.I), re.compile(r"скидк[аи]|акци[яи]|купите|заработок|подработка", re.I)]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def join_message_parts(parts: list[str]) -> str:
    result = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in {".", ",", "?", "!", ":", ";", "…"}:
            result = result.rstrip() + part
        elif not result:
            result = part
        else:
            result += " " + part
    return normalize_text(result)


def classify_message(text: str) -> str:
    lower = text.lower().strip()
    admin_markers = [
        "жалоб", "не соглас", "разбер", "лично", "индивидуально",
        "возврат", "верните деньги",
    ]
    if any(marker in lower for marker in admin_markers):
        return "admin_required"

    parent_coordination_markers = [
        "у кого", "кто может", "кто сможет", "девочки", "родители",
        "кто едет", "кто идёт", "кто идет", "кто забер",
    ]
    if any(marker in lower for marker in parent_coordination_markers):
        return "ignore"

    studio_words = [
        "занят", "репетиц", "сбор", "форма", "оплат", "абонем", "распис",
        "концерт", "кубок", "турнир", "педагог", "студ", "админ",
        "даша", "дарья", "проспект", "зал", "адрес", "групп", "договор",
        "соглашен", "документ", "справк", "заявлен", "анкет", "правил",
        "услов", "стоим", "цен", "реквизит", "срок", "каникул", "пропуск",
        "отработ", "замен", "перенос", "болезн", "медицин", "выступ",
        "костюм", "обув", "контакт", "связ", "взнос", "долг",
    ]
    question_starts = [
        "когда", "где", "куда", "во сколько", "со скольки", "до скольки",
        "сколько", "можно", "надо", "нужно", "какая", "какой", "какие",
        "что", "как", "почему", "кто", "есть ли", "имеется ли", "к кому",
        "на когда",
    ]
    interrogative_words = {
        "когда", "где", "куда", "откуда", "сколько", "что", "чего", "зачем",
        "почему", "как", "какой", "какая", "какое", "какие", "каким",
        "какими", "каком", "чей", "чья", "чьё", "чьи", "кто", "кому", "кого",
    }
    leading_words = re.findall(r"[а-яё]+", lower)[:5]

    request_patterns = [
        r"\bподскаж(?:и|ите)\b",
        r"\bскаж(?:и|ите)\b",
        r"\bда(?:й|йте)\b",
        r"\bрасскаж(?:и|ите)\b",
        r"\bнапиш(?:и|ите)\b",
        r"\bуточн(?:и|ите)\b",
        r"\bпоясн(?:и|ите)\b",
        r"\bпришл(?:и|ите)\b",
        r"\bсообщ(?:и|ите)\b",
        r"\bпокаж(?:и|ите)\b",
        r"\bнапомн(?:и|ите)\b",
        r"\bхочу\s+(?:узнать|уточнить|понять)\b",
        r"\bинтересует\b",
        r"\bнужн(?:а|о|ы)\s+(?:информац|инф|данн)",
        r"\b(?:есть|имеется)\s+(?:ли\s+)?(?:информац|инфа|данные)",
    ]
    is_request = (
        "?" in lower
        or any(lower.startswith(word) for word in question_starts)
        or any(word in interrogative_words for word in leading_words)
        or any(re.search(pattern, lower) for pattern in request_patterns)
    )
    is_studio_related = any(word in lower for word in studio_words) or "у нас" in lower
    if is_request and is_studio_related:
        return "studio_question"
    if is_request:
        return "admin_required"
    return "ignore"


def is_admin(user_id: int | None, config: Config) -> bool:
    return bool(user_id and user_id in config.admin_ids)


def should_moderate(text: str, user_is_admin: bool) -> str | None:
    if not user_is_admin and any(pattern.search(text) for pattern in AD_PATTERNS):
        return "возможная реклама или ссылка"
    return None


def knowledge_keyboard(item_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="✅ Добавить в базу", callback_data=f"kb:approve:{item_id}")], [InlineKeyboardButton(text="❌ Не добавлять", callback_data=f"kb:reject:{item_id}")]])


def chat_control_keyboard(chat_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🛡 Только модерация", callback_data=f"chat:moderation:{chat_id}")], [InlineKeyboardButton(text="⏸ Отключить чат", callback_data=f"chat:ignored:{chat_id}")], [InlineKeyboardButton(text="✅ Родительский чат", callback_data=f"chat:parent:{chat_id}")]])


async def download_to_temp(bot: Bot, file_id: str, suffix: str) -> str:
    tg_file = await bot.get_file(file_id)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        temp_path = tmp.name
    await bot.download_file(tg_file.file_path, destination=temp_path)
    return temp_path


def limited_text(lines: list[str]) -> str:
    selected: list[str] = []
    total = 0
    for raw_line in lines:
        line = str(raw_line).strip()
        if not line:
            continue
        remaining = MAX_EXTRACTED_TEXT_CHARS - total
        if remaining <= 0:
            break
        selected.append(line[:remaining])
        total += len(selected[-1]) + 1
    return "\n".join(selected)


def validate_office_archive(path: str) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            total_size = sum(member.file_size for member in members)
            if len(members) > 5000 or total_size > MAX_OFFICE_UNCOMPRESSED_BYTES:
                raise ValueError("Office document is too large after unpacking")
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid Office document") from exc


def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    return limited_text([page.extract_text() or "" for page in reader.pages])


def extract_docx_text(path: str) -> str:
    validate_office_archive(path)
    document = WordDocument(path)
    lines = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    for table_index, table in enumerate(document.tables, start=1):
        lines.append(f"Таблица {table_index}:")
        for row in table.rows:
            values = [normalize_text(cell.text) for cell in row.cells]
            if any(values):
                lines.append(" | ".join(values))
    return limited_text(lines)


def extract_xlsx_text(path: str) -> str:
    validate_office_archive(path)
    workbook = load_workbook(path, read_only=True, data_only=True)
    lines: list[str] = []
    try:
        for worksheet in workbook.worksheets:
            lines.append(f"Лист: {worksheet.title}")
            for row in worksheet.iter_rows(values_only=True):
                values = [str(value).strip() if value is not None else "" for value in row]
                if any(values):
                    lines.append(" | ".join(values))
                if sum(len(line) + 1 for line in lines) >= MAX_EXTRACTED_TEXT_CHARS:
                    return limited_text(lines)
    finally:
        workbook.close()
    return limited_text(lines)


def extract_xls_text(path: str) -> str:
    workbook = xlrd.open_workbook(path, on_demand=True)
    lines: list[str] = []
    try:
        for worksheet in workbook.sheets():
            lines.append(f"Лист: {worksheet.name}")
            for row_index in range(worksheet.nrows):
                values = [str(worksheet.cell_value(row_index, column)).strip() for column in range(worksheet.ncols)]
                if any(values):
                    lines.append(" | ".join(values))
                if sum(len(line) + 1 for line in lines) >= MAX_EXTRACTED_TEXT_CHARS:
                    return limited_text(lines)
    finally:
        workbook.release_resources()
    return limited_text(lines)


def document_kind(file_name: str | None, mime_type: str | None) -> str | None:
    suffix = Path(file_name or "").suffix.lower()
    by_suffix = {".pdf": "pdf", ".docx": "docx", ".xlsx": "xlsx", ".xls": "xls"}
    if suffix in by_suffix:
        return by_suffix[suffix]
    by_mime = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        "application/vnd.ms-excel": "xls",
    }
    return by_mime.get(mime_type or "")


async def extract_uploaded_document(bot: Bot, file_id: str, file_name: str | None, mime_type: str | None) -> str:
    kind = document_kind(file_name, mime_type)
    if not kind:
        return ""
    temp_path = await download_to_temp(bot, file_id, f".{kind}")
    try:
        if kind == "pdf":
            return extract_pdf_text(temp_path)
        if kind == "docx":
            return extract_docx_text(temp_path)
        if kind == "xlsx":
            return extract_xlsx_text(temp_path)
        return extract_xls_text(temp_path)
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
            seen.add(row["id"])
            merged.append(row["content"])
    return "\n\n".join(merged)


def bot_is_active(db: Database, config: Config) -> bool:
    mode = db.get_setting("bot_mode", config.default_mode)
    if mode == "always":
        return True
    if mode == "off":
        return False
    admin_working = db.is_admin_working_now(config.timezone)
    if mode == "outside_working_hours":
        return not admin_working
    if mode == "working_hours_only":
        return admin_working
    return True


def mode_menu_keyboard(current_mode: str) -> InlineKeyboardMarkup:
    def mode_button(mode: str, label: str) -> InlineKeyboardButton:
        prefix = "✅ " if current_mode == mode else ""
        return InlineKeyboardButton(text=prefix + label, callback_data=f"mode:{mode}")

    return InlineKeyboardMarkup(inline_keyboard=[
        [mode_button("always", "🟢 Включить сейчас")],
        [mode_button("outside_working_hours", "🕒 Включаться после смены")],
        [mode_button("off", "🔴 Выключить сейчас")],
        [InlineKeyboardButton(text="⚙️ Настроить график администратора", callback_data="menu:hours")],
    ])


def working_hours_keyboard(db: Database) -> InlineKeyboardMarkup:
    rows = []
    for row in db.list_working_hours():
        day = WEEKDAY_LABELS[row["weekday"]]
        hours = f"{row['start_time']}-{row['end_time']}" if row["enabled"] else "выходной"
        rows.append([InlineKeyboardButton(text=f"{day}: {hours}", callback_data=f"hours:edit:{row['weekday']}")])
    rows.append([InlineKeyboardButton(text="⬅️ Назад в меню", callback_data="menu:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def working_hours_menu_text(db: Database) -> str:
    return (
        "⚙️ Рабочее время живого администратора\n\n"
        + db.get_working_hours_text()
        + "\n\nВ автоматическом режиме бот молчит во время смены "
        "и включается сразу после её окончания. Нажмите на день, чтобы изменить время."
    )


def parse_working_hours(value: str) -> tuple[str | None, str | None, bool] | None:
    cleaned = value.strip().lower().replace("—", "-").replace("–", "-")
    if cleaned in {"выходной", "нет", "off"}:
        return None, None, False
    match = re.fullmatch(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})", cleaned)
    if not match:
        return None
    sh, sm, eh, em = map(int, match.groups())
    if sh > 23 or eh > 23 or sm > 59 or em > 59:
        return None
    start_value = time(sh, sm)
    end_value = time(eh, em)
    if start_value >= end_value:
        return None
    return start_value.strftime("%H:%M"), end_value.strftime("%H:%M"), True


def mode_menu_text(db: Database, config: Config) -> str:
    current_mode = db.get_setting("bot_mode", config.default_mode)
    current_label = MODE_LABELS.get(current_mode, current_mode)
    admin_status = "на рабочем месте" if db.is_admin_working_now(config.timezone) else "не на рабочем месте"
    reply_status = "отвечает родителям" if bot_is_active(db, config) else "не отвечает родителям"
    return (
        "⚙️ Меню AI-администратора\n\n"
        f"Режим: {current_label}\n"
        f"По графику администратор сейчас: {admin_status}.\n"
        f"Бот сейчас: {reply_status}.\n\n"
        "Можно включить или выключить бота вручную либо выбрать автоматическую работу после смены."
    )


async def notify_admins(bot: Bot, config: Config, text: str, reply_markup: InlineKeyboardMarkup | None = None) -> None:
    for admin_id in config.admin_ids:
        try:
            await bot.send_message(admin_id, text, reply_markup=reply_markup)
        except Exception as exc:
            logger.warning("Cannot notify admin %s: %s", admin_id, exc)


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
    config = load_config()
    db = Database(config.database_path, config.default_mode)
    ai = OpenAIService(config)
    bot = Bot(config.bot_token)
    me = await bot.get_me()
    await bot.set_my_commands([
        BotCommand(command="start", description="Запустить бота"),
        BotCommand(command="menu", description="Открыть меню администратора"),
        BotCommand(command="hours", description="Настроить график администратора"),
    ])
    dp = Dispatcher()
    router = Router()
    message_buffers: dict[tuple[int, int], dict[str, object]] = {}

    @router.message(Command("start"))
    async def start(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else None
        if is_admin(user_id, config):
            current_mode = db.get_setting("bot_mode", config.default_mode)
            await message.answer(
                mode_menu_text(db, config),
                reply_markup=mode_menu_keyboard(current_mode),
            )
            return
        await message.answer(f"Здравствуйте! Я AI-администратор студии {config.studio_name}.")

    @router.message(Command("menu"))
    async def admin_menu(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else None
        if not is_admin(user_id, config):
            await message.answer("Меню управления доступно только администраторам.")
            return
        current_mode = db.get_setting("bot_mode", config.default_mode)
        await message.answer(
            mode_menu_text(db, config),
            reply_markup=mode_menu_keyboard(current_mode),
        )

    @router.my_chat_member()
    async def bot_added(event: ChatMemberUpdated) -> None:
        if event.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
            return
        if event.new_chat_member.status not in {"member", "administrator"}:
            return
        db.add_chat(event.chat.id, "parent", event.chat.title or "Без названия")
        await notify_admins(bot, config, f"Чат автоматически активирован.\n\nНазвание: {event.chat.title or 'Без названия'}\nChat ID: {event.chat.id}\nСтатус: {ACTIVE_PARENT_STATUS}", chat_control_keyboard(event.chat.id))

    @router.message(F.new_chat_members)
    async def delete_join_message(message: Message) -> None:
        if message.new_chat_members and any(member.id == me.id for member in message.new_chat_members):
            db.add_chat(message.chat.id, "parent", message.chat.title)
            try:
                await message.delete()
            except TelegramBadRequest:
                pass

    @router.message(Command("set_service_chat"))
    async def set_service_chat(message: Message) -> None:
        if is_admin(message.from_user.id if message.from_user else None, config):
            db.add_chat(message.chat.id, "service", message.chat.title)
            db.set_setting("service_chat_id", str(message.chat.id))
            await message.answer("Этот чат назначен сервисным чатом базы знаний.")

    @router.message(Command("add_parent_chat"))
    async def add_parent_chat(message: Message) -> None:
        if is_admin(message.from_user.id if message.from_user else None, config):
            db.add_chat(message.chat.id, "parent", message.chat.title)
            await message.answer("Этот чат назначен родительским чатом студии.")

    @router.message(Command("mode"))
    async def set_mode(message: Message) -> None:
        if not is_admin(message.from_user.id if message.from_user else None, config):
            return
        parts = (message.text or "").split(maxsplit=1)
        if len(parts) == 1:
            current_mode = db.get_setting("bot_mode", config.default_mode)
            await message.answer(
                mode_menu_text(db, config),
                reply_markup=mode_menu_keyboard(current_mode),
            )
            return
        if parts[1] not in MODE_LABELS:
            await message.answer("Неизвестный режим. Откройте /menu и выберите режим кнопкой.")
            return
        db.set_setting("bot_mode", parts[1])
        await message.answer(
            mode_menu_text(db, config),
            reply_markup=mode_menu_keyboard(parts[1]),
        )

    @router.message(Command("hours"))
    async def hours(message: Message) -> None:
        if not is_admin(message.from_user.id if message.from_user else None, config):
            return
        await message.answer(
            working_hours_menu_text(db),
            reply_markup=working_hours_keyboard(db),
        )

    @router.message(F.reply_to_message.text.startswith(HOURS_PROMPT_PREFIX))
    async def save_working_hours(message: Message) -> None:
        if not is_admin(message.from_user.id if message.from_user else None, config):
            return
        prompt_text = message.reply_to_message.text or ""
        first_line = prompt_text.splitlines()[0]
        day_name = first_line.removeprefix(HOURS_PROMPT_PREFIX).strip()
        if day_name not in WEEKDAY_LABELS:
            await message.answer("Не удалось определить день недели. Откройте /hours и попробуйте снова.")
            return
        parsed = parse_working_hours(message.text or "")
        if parsed is None:
            await message.answer(
                "Неверный формат. Ответьте временем, например 09:00-18:00, "
                "или словом «выходной»."
            )
            return
        start_time, end_time, enabled = parsed
        weekday = WEEKDAY_LABELS.index(day_name)
        db.set_working_hours(weekday, start_time, end_time, enabled)
        saved_value = f"{start_time}-{end_time}" if enabled else "выходной"
        await message.answer(
            f"✅ {day_name}: {saved_value}\n\n" + working_hours_menu_text(db),
            reply_markup=working_hours_keyboard(db),
        )

    @router.callback_query(F.data.startswith("mode:"))
    async def mode_callback(callback: CallbackQuery) -> None:
        if not is_admin(callback.from_user.id, config):
            await callback.answer("Недостаточно прав", show_alert=True)
            return
        _, selected_mode = callback.data.split(":", maxsplit=1)
        if selected_mode not in MODE_LABELS:
            await callback.answer("Неизвестный режим", show_alert=True)
            return
        current_mode = db.get_setting("bot_mode", config.default_mode)
        if selected_mode == current_mode:
            await callback.answer("Этот режим уже включён")
            return
        db.set_setting("bot_mode", selected_mode)
        await callback.message.edit_text(
            mode_menu_text(db, config),
            reply_markup=mode_menu_keyboard(selected_mode),
        )
        await callback.answer("Режим изменён")

    @router.callback_query(F.data == "menu:hours")
    async def menu_hours(callback: CallbackQuery) -> None:
        if not is_admin(callback.from_user.id, config):
            await callback.answer("Недостаточно прав", show_alert=True)
            return
        await callback.message.edit_text(
            working_hours_menu_text(db),
            reply_markup=working_hours_keyboard(db),
        )
        await callback.answer()

    @router.callback_query(F.data == "menu:back")
    async def menu_back(callback: CallbackQuery) -> None:
        if not is_admin(callback.from_user.id, config):
            await callback.answer("Недостаточно прав", show_alert=True)
            return
        current_mode = db.get_setting("bot_mode", config.default_mode)
        await callback.message.edit_text(
            mode_menu_text(db, config),
            reply_markup=mode_menu_keyboard(current_mode),
        )
        await callback.answer()

    @router.callback_query(F.data.startswith("hours:edit:"))
    async def edit_working_hours(callback: CallbackQuery) -> None:
        if not is_admin(callback.from_user.id, config):
            await callback.answer("Недостаточно прав", show_alert=True)
            return
        try:
            weekday = int(callback.data.rsplit(":", maxsplit=1)[1])
        except (TypeError, ValueError):
            await callback.answer("Некорректный день", show_alert=True)
            return
        if weekday not in range(7):
            await callback.answer("Некорректный день", show_alert=True)
            return
        day_name = WEEKDAY_LABELS[weekday]
        await callback.message.answer(
            f"{HOURS_PROMPT_PREFIX}{day_name}\n"
            "Введите интервал в формате 09:00-18:00. "
            "Если администратор не работает в этот день, напишите «выходной».",
            reply_markup=ForceReply(
                selective=True,
                input_field_placeholder="Например: 09:00-18:00",
            ),
        )
        await callback.answer()

    @router.callback_query(F.data.startswith("chat:"))
    async def chat_callback(callback: CallbackQuery) -> None:
        if not is_admin(callback.from_user.id, config):
            await callback.answer("Недостаточно прав", show_alert=True)
            return
        _, action, raw_chat_id = callback.data.split(":")
        chat_id = int(raw_chat_id)
        try:
            chat = await bot.get_chat(chat_id)
            title = chat.title or "Без названия"
        except Exception:
            title = "Без названия"
        chat_type = "parent" if action == "parent" else "moderation" if action == "moderation" else "ignored"
        status = ACTIVE_PARENT_STATUS if chat_type == "parent" else MODERATION_ONLY_STATUS if chat_type == "moderation" else IGNORED_STATUS
        db.add_chat(chat_id, chat_type, title)
        await callback.message.edit_text(f"Чат: {title}\nChat ID: {chat_id}\nСтатус: {status}")
        await callback.answer("Сохранено")

    @router.callback_query(F.data.startswith("kb:"))
    async def knowledge_callback(callback: CallbackQuery) -> None:
        if not is_admin(callback.from_user.id, config):
            await callback.answer("Недостаточно прав", show_alert=True)
            return
        _, action, raw_id = callback.data.split(":")
        item_id = int(raw_id)
        if action == "approve":
            row = db.get_pending_knowledge(item_id)
            if not row:
                await callback.answer("Материал уже обработан", show_alert=True)
                return
            db.approve_knowledge(item_id, await ai.embedding(row["content"]))
            await callback.message.edit_text("✅ Материал добавлен в базу знаний.")
            await callback.answer("Добавлено")
        else:
            db.reject_knowledge(item_id)
            await callback.message.edit_text("❌ Материал не добавлен.")
            await callback.answer("Отклонено")

    async def handle_knowledge_upload(message: Message) -> bool:
        user_id = message.from_user.id if message.from_user else None
        service_chat_id = db.get_setting("service_chat_id", "")
        is_private_admin = message.chat.type == ChatType.PRIVATE and is_admin(user_id, config)
        is_service = db.get_chat_type(message.chat.id) == "service" or str(message.chat.id) == service_chat_id or message.chat.id == config.service_chat_id
        if not is_private_admin and not is_service:
            return False
        raw_text = ""
        title = "Материал"
        if message.text and not message.text.startswith("/"):
            raw_text = message.text
            title = raw_text[:80]
        elif message.document:
            kind = document_kind(message.document.file_name, message.document.mime_type)
            if not kind:
                await message.answer(
                    "Поддерживаются документы PDF, Word (.docx) и Excel (.xlsx, .xls)."
                )
                return True
            if message.document.file_size and message.document.file_size > MAX_KNOWLEDGE_FILE_BYTES:
                limit_mb = MAX_KNOWLEDGE_FILE_BYTES // (1024 * 1024)
                await message.answer(f"Файл слишком большой. Максимальный размер — {limit_mb} МБ.")
                return True
            try:
                raw_text = await extract_uploaded_document(
                    bot,
                    message.document.file_id,
                    message.document.file_name,
                    message.document.mime_type,
                )
            except Exception as exc:
                logger.exception("Cannot extract document %s: %s", message.document.file_name, exc)
                await message.answer(
                    "Не удалось прочитать документ. Проверьте, что файл не повреждён "
                    "и сохранён в формате PDF, DOCX, XLSX или XLS."
                )
                return True
            title = message.document.file_name or f"{kind.upper()}-документ"
        elif message.photo:
            path = await download_to_temp(bot, message.photo[-1].file_id, ".jpg")
            try:
                raw_text = await ai.image_to_text(path)
                title = "Скриншот/фото"
            finally:
                Path(path).unlink(missing_ok=True)
        elif message.voice:
            path = await download_to_temp(bot, message.voice.file_id, ".ogg")
            try:
                raw_text = await ai.transcribe_audio(path)
                title = "Голосовое сообщение"
            finally:
                Path(path).unlink(missing_ok=True)
        else:
            await message.answer(
                "Можно добавлять текст, PDF, Word (.docx), Excel (.xlsx, .xls), "
                "изображения и голосовые сообщения."
            )
            return True
        if not normalize_text(raw_text):
            await message.answer("Не удалось извлечь текст из материала.")
            return True
        summary = await ai.summarize_knowledge(raw_text)
        item_id = db.add_pending_knowledge(title, summary, message.chat.id, message.message_id)
        await message.answer(f"Я подготовил материал для базы знаний:\n\n{summary[:2500]}\n\nДобавить это в базу?", reply_markup=knowledge_keyboard(item_id))
        return True

    async def moderate_if_needed(message: Message) -> bool:
        text = message.text or message.caption or ""
        user_id = message.from_user.id if message.from_user else None
        reason = should_moderate(text, is_admin(user_id, config)) if text else None
        if not reason:
            return False
        db.save_moderation_log(message.chat.id, user_id, message.message_id, reason, text)
        try:
            await message.delete()
        except TelegramBadRequest:
            pass
        await notify_admins(bot, config, f"Модерация: {reason}\nЧат: {message.chat.title or message.chat.id}\nТекст: {text[:500]}")
        return True

    async def extract_parent_message_text(message: Message) -> str:
        parts: list[str] = []
        if message.text:
            parts.append(message.text)
        if message.caption:
            parts.append(message.caption)
        # Фото в родительских чатах намеренно не анализируем.
        # Фото и скриншоты используются только в личке админа или сервисном чате для базы знаний.
        if message.voice:
            path = await download_to_temp(bot, message.voice.file_id, ".ogg")
            try:
                voice_text = await ai.transcribe_audio(path)
                if voice_text:
                    parts.append(f"[Голосовое сообщение: {voice_text}]")
            finally:
                Path(path).unlink(missing_ok=True)
        return join_message_parts(parts)

    async def process_parent_message(message: Message, text: str) -> None:
        if not text or text.startswith("/") or not bot_is_active(db, config):
            return
        kind = classify_message(text)
        logger.info("Message kind=%s chat=%s text=%s", kind, message.chat.id, text[:120])
        if kind == "ignore":
            return
        if kind == "admin_required":
            db.save_question(message.chat.id, message.from_user.id if message.from_user else None, message.message_id, text, "waiting_admin")
            await message.reply(FALLBACK_TO_ADMIN)
            await notify_admins(bot, config, f"Вопрос/ситуация для администратора:\n{text}\n\nЧат: {message.chat.title or message.chat.id}")
            return
        search_text = f"{config.studio_name}\n{config.studio_aliases}\n{message.chat.title or ''}\n{text}\n{ai.now_text()}"
        context = find_relevant_context(db, await ai.embedding(search_text))
        if not context:
            db.save_question(message.chat.id, message.from_user.id if message.from_user else None, message.message_id, text, "waiting_admin")
            await message.reply(FALLBACK_TO_ADMIN)
            await notify_admins(bot, config, f"Вопрос родителя:\n{text}\n\nЧат: {message.chat.title or message.chat.id}")
            return
        answer = await ai.answer_from_context(text, context, message.chat.title)
        status = "waiting_admin" if answer == FALLBACK_TO_ADMIN else "answered"
        db.save_question(message.chat.id, message.from_user.id if message.from_user else None, message.message_id, text, status)
        if answer == FALLBACK_TO_ADMIN:
            await notify_admins(bot, config, f"Вопрос родителя:\n{text}\n\nЧат: {message.chat.title or message.chat.id}")
        await message.reply(answer)

    async def flush_message_buffer(key: tuple[int, int]) -> None:
        await asyncio.sleep(MESSAGE_BUFFER_SECONDS)
        entry = message_buffers.pop(key, None)
        if not entry:
            return
        text = join_message_parts(entry["parts"])
        message = entry["message"]
        await process_parent_message(message, text)

    @router.message()
    async def all_messages(message: Message) -> None:
        if await handle_knowledge_upload(message):
            return
        if message.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
            return
        chat_type = db.get_chat_type(message.chat.id)
        if chat_type is None:
            db.add_chat(message.chat.id, "parent", message.chat.title)
            chat_type = "parent"
            await notify_admins(bot, config, f"Чат автоматически активирован по первому сообщению.\n\nНазвание: {message.chat.title or 'Без названия'}\nChat ID: {message.chat.id}\nСтатус: {ACTIVE_PARENT_STATUS}", chat_control_keyboard(message.chat.id))
        if chat_type in {"parent", "moderation"} and await moderate_if_needed(message):
            return
        if chat_type != "parent":
            return
        text = await extract_parent_message_text(message)
        if not text or text.startswith("/"):
            return
        user_id = message.from_user.id if message.from_user else 0
        key = (message.chat.id, user_id)
        old = message_buffers.get(key)
        if old and old.get("task"):
            old["task"].cancel()
        parts = list(old["parts"]) if old else []
        parts.append(text)
        task = asyncio.create_task(flush_message_buffer(key))
        message_buffers[key] = {"parts": parts, "message": message, "task": task}

    dp.include_router(router)
    await send_startup_report(bot, db, config)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
