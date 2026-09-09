import re


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
        "занят", "репетиц", "сбор", "форм", "оплат", "абонем", "распис",
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
        "какими", "каком", "какого", "какую", "чей", "чья", "чьё",
        "чьи", "чьего", "чью", "кто", "кому", "кого",
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
