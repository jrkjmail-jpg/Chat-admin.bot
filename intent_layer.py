def classify_message(text: str) -> str:
    lower = text.lower().strip()

    image_markers = [
        '[изображение', 'фото', 'картин', 'на фото', 'что это', 'что на', 'скрин'
    ]
    if any(marker in lower for marker in image_markers):
        return 'image_question'

    admin_markers = [
        'жалоб', 'не соглас', 'разбер', 'лично', 'индивидуально', 'возврат'
    ]
    if any(marker in lower for marker in admin_markers):
        return 'admin_required'

    studio_words = [
        'занят', 'репетиц', 'сбор', 'форма', 'оплат', 'абонем', 'распис',
        'концерт', 'кубок', 'турнир', 'педагог', 'студ', 'админ',
        'даша', 'дарья', 'проспект'
    ]
    question_words = [
        'когда', 'где', 'куда', 'во сколько', 'сколько', 'можно', 'надо',
        'нужно', 'какая', 'какой', 'какие', 'что', 'как', 'почему'
    ]
    is_question = '?' in lower or any(lower.startswith(word) for word in question_words)
    is_studio_related = any(word in lower for word in studio_words) or 'у нас' in lower
    if is_question and is_studio_related:
        return 'studio_question'

    parent_markers = ['у кого', 'кто может', 'девочки', 'родители', 'кто едет']
    if any(marker in lower for marker in parent_markers):
        return 'ignore'

    if is_question:
        return 'admin_required'

    return 'ignore'
