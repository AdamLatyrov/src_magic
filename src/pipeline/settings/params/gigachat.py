from enum import Enum
from typing import Optional


class Scope(Enum):
    GIGACHAT_API_PERS = "GIGACHAT_API_PERS"
    GIGACHAT_API_CORP = "GIGACHAT_API_CORP"


IMAGE_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
}

FILE_REF = "file_ref"

MESSAGE_ROLES = {"human": "user", "ai": "assistant", "system": "system"}


class ReasoningEfforts(Enum):
    """Усилия рассуждения, которые пропускает контракт V1.

    Его `reasoning_effort` — перечень из трёх значений, и `off` в нём нет:
    обёртка LangChain отдаёт поле на контур при любом непустом значении, а
    `gigachat.models.Chat` разбирает тело запроса по этому перечню и на `off`
    спотыкается ещё до обращения к контуру.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def accepts(cls, effort: Optional[str]) -> bool:
        """Пропускает ли V1 такое значение: сверка по коду перечня, а не по члену.

        Строгое сравнение, без регистра и пробелов: перечня касается то, что
        уходит на контур, и самоподогнанное « LOW » должно сниматься, а не
        разводиться до «low» — иначе мы отправляли бы значение, которого модель
        не просила.
        """
        return effort in {item.value for item in cls}
