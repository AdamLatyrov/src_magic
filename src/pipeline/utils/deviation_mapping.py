"""Разметка находок кодами справочника: общая политика размечающих этапов.

Находят проблему там, где её видно, а адресуют её потом: у находок поэкранного
прохода (`pipeline.utils.screen_findings`) и у находок проверок проблем пути
(`pipeline.utils.friction_findings`) нет ни блока, ни кода дефекта, ни взятой из
справочника зоны. Адрес возвращает им один и тот же порядок — разметка одним
обращением без изображений (`summarize_screen`, `map_journey_frictions`).

Общий здесь и разбор ответа: соответствие привязано к находке номером (тексты
находок похожи и различаются экраном и цитатой), код сверяется со справочником, а
зона берётся справочником, а не тем, что показалось ветке. Держать это дважды
значило бы развести правила ровно там, где отчёт обязан быть одинаковым: одно и то
же нарушение не должно штрафоваться по-разному, когда его нашёл экран и когда его
увидел весь путь (то же правило, что у ветки блоков —
`pipeline.utils.block_findings._deviations`).

Форма строга, значения мягкие. Ответа без перечня `matches` не бывает — это
неразборчивый ответ, и он поднимается наверх как сбой формы, а не как «кода не
нашлись». Код вне справочника стоит кода, но не находки: она остаётся ровно такой,
какой пришла из ветки, потому что «вижу проблему, кода под неё нет» — регулярный
итог сверки с каталогом, а не вымысел.
"""

import logging
from collections import Counter
from collections.abc import Sequence
from typing import Optional, Union

from src.pipeline.schemas.nodes import (
    CXCriteriaModel,
    CXDeviation,
    CXPinnedDeviation,
    CXZone,
)
from src.pipeline.settings.params.logging import LogLabel
from src.pipeline.settings.params.nodes import (
    DROP_AMBIGUOUS_CODE,
    DROP_MATCH_WITHOUT_FINDING,
    DROP_NOT_OBJECT,
    DROP_UNKNOWN_CODE,
    NO_CODE_WORDS,
)
from src.pipeline.utils.catalog import DefectAt, locate
from src.pipeline.utils.json_reply import json_payload

logger = logging.getLogger(__name__)

#: находка обеих размечаемых веток: у поэкранной есть пины, у пути — их нет
AnyDeviation = Union[CXDeviation, CXPinnedDeviation]


def codes_of(
    answer: str, count: int, label: LogLabel
) -> tuple[dict[int, str], Counter[str]]:
    """Ответ разметки — в словарь «номер находки → код».

    Номер — единственное, чем находка различима в ответе: тексты похожи («поле
    просят ввести заново» на трёх экранах), а два одинаковых ответа об одном экране
    разметке не нужны. Присвоенный номеру код остаётся последним: повторный ответ
    про ту же находку не имеет права перебить первый.

    :param answer: ответ модели — перечень соответствий находкам
    :param count: сколько находок у ветки — за её пределами номера нет
    :param label: метка этапа для журнала
    :return: код по номеру находки и счётчик снятых соответствий
    :raises ValueError: если в ответе нет перечня `matches`
    """
    payload = json_payload(answer, "с перечнем соответствий")
    raw = payload.get("matches")
    if not isinstance(raw, list):
        raise ValueError(
            f"в ответе разметки {label.value} нет перечня matches"
        )

    dropped: Counter[str] = Counter()
    codes: dict[int, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            dropped[DROP_NOT_OBJECT] += 1
            continue
        number = _id_of(item.get("id"))
        if not 1 <= number <= count:
            dropped[DROP_MATCH_WITHOUT_FINDING] += 1
            logger.warning(
                "%s: соответствие прислано к находке %s, которой в перечне нет "
                "(её %s) — код снят",
                label,
                item.get("id"),
                count,
            )
            continue
        codes[number] = _code_of(item.get("code"))
    return codes, dropped


def mark(
    deviations: Sequence[AnyDeviation],
    codes: dict[int, str],
    model: CXCriteriaModel,
    label: LogLabel,
    dropped: Counter[str],
) -> list[AnyDeviation]:
    """Находки ветки с адресом: код, проверка, блок и зона — из справочника.

    Берутся ровно четыре поля, и все четыре — из найденной строки справочника, а не
    из ответа: `code` — код этого дефекта (ответ мог написать его не в том регистре),
    `check_key` — проверка, где он живёт (короткая метка различает дефект только
    внутри своей проверки, и без неё адрес ниже по потоку снова слипается с меткой
    чужой проверки), `criterion` — блок, в котором он живёт (по нему сведение и
    отчёт кладут находку на место), `zone` — тяжесть этого кода, а не та, что
    показалась ветке. Всё остальное — текст, улики, номер экрана и пин — не
    переписывается: разметка не смотрела кадр и не имеет права ни передвинуть
    находку, ни переформулировать её.

    :param deviations: находки ветки в том порядке, в каком они уходили в вопрос
    :param codes: адрес по номеру находки; пусто — находка не размечена
    :param model: критериальная модель — по ней адрес сверяется с блоком и зоной
    :param label: метка этапа для журнала
    :param dropped: счётчик снятых соответствий, пополняется здесь
    :return: те же находки — с адресом справочника либо как пришли
    """
    marked: list[AnyDeviation] = []
    for number, deviation in enumerate(deviations, start=1):
        here = _address(
            codes.get(number, ""), model, label, number, dropped
        )
        if here is None:
            marked.append(deviation)
            continue
        marked.append(
            deviation.model_copy(
                update={
                    "code": here.defect.code,
                    "check_key": here.check.key,
                    "criterion": here.block.key,
                    "zone": _raised(
                        deviation,
                        here.defect.code,
                        here.defect.zone,
                        label,
                        number,
                    ),
                }
            )
        )
    return marked


def _raised(
    deviation: AnyDeviation,
    code: str,
    zone: CXZone,
    label: LogLabel,
    number: int,
) -> CXZone:
    """Справочник поднимает тяжесть только доказанной находке.

    Зона кода справочная и перебивает то, что показалось ветке, — но не тогда,
    когда ветка пришла без единой цитаты. Поднять жёлтую бездоказательную находку
    до чёрного пояса по одному совпадению описания с дефектом значит вернуть
    прогону ровно то, что сняла сверка улики с кадром
    (`pipeline.utils.evidence`, `pipeline.utils.friction_findings._zone_of`): выдуманные
    цитаты там убраны, зона опущена, а разметка смотрит текст и картинок не
    видела. Без улики справочник остаётся адресом — кодом и блоком, но не тяжестью.

    Зона приходит то членом перечисления, то его значением (`use_enum_values`
    оставляет в модели строку), поэтому тяжесть сравнивается после `parse`, а
    значению вне канона достаётся мягкое дно — как было, когда сверка шла по
    справочнику рангов.

    :param deviation: находка ветки со своей зоной и своими уликами
    :param code: код, которым разметка закрыла находку — для строки журнала
    :param zone: зона этого кода из справочника
    :param label: метка этапа для журнала
    :param number: номер находки в перечне — им подписана строка журнала
    :return: зона находки после разметки
    """
    branch = CXZone.parse(deviation.zone) or CXZone.GREEN
    catalogued = CXZone.parse(zone) or CXZone.GREEN
    if branch.at_least(catalogued) or deviation.evidence:
        return zone
    logger.info(
        "%s: находка %s размечена кодом %s (зона %s), улик у неё нет — тяжесть "
        "остаётся той, что поставила ветка (%s)",
        label,
        number,
        code,
        zone,
        deviation.zone,
    )
    return deviation.zone


def _address(
    raw: str,
    model: CXCriteriaModel,
    label: LogLabel,
    number: int,
    dropped: Counter[str],
) -> Optional[DefectAt]:
    """Где в справочнике адрес, который подставила разметка.

    Ответ приносит короткую метку (`CL-11`) либо адрес целиком (`cost.total:CL-11`).
    Различает дефект адрес: метка обязана различаться только внутри своей проверки,
    и две проверки с одним кодом и разными зонами для каталога законны — молча взяв
    первую, прогон отдал бы находке штраф чужого нарушения, поэтому совпадение
    ожидается ровно одно.

    Пустой адрес — штатный ответ разметки («соответствия нет»), он ничего не снимает
    и в потери не считается. Код, которого в справочнике нет, стоит кода: находка
    остаётся внекаталожной ровно такой, какой пришла из ветки.

    :param raw: адрес из ответа разметки
    :param model: критериальная модель
    :param label: метка этапа для журнала
    :param number: номер находки в перечне — им подписана строка журнала
    :param dropped: счётчик снятых кодов
    :return: найденный дефект с его блоком и проверкой либо None
    """
    if not raw:
        return None
    found = locate(model, raw)
    if not found:
        dropped[DROP_UNKNOWN_CODE] += 1
        logger.warning(
            "%s: кода %s нет в справочнике — находка %s остаётся внекаталожной",
            label,
            raw,
            number,
        )
        return None
    if len(found) > 1:
        dropped[DROP_AMBIGUOUS_CODE] += 1
        logger.warning(
            "%s: код %s живёт в %s проверках (%s) — найти, о чём находка %s, "
            "нечем, она остаётся внекаталожной",
            label,
            raw,
            len(found),
            ", ".join(item.ref for item in found),
            number,
        )
        return None
    here = found[0]
    logger.info(
        "%s: находка %s размечена адресом %s (зона %s)",
        label,
        number,
        here.ref,
        here.defect.zone,
    )
    return here


def report_coverage(
    deviations: Sequence[AnyDeviation],
    dropped: Counter[str],
    label: LogLabel,
) -> None:
    """Итог разметки в журнал: сколько находок всё-таки закрыто кодом.

    Поштучные строки (`codes_of`, `_address`) объясняют каждый снятый код, но по
    ним не видно размера ущерба: прогон обязан отличать «кода не хватило одной
    находке» от «разметка вернула пустой перечень». Молча это проходит только
    когда потерь нет.

    :param deviations: размеченные находки ветки
    :param dropped: счётчик снятых соответствий
    :param label: метка этапа для журнала
    """
    if not dropped:
        return
    logger.warning(
        "%s: разметка вернула не все соответствия — кодом закрыто %s из %s",
        label,
        sum(1 for deviation in deviations if deviation.code),
        len(deviations),
    )


def _id_of(raw: object) -> int:
    """Номер находки из ответа; нечисловой и пропущенный даёт ноль.

    Ноль заведомо вне перечня (нумерация начинается с единицы), поэтому такая
    строка ответа считается снятой, а не повесившей код на первую находку.
    """
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return 0


def _code_of(raw: object) -> str:
    """Адрес из ответа; «кода нет» остаётся пустой строкой.

    Регистр здесь не снимается: значение может быть адресом целиком
    (`cost.total:CL-11`), где ключ проверки строчный, а метка заглавная. И то и
    другое сравнивается без учёта регистра в `pipeline.utils.catalog.locate`, а наружу
    идёт код из справочника; своё слово об отсутствии кода снимается до пустой
    строки, чтобы отсутствующий адрес не отличался от кода, которого нет в
    справочнике.
    """
    code = str(raw or "").strip()
    return "" if code.lower() in NO_CODE_WORDS else code
