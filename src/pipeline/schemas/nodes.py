import re
from enum import Enum
from typing import Optional, get_args, get_origin

from pydantic import (
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from src.pipeline.schemas.core import CoreSchema
from src.pipeline.settings.params.nodes import (
    BLANK_FIELDS,
    BLANK_TEXT,
    DROP_NO_VERDICT,
    DROP_UNKNOWN_CODE,
    DROP_UNKNOWN_SUBCRITERION,
    EVIDENCE_NUMBER,
    OFF_FRAME_TEXT,
    PATH_EXCUSE,
    PIN_CENTER,
    TRUE_FLAGS,
    UNIT_FINDING,
    UNIT_NORM,
    CXItemStatus,
    CXJourneyType,
    CXNodeStatus,
    CXZone,
)


class NodeAnswer(CoreSchema):
    """Класс-интерфейс ответа ноды графа: форма, к которой приводят ответ контура.

    Форму ответа задаёт JSON-схема со строго типизированными полями — с `anyOf`
    в каждом она подсказывала бы модели сомневаться. Контур при этом пишет
    `null` там, где ждут списка, список там, где ждут строку, число там, где
    ждут `"7"`, «да» вместо `true` и `OK ` вместо `ok`, а одна ошибка валидации
    обрывает разбор всего ответа разом — вместе со всеми находками сведения.
    Ответ каждой ноды приводится к форме поля по аннотации самого поля:

    - отсутствующее значение получает значение по умолчанию поля, поле без
      дефолта — пустоту своего типа (`[]`, `{}`, `""`, `False`, `0`), а
      `Optional[...]` остаётся `None`;
    - список в строковое поле склеивается построчно, число в строковое поле
      пишется текстом, пустые элементы списка выбрасываются;
    - флаг читается и «да», и `true`, и `1`, слово статуса принимается в любом
      регистре и с пробелами: способ написать слово — не промах модели.

    Строгость остаётся там, где её требует контракт: у поля без значения по
    умолчанию (`items` ответа блока) пустой ответ по-прежнему ошибка схемы —
    молчаливый пустой список дал бы блоку полный балл.

    Полей класс не объявляет: порядок полей ответа есть порядок, в котором
    обязана отвечать модель (`CXFrameReadout`), а база поставила бы свои первыми.
    """

    @staticmethod
    def _kind(annotation: object) -> object:
        """Род поля без обёртки `Optional`: `list[str]` → `list`, `X | None` → `X`."""
        args = get_args(annotation)
        if type(None) in args:
            annotation = next(a for a in args if a is not type(None))
        return get_origin(annotation) or annotation

    @staticmethod
    def _as_text(value: object) -> object:
        """Число в поле строки: модель пишет `steps: [7]` там, где схема ждёт `"7"`."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return value
        return str(value)

    @classmethod
    def _absent(cls, name: str) -> object:
        """Чем заменён написанный моделью `null`: дефолт поля, затем пустота типа."""
        field = cls.model_fields[name]
        if not field.is_required():
            return field.get_default(
                call_default_factory=True, validated_data={}
            )
        if type(None) in get_args(field.annotation):
            return None
        kind = cls._kind(field.annotation)
        if kind in (list, set, frozenset, tuple):
            return []
        if kind is dict:
            return {}
        if kind is bool:
            return False
        if kind in (int, float):
            return 0
        return ""

    @field_validator("*", mode="before")
    @classmethod
    def _to_form(cls, value: object, info: ValidationInfo) -> object:
        annotation = cls.model_fields[info.field_name].annotation
        kind = cls._kind(annotation)
        if value is None:
            return cls._absent(info.field_name)
        if isinstance(value, list):
            usable = [cls._as_text(item) for item in value if item is not None]
            if annotation is str:
                return "\n".join(str(item) for item in usable)
            if kind in (list, set, frozenset, tuple):
                return usable
            if not usable:
                return cls._absent(info.field_name)
            value = usable[0]
        if kind is bool:
            if isinstance(value, str):
                return value.strip().lower() in TRUE_FLAGS
            return bool(value)
        if isinstance(kind, type) and issubclass(kind, Enum):
            return value.strip().lower() if isinstance(value, str) else value
        return cls._as_text(value)

    @property
    def answered(self) -> bool:
        """Узел ответил: итог — успех, а не пустой ответ и не ошибка узла.

        Отвеченность есть не у каждого ответа (вердикт подкритерия её не
        имеет вовсе), поэтому статус читается мягко.
        """
        return getattr(self, "status", None) == CXNodeStatus.SUCCESS


class CXCriteriaDefect(CoreSchema):
    ref: str = Field(
        default="",
        description="Адрес дефекта: ключ проверки и код через двоеточие",
    )
    code: str = Field(description="Код дефекта")
    defect: str = Field(description="Описание дефекта")
    symptom: str = Field(description="Симптом дефекта")
    zone: CXZone = Field(description="Степень критичности дефекта")


class CXCriteria(CoreSchema):
    key: str = Field(description="Ключ критерия")
    text: str = Field(description="Описание критерия")
    essence: str = Field(description="Сущность критерия")
    defects: list[CXCriteriaDefect] = Field(description="Список дефектов")


class CXBlock(CoreSchema):
    """Блок справочника: подкритерии и закреплённые за ними дефекты.

    Канон блока живёт здесь, а не в настройках узла: разбор сверяет вердикты и
    коды со справочником прогона (`CXCriteriaModel`), который лежит в состоянии,
    — настроек конкретной ветки разбору видеть не за чем.
    """

    key: str = Field(description="Ключ блока")
    name: str = Field(description="Наименование блока")
    essence: str = Field(description="Сущность блока")
    criteria: list[CXCriteria] = Field(description="Список критериев")

    @property
    def subcriteria(self) -> dict[str, str]:
        """Подкритерии блока в порядке справочника: ключ → текст."""
        return {criteria.key: criteria.text for criteria in self.criteria}

    @property
    def required_keys(self) -> list[str]:
        """Канон-порядок подкритериев — его обязан повторить ответ блока."""
        return list(self.subcriteria)

    @property
    def defect_zones(self) -> dict[str, str]:
        """Зона дефекта по его коду: её называет справочник, не модель.

        Ключом стоит короткая метка, а она различает дефект только внутри своей
        проверки: две проверки с одной меткой сливаются здесь в одну строку, и
        зона доставётся одной из них. Перечню дефектов блока и его числу
        (`defect_count`) этого хватает; вердикты адрес разбирает через
        `pipeline.utils.catalog`, где совпадения не склеиваются.
        """
        return {
            defect.code: defect.zone
            for criteria in self.criteria
            for defect in criteria.defects
        }


class CXModelInfo(CoreSchema):
    """Версия справочника, которой измерен прогон.

    Счётчики блоков и зон здесь не живут: их задаёт содержимое, и перечень
    зон именами полей пережил бы первую же перекалибровку каталога. Здесь
    только подпись версии — без неё результат прогона несопоставим с редакцией
    линейки, а справочник меняется со временем.
    """

    id: str = Field(default="", description="Идентификатор версии")
    key: str = Field(default="", description="Ключ модели")
    version: str = Field(default="", description="Версия модели")
    title: str = Field(default="", description="Название версии")
    status: str = Field(default="", description="Статус версии")
    created_at: str = Field(default="", description="Создана")
    published_at: str = Field(default="", description="Опубликована")
    source_s3_key: str = Field(default="", description="Исходник выгрузки")
    source_sha256: str = Field(default="", description="Отпечаток исходника")


class CXCriteriaModel(CoreSchema):
    info: CXModelInfo = Field(description="Версия справочника прогона")
    blocks: list[CXBlock] = Field(description="Список блоков")

    def block(self, key: str) -> CXBlock:
        for block in self.blocks:
            if block.key == key:
                return block
        raise KeyError(
            f"Блока {key} нет в критериальной модели {self.info.version}"
        )


class CXFrame(CoreSchema):
    step: int = Field(
        description="Номер кадра в маршруте — от него считается шаг находки"
    )
    data_url: str = Field(
        description="Кадр как data-URL: провайдер заберёт его в контур сам"
    )
    ref: str = Field(
        default="",
        description=(
            "Id загруженного в контур файла этого кадра; пусто, если кадр в контуре "
            "не загружен или его ссылка снята"
        ),
    )


class CXJourney(CoreSchema):
    """Клиентский путь: текстовый контекст и кадры, которые модель видит изображениями.

    Продукт, сценарий, приложение и тип пути — метки золотого набора: по ним сверяют
    распознавание ПОСЛЕ прогона. В запросе модели их нет: `material` отдаёт только
    заметку эксперта. Путь, про который модель узнала продукт из вопроса, — не путь,
    а проверка того, что модель повторит подсказку.
    """

    product: str = Field(
        default="", description="Метка продукта золотого набора"
    )
    scenario: str = Field(
        default="", description="Метка сценария золотого набора"
    )
    application: str = Field(
        default="", description="Метка приложения золотого набора"
    )
    platform: str = Field(
        default="", description="Метка платформы золотого набора"
    )
    journey_type: str = Field(
        default="", description="Метка типа пути золотого набора"
    )
    note: str = Field(
        default="",
        description="Заметка эксперта к материалам — единственный текст, уходящий модели",
    )
    frames: list[CXFrame] = Field(
        default_factory=list, description="Кадры пути в порядке следования"
    )

    @property
    def material(self) -> str:
        """Текстовые материалы прогона: только заметка эксперта, без атрибуции пути."""
        return self.note.strip()


class CXJourneyPath(NodeAnswer):
    """Сквозной путь по описаниям кадров: маршрут, переходы и границы материала.

    Кадры до узла сведения не доходят — у него только описания, поэтому здесь
    живут факты про ПУТЬ целиком, а не про отдельный экран: порядок экранов,
    переходы между поверхностями, возвраты и обрывы, где показаны сроки и цены
    и чего по статичным кадрам сказать нельзя. Атрибуция (`CXJourneyDetails`)
    отвечает на вопрос «что это за путь», этот ответ — на вопрос «как он идёт».

    Ветки проверки спрашивают экраны и видят переходы по изображениям, но сверяют
    их с этим перечнем — поэтому каждое утверждение обязано держать номер кадра,
    иначе улику не на что сослаться.

    Пустое поле значит «в описаниях этого не было», а не «этого нет на пути»:
    там, где материал молчит, честнее пустая строка, чем догадка (та же политика,
    что у атрибуции).
    """

    route: str = Field(
        default="",
        description=(
            "Цепочка экранов пути с номерами кадров: «1-3 витрина вклада, "
            "4-8 анкета, 9 подтверждение, 10 результат». Номер кадра обязателен"
        ),
    )
    surface: str = Field(
        default="",
        description=(
            "Поверхность пути: мобильное приложение | веб | планшет | "
            "смешанная | не видно"
        ),
    )
    access: str = Field(
        default="",
        description=(
            "Клиент идёт авторизованным: авторизован | гость | не видно — от "
            "этого зависит, обязаны ли данные быть предзаполнены"
        ),
    )
    rhythm: str = Field(
        default="",
        description=(
            "Путь проходится один раз или возвращается: однократный | "
            "повторяемый | не видно"
        ),
    )
    transfers: list[str] = Field(
        default_factory=list,
        description=(
            "Переходы в другое приложение, на другой сайт или во внешний "
            "сервис и потеря контекста на них — с номерами кадров"
        ),
    )
    dead_ends: list[str] = Field(
        default_factory=list,
        description=(
            "Экраны, где клиенту некуда идти, и обрывы маршрута — с номерами "
            "кадров"
        ),
    )
    repeats: list[str] = Field(
        default_factory=list,
        description=(
            "Возвраты к тому же экрану или фазе и повторный ввод тех же данных "
            "— с номерами кадров"
        ),
    )
    conditions: list[str] = Field(
        default_factory=list,
        description=(
            "Где показаны срок, ставка, итоговая стоимость, риски и "
            "дополнительные продукты: «итоговая стоимость — кадры 7, 10». Если "
            "что-то из этого не показано ни разу — сказать прямо"
        ),
    )
    values: list[str] = Field(
        default_factory=list,
        description=(
            "Дословные суммы, ставки, сроки и даты с номерами кадров, где они "
            "показаны; одно и то же значение на разных экранах — одной строкой"
        ),
    )
    names: list[str] = Field(
        default_factory=list,
        description=(
            "Названия продуктов, сервисов, разделов и кнопок так, как они "
            "написаны, с кадрами; разнобой написания одного и того же — одной "
            "строкой"
        ),
    )
    communications: list[str] = Field(
        default_factory=list,
        description=(
            "Письма, SMS, push и звонки, которые видны или упоминаются в "
            "материалах, — с кадрами"
        ),
    )
    not_observable: list[str] = Field(
        default_factory=list,
        description=(
            "Что нельзя проверить по статичным кадрам этого пути: тайминги и "
            "скорость, анимация, иные каналы, другие устройства, история "
            "клиента"
        ),
    )

    def listing(self) -> str:
        """Сквозной путь построчно — отдельным блоком его печатают сверка и лог.

        Строки идут в порядке значимости: сначала маршрут и поверхность, потом
        переходы и покрытие условий, в конце границы материала. Пустая строка,
        если сказать нечего: блок тогда проверяет по тому, что видит сам, а не
        по выдуманному маршруту.

        :return: перечень фактов о пути по одному на строку
        """
        lines = [
            f"{name}: {value}"
            for name, value in (
                ("Маршрут", self.route),
                ("Поверхность", self.surface),
                ("Доступ клиента", self.access),
                ("Характер пути", self.rhythm),
            )
            if value
        ]
        lines += [
            f"{name}: {item}"
            for name, items in (
                ("Переходы и потеря контекста", self.transfers),
                ("Тупики и обрывы", self.dead_ends),
                ("Возвраты и повторный ввод", self.repeats),
                ("Где показаны условия", self.conditions),
                ("Значения по кадрам", self.values),
                ("Названия по кадрам", self.names),
                ("Внешние коммуникации", self.communications),
                ("Не проверяется по материалу", self.not_observable),
            )
            for item in items
        ]
        return "\n".join(lines)


class CXJourneyDetails(NodeAnswer):
    """Что это за клиентский путь — распознавание по кадрам до проверки критериев.

    Единственный источник продукта и сценария прогона: блоки оценки читают их отсюда и
    перестают угадывать домен каждый сам, по-своему решая, что здесь `na`.

    `basis` — якорь называния: продукт и сценарий существуют только вместе с дословной
    надписью того кадра, откуда они взяты. Название без цитаты — догадка, и политика
    приёмки (`pipeline.utils.journey.is_incomplete`) считает такой ответ неполным.

    `journey_type` в вопросы ветки оценки НЕ отдаётся (`context`): тип пути — единственный
    атрибут, на который ветки ссылались как на разрешение не проверять, поэтому мерить
    прогон призван продуктом, а не типом. Тип продолжает жить в состоянии — для журнала,
    отчёта и сверки распознавания.

    `is_journey` решает, запускается ли оценка, поэтому это обязан быть настоящий
    вердикт модели: не удалось его получить — в состоянии нет всего объекта, а не
    `is_journey=False`, иначе сбой контура стал бы «пути нет» и обнулил бы прогон.
    """

    is_journey: bool = Field(
        description="Есть ли хотя бы один кадр с интерфейсом приложения или сайта"
    )
    organization: str = Field(default="", description="Компания по интерфейсу")
    basis: str = Field(
        default="",
        description=(
            "Дословные надписи с номерами кадров, по которым названы продукт и "
            "сценарий: «кадр 1 — „Долевой капитал“, кадр 8 — „Страховые "
            "покрытия“». Продукт, которого нет ни на одной надписи кадра, "
            "считается выдуманным: нет цитаты — продукт не называй"
        ),
    )
    product: str = Field(
        default="",
        description=(
            "Продукт по природе договора, а не по вывеске раздела: страхование, "
            "кредит, вклад, счёт, брокерский счёт; внутренность (фонд, пай, "
            "программа, тариф) остаётся уточнением в скобках"
        ),
    )
    scenario: str = Field(
        default="", description="Сценарий пути короткой фразой"
    )
    journey_type: Optional[CXJourneyType] = Field(
        default=None, description="Тип пути; пусто, если путь не распознан"
    )
    note: str = Field(
        default="", description="Причина, если это не клиентский путь"
    )
    path: CXJourneyPath = Field(
        default_factory=CXJourneyPath,
        description=(
            "Как путь идёт: маршрут, переходы, где показаны условия и чего по "
            "материалу не проверить — по описаниям кадров"
        ),
    )
    degraded: str = Field(
        default="",
        description="Чем плох ответ узла: no_answer | unparsed | incomplete",
    )
    status: CXNodeStatus = Field(
        default=CXNodeStatus.SUCCESS,
        validate_default=True,
        description="Чем закончился узел: ответ, пустой ответ или ошибка",
    )

    def context(self) -> str:
        """Атрибуция пути текстом — часть вопроса блоков оценки и сведения.

        Строки говорят, чем мерить: компания, продукт, сценарий, названное
        основание и доступ клиента. `journey_type` сюда не входит намеренно: в
        прогоне 032 ветка сослалась на «тип пути» в 13 вердиктах из 24, и каждый
        такой вердикт был отказом от проверки, а не её результатом. Тип пути
        по-прежнему остаётся в состоянии — его печатают журнал, отчёт и сверка
        распознавания — просто он больше не раздаёт разрешений. О том, что
        атрибуция задаёт мерки, а не объём проверки, пишет промпт самой проверки.

        Пустая строка, если сказать нечего: блок тогда проверяет вслепую — по
        описаниям кадров, а не по выдуманной атрибуции. `Доступ клиента` берётся из
        сквозного пути: проверки проблем пути его не получают целиком, а мерка
        предзаполнения («гость» или «авторизован») нужна именно им.

        :return: построчный перечень распознанного
        """
        lines = [
            f"{name}: {value}"
            for name, value in (
                ("Компания", self.organization),
                ("Продукт", self.product),
                ("Сценарий", self.scenario),
                ("Названо по надписям", self.basis),
                ("Доступ клиента", self.path.access),
            )
            if value
        ]
        return "\n".join(lines)

    @property
    def journey_type_value(self) -> str:
        """Тип пути текстом — для журнала, отчёта и сверки распознавания.

        Вопросам ветки оценки тип не отдают (`context`), поэтому значение читают
        только там, где его проверяют человек или эталон.

        Схема хранит значение перечисления, а не его член (`use_enum_values`), но
        объект собирают и руками — в том числе с членом перечисления. Формат
        строки у смешанного с `str` перечисления зависит от версии Python, поэтому
        здесь приведение явное, а не через f-строку.
        """
        value = self.journey_type
        if isinstance(value, CXJourneyType):
            return value.value
        return str(value or "")


class CXFrameReadout(NodeAnswer):
    """Что происходит на одном кадре — ответ узла сканирования пути.

    Сканирование идёт по окну `[i-1, i, i+1]`, но отвечает модель про один кадр:
    соседи нужны затем, чтобы понять, что произошло до и после. `narrative` —
    главное поле ответа, подробное описание экрана вплоть до доступных действий;
    по нему `summarize_journey` и решает, что это за путь.

    Порядок полей = порядок ответа: `heading` (самая крупная надпись разбираемого
    кадра), `text_on_screen` (текст этого кадра построчно), затем `narrative`.
    Переписанный до пересказа текст меньше смешивается с соседним кадром окна, а
    заголовок служит якорем: если его нет в переписанном тексте, схема помечает
    ответ `degraded = off_frame_text` — узел переспрашивает кадр.

    `is_journey` остаётся трёхзначным: `False` здесь — ответ модели («интерфейса
    нет»), `None` — узел не ответил. Разница та же, что в `pipeline.utils.journey`:
    молчание контура не имеет права выглядеть отказом признать путь.

    `need_filter` — чем кадр объясняет, что он вообще не клиентский путь: фото,
    скан бумаги, чужое приложение, экран без интерфейса. Флаг ставит модель, а
    схема дописывает его по `is_journey = false`. Незаполненное поле фильтром не
    считается никогда: `narrative` и `text_on_screen` обязательны для годного
    ответа, но пустота наказывается отметкой `degraded = blank_text` и одним
    переспросом узла (`scan_journey_frame._reask_frame`), а не потерей кадра —
    экран с переписанным текстом, но без описания (или наоборот) остаётся в
    анализе, и кадровая ветка продолжает его видеть.
    """

    step: int = Field(
        default=0, description="Номер кадра, который описан этим ответом"
    )
    is_journey: Optional[bool] = Field(
        default=None,
        description=(
            "Частью клиентского пути является ли разбираемый кадр: true — на "
            "кадре интерфейс приложения или сайта (включая загрузку, ошибку и "
            "пустое состояние); false — только если интерфейса нет вовсе или "
            "кадр целиком посторонний. Пусто, если интерфейс разобрать не удалось"
        ),
    )
    screen: str = Field(
        default="",
        description=(
            "Что это за экран — короткая подпись к кадру («Выбор условий по "
            "вкладу», «Анкета клиента», «Экран подтверждения операции»)"
        ),
    )
    heading: str = Field(
        default="",
        description=(
            "Самая крупная надпись разбираемого кадра — дословно, как нарисована "
            "(заголовок экрана, имя вкладки или верхняя строка). Её ищет "
            "`text_on_screen` первым рядом: бери её только с разбираемого "
            "изображения, соседний кадр окна заголовком не подтверждается"
        ),
    )
    text_on_screen: str = Field(
        default="",
        description=(
            "ВЕСЬ текст РАЗБИРАЕМОГО кадра дословно: каждой строке "
            "кадра — своя строка ответа (перенос строки), сверху вниз и до нижнего "
            "края экрана, от заголовка, названного в `heading`, до нижнего меню. "
            "Сноски мелким шрифтом, текст согласий и ссылок на документы, "
            "подписи под полями, счётчики шагов и плашки статуса — часть текста "
            "кадра, а не украшение: сокращать их многоточием или «и так далее» "
            "нельзя. Пересказ запрещён; закрытое деперсонализацией значение пиши как "
            "[скрыто]. Соседние кадры окна не источник: их надписи сюда не "
            "попадают, даже если они читаются лучше"
        ),
    )
    narrative: str = Field(
        default="",
        description=(
            "Главное поле ответа: подробное описание разбираемого кадра одним "
            "абзацем, начинающееся со слов «На экране видно, что пользователь ». "
            "Пиши его после `text_on_screen` и по тому, что переписал с этого же "
            "кадра. Опиши раздел и заголовок, какие данные и значения показаны "
            "(суммы, ставки, даты, статусы, подписи полей), какие элементы "
            "доступны (кнопки, поля, чипы, переключатели, ссылки, меню) и что "
            "только что произошло. Заканчивай абзац перечислением доступных "
            "действий словами «Доступные действия: …» — только по тем кнопкам и "
            "ссылкам, которые действительно видны на кадре"
        ),
    )
    need_filter: bool = Field(
        default=False,
        description=(
            "Кадр вообще не клиентский путь и дальше по нему не судят: true — "
            "только когда это точно видно (фото, видео, мем, скан бумаги, "
            "слайд, картинка без интерфейса, чужое приложение, экран устройства "
            "не по задаче, пустой чёрный экран), и всегда вместе с "
            "`is_journey=false` и `why_not_journey`. Нечитаемый или короткий "
            "текст, размытость, плашки, непонятный экран, загрузка и ошибка — "
            "не основания фильтра: при false кадр остаётся в анализе с любым "
            "неполным ответом, узел только переспросит про пустое поле"
        ),
    )
    stage: str = Field(
        default="",
        description=(
            "Фаза пути одним словом: вход, выбор продукта, выбор условий, ввод "
            "данных, согласия, подтверждение, ожидание результата, результат, "
            "постобслуживание"
        ),
    )
    changed: str = Field(
        default="",
        description=(
            "Чем разбираемый кадр отличается от предыдущего кадра окна: что "
            "появилось, исчезло или переключилось. Пусто, если левого соседа "
            "в окне нет"
        ),
    )
    why_not_journey: str = Field(
        default="",
        description=(
            "Заполняется только при is_journey=false: почему кадр посторонний "
            "пути"
        ),
    )
    confidence: str = Field(
        default="",
        description=(
            "low | med | high — уверенность в описании; low, если кадр размыт "
            "или закрыт плашками, и всегда low, если текст кадра пришлось "
            "обрвать на середине"
        ),
    )
    status: CXNodeStatus = Field(
        default=CXNodeStatus.SUCCESS,
        validate_default=True,
        description="Чем закончился узел: ответ, пустой ответ или ошибка",
    )
    degraded: str = Field(
        default="",
        description=(
            "Чем плох ответ узла: no_answer | unparsed | blank_text | "
            "off_frame_text"
        ),
    )

    @model_validator(mode="after")
    def _only_a_foreign_frame_is_filtered(self) -> "CXFrameReadout":
        """С счёта снимается только кадр, который не может быть путём.

        Основание ровно одно — ответ модели `is_journey = false`: фото, скан
        бумаги, мем, чужое приложение, экран без интерфейса вовсе. Пустое поле
        таковым не было и не станет: в прогоне от 15.10.2026 экран «Автоплатеж
        подключён» выпал из проверки всех семи блоков только потому, что форма
        сняла его за пустой `text_on_screen`, — тогда как `narrative` у него был,
        и кадровой ветке было на что смотреть. Кадр пути остаётся в анализе при
        любой неполноте ответа; судить по нему будут по тому, что в ответе есть.

        Молчание контура сюда не относится: у записи `unread` статус ошибки, и
        кадр остаётся непроверенным экраном, а не отфильтрованным.
        """
        if self.answered and self.is_journey is False:
            self.need_filter = True
        return self

    @model_validator(mode="after")
    def _blank_answer_keeps_the_frame(self) -> "CXFrameReadout":
        """Пустое обязательное поле — отметка и переспрос, но не фильтр.

        Отметка `degraded = blank_text` нужна: по ней узел переспрашивает кадр
        один раз (`scan_journey_frame._reask_frame`), она же попадает в журнал и
        в текстовый след пути, чтобы «текст не переписан» читался неполнотой
        ответа, а не пустотой экрана. Сам кадр при этом никуда не уходит.
        """
        if self.answered and not self.need_filter and self.blank_answer:
            self.degraded = BLANK_TEXT
        return self

    @model_validator(mode="after")
    def _heading_anchors_the_transcription(self) -> "CXFrameReadout":
        """Заголовок кадра обязан найтись в его же переписанном тексте.

        Единственная связь текста с изображением, которую схема может проверить
        без картинки: модель сначала называет самую крупную надпись разбираемого
        кадра, потом переписывает текст. В прогоне 009 кадр 3 («Детальная
        информация о вкладе») вернулся с текстом кадра 2 («Накопления») — по
        такому ответу ветка цитировала бы надписи, которых на экране не было.

        Пустой заголовок ошибкой не считается: им становится верхняя строка
        текста, и проверка на этом умолкает. Пустой же текст уже отфильтрован
        предыдущим правилом, и сравнивать не с чем.
        """
        if not self.answered or self.need_filter:
            return self
        if not self.heading.strip():
            self.heading = self._top_line()
            return self
        if self._looks_like_another_frame():
            self.degraded = OFF_FRAME_TEXT
        return self

    @staticmethod
    def _flat(value: str) -> str:
        """Надпись одной строкой в нижнем регистре — так её ищут в тексте."""
        return " ".join(value.split()).casefold()

    def _top_line(self) -> str:
        """Верхняя непустая строка текста кадра — якорь вместо пустого заголовка."""
        for line in self.text_on_screen.splitlines():
            if line.strip():
                return line.strip()
        return ""

    def _looks_like_another_frame(self, min_heading_word: int = 4) -> bool:
        """Признак переписанного соседа: нет ни заголовка, ни его главного слова.

        Сверять надпись целиком жёстко — та же строка пишется то с дефисом, то с
        кавычкой, — поэтому вторая попытка ищет самое длинное слово заголовка.
        Если не нашлось и оно, переписан явно другой экран, а не та же надпись в
        другой орфографии.
        """
        flat_text = self._flat(self.text_on_screen)
        flat_heading = self._flat(self.heading)
        if not flat_text or not flat_heading:
            return False
        if flat_heading in flat_text:
            return False
        longest = max(flat_heading.split(), key=len).strip(
            "«»\"'()[]{},.:;!?—–-"
        )
        return len(longest) >= min_heading_word and longest not in flat_text

    @property
    def usable(self) -> bool:
        """Кадр годится для анализа: описание есть и его не отбросили."""
        return self.answered and not self.need_filter

    @property
    def blank_answer(self) -> bool:
        """Описания или дословного текста кадра нет — переписывать нечего."""
        return bool(self.missing_fields)

    @property
    def suspect_text(self) -> bool:
        """Текст кадра под подозрением: его заголовка в нём нет.

        Кадр остаётся в анализе — описание-то верное, — но цитировать его текст
        как надпись этого экрана нельзя, пока подозрение не снят переспросом.
        """
        return self.degraded == OFF_FRAME_TEXT

    @property
    def needs_reask(self) -> bool:
        """Ответ построен правильно, но по существу его надо переспросить."""
        return self.degraded in (BLANK_TEXT, OFF_FRAME_TEXT)

    @property
    def fault(self) -> str:
        """Чем неудобен ответ — одной строкой для журнала и требования к нему."""
        if self.suspect_text:
            return (
                "заголовок разбираемого кадра не найден в переписанном тексте"
            )
        return self.blank_reason

    @property
    def missing_fields(self) -> tuple[str, ...]:
        """Имена пустых обязательных полей ответа — по ним переспрашивают."""
        return tuple(
            name for name in BLANK_FIELDS if not getattr(self, name).strip()
        )

    @property
    def blank_reason(self) -> str:
        """Чем пустой ответ кадра объясняется человеку: по каждому полю словом."""
        return "; ".join(BLANK_FIELDS[name] for name in self.missing_fields)

    @property
    def filter_note(self) -> str:
        """Чем объяснён фильтр: слово модели, в крайнем случае — пустые поля.

        Строка уходит и в текстовый след пути, и в печать прогона: без причины
        отфильтрованный кадр неотличим от выброшенного наугад. Основанием фильтра
        бывает только посторонний кадр, поэтому первым идёт `why_not_journey` —
        почему модель сочла кадр не путём. Пустые поля называются следствием: у
        фото переписывать нечего, и без слова модели отличить «экран чужой» от
        «с экрана нечего взять» нельзя.
        """
        if not self.need_filter:
            return ""
        if self.why_not_journey:
            return f"Почему посторонний: {self.why_not_journey}"
        if self.degraded == BLANK_TEXT and self.blank_reason:
            return f"Почему снят с анализа: {self.blank_reason}"
        return ""

    def listing(self) -> str:
        """Описание кадра одной строкой — так оно уходит в контекст следующих
        кадров и в вопрос узла сведения пути."""
        note = self.narrative or self.screen or "описания не было"
        return f"Кадр {self.step} — {self.screen or 'экран не назван'}: {note}"


class CXDeviation(NodeAnswer):
    """Одно отклонение на пути — общий выход всех проверяющих веток.

    Ветка блоков находит отклонение как дефект справочника, ветка проблем пути —
    как проблему пути. Сведению безразлично, кто его нашёл: оно сравнивает сами
    отклонения и решает, одна это проблема или две. Поэтому у отклонения одна
    форма на обе ветки, а различаются ветки заполнением:

    - у находки проверок пути адреса нет по построению — она ищет по кадрам, не
      зная справочника; код, блок и зону ей ставит разметка
      (`pipeline.utils.journey_mapping`), после которой находка неотличима от находки
      блока;
    - `related_step`, `rec` и `severity` пишет ветка проблем пути: блок проверяет
      подкритерий и не советует, а тяжесть отчёта всё равно выводится из зоны.

    Держать две формы значило бы иметь два списка в сырье сведения и два места,
    где правила доказательства расходятся, — сравнить прогоны веток было бы
    нельзя.
    """

    criterion: str = Field(
        default="",
        description=(
            "Блок справочника, куда легла находка: ветка блоков берёт его из "
            "собственного блока, проверки — от своей разметки по найденному коду, "
            "проход автозаполнения получает на сведении"
        ),
    )
    code: str = Field(
        default="",
        description=(
            "Код дефекта из справочника; пусто у незаматченной находки — она "
            "остаётся внекаталожной"
        ),
    )
    check_key: str = Field(
        default="",
        description=(
            "Проверка справочника, где живёт этот код; с кодом она названа там, "
            "где адрес удалось разобрать. Короткая метка различает дефект только "
            "внутри своей проверки, поэтому без неё одна и та же метка двух "
            "проверок снова слипается в незнакомый код"
        ),
    )
    zone: CXZone = Field(
        default=CXZone.YELLOW,
        description=(
            "Зона: её берёт справочник по коду, и только без кода остаётся зона "
            "той ветки, которая нашла отклонение"
        ),
    )
    step: int = Field(
        default=0,
        description="Номер экрана из подписи `[кадр N]`, где отклонение видно; 0 — уровень пути",
    )
    related_step: int = Field(
        default=0,
        description=(
            "Второй экран, когда отклонение доказывается парой кадров (обещание и "
            "факт, ввод и повторный ввод); 0 — достаточно одного экрана"
        ),
    )
    text: str = Field(
        default="",
        description="Что видно на экране → как это трактуется → чем грозит клиенту",
    )
    rec: str = Field(
        default="",
        description="Конкретное изменение экрана или сценария; блок рекомендации не даёт",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Дословные цитаты с кадров, каждая с номером кадра",
    )
    severity: str = Field(
        default="",
        description="high | med | low — слово проверки; вес штрафа всё равно выводится из зоны",
    )


class CXSubVerdict(NodeAnswer):
    """Вердикт модели по одному подкритерию — элемент `items` ответа блока.

    Легальных оснований у `na` два, и оба про материал: «проверяется только по
    изображению» и «материал этого не показал» (перечисление кадров в `note`).
    Снятие ссылкой на то, чем путь является, основанием не считается и помечается
    отдельно — `dismissed_by_attribution`. Симметричную проверку держит `ok`:
    статус соблюдён, а ссылки на кадр или посчитанное число в `note` нет — это
    `unproven_ok`, и он же стоит блоку отметки в разборе.
    """

    key: str = Field(description="Ключ подкритерия из справочника")
    status: CXItemStatus = Field(
        description="ok — соблюдается, fail — видна проблема, "
        "na — проверяется только по изображению либо материал этого не показал"
    )
    note: str = Field(
        default="", description="Что видно на кадрах и почему такой статус"
    )

    @property
    def dismissed_by_attribution(self) -> bool:
        """Снят ли подкритерий ссылкой на атрибуцию вместо проверки по материалу."""
        return CXItemStatus(self.status) is CXItemStatus.NA and bool(
            PATH_EXCUSE.search(self.note)
        )

    @property
    def unproven_ok(self) -> bool:
        """Объявлен ли подкритерий соблюдённым без ссылки на материал.

        Уликой считается номер кадра или посчитанное число — то, что прогон может
        показать валидатору. Без цифры `note` остаётся мнением, а балл за неё уже
        получен: доля `ok` знаменателя не знает, смотрели этот подкритерий или нет.
        """
        return CXItemStatus(self.status) is CXItemStatus.OK and not (
            EVIDENCE_NUMBER.search(self.note)
        )


class CXBlockAnswerItem(NodeAnswer):
    """Вердикт подкритерия в ответе блока — элемент `items`.

    Повторяет `CXSubVerdict`, кроме формы `status`: схема ответа решает только
    то, что статус вообще назван одним из трёх слов, а чем считать всё
    остальное, решает разбор (`pipeline.utils.block_findings`) — он же ведает, что
    чужой ключ подкритерия стоит счётчика, а не потерянного ответа блока.
    """

    key: str = Field(description="Ключ подкритерия строго из списка блока")
    status: CXItemStatus = Field(
        description="ok — соблюдается, fail — видна проблема, na — проверяется "
        "только по изображению либо материал этого не показал"
    )
    note: str = Field(
        default="",
        description="Что записано в материале и почему такой статус",
    )


class CXBlockAnswerDeviation(NodeAnswer):
    """Отклонение в ответе блока — элемент `deviations`.

    Полей ровно четыре, как требует промпт: зону модель не выбирает (она
    закреплена за кодом в справочнике), а `related_step`, `rec` и `severity`
    принадлежат проверкам проблем пути. Шире схема стала бы приглашением
    заполнить лишнее, и разбор всё равно выбросил бы это слово модели.
    """

    code: str = Field(description="Код дефекта строго из списка этого блока")
    step: int = Field(
        default=0,
        description="Номер кадра, на котором виден дефект; 0 — кадр не назван",
    )
    text: str = Field(
        default="",
        description="Что записано на кадре → почему это указанный дефект, "
        "а не норма",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Дословно переписанный текст кадра, каждая цитата со своим "
        "номером",
    )


class CXBlockAnswer(NodeAnswer):
    """Форма ответа блока критериальной модели.

    Схема повторяет контракт, который промпт обеих веток задаёт текстом, —
    иначе инструкция формата спорила бы с вопросом. Строгость здесь нарочно:
    вердикта без `items` контуру не простить, потому что молчаливый пустой
    список дал бы блоку полный балл.
    """

    items: list[CXBlockAnswerItem] = Field(
        min_length=1,
        description="Вердикты по всем подкритериям блока в порядке справочника",
    )
    deviations: list[CXBlockAnswerDeviation] = Field(
        default_factory=list,
        description="Отклонения только по подкритериям со статусом fail",
    )
    note: str = Field(
        default="",
        description="Общее впечатление по блоку и чем объяснена "
        "пустота отклонений",
    )


class CXCheckFindings(NodeAnswer):
    """Ответ одной проверяющей ветки: что она нашла и чем закончился узел.

    Источников находок в прогоне два — блок критериальной модели и проверка
    проблем пути, — и каждый из них пишет в состояние свой ключ. Общими держат
    четыре поля: кто нашёл (`source_key`), что нашёл (`deviations`), чем
    объясняет пустоту (`note`) и чем закончился узел (`status`). Сведение и отчёт
    читают их у обеих веток одинаково, а всё, что есть только у блоков, — `items`
    с вердиктами подкритериев, из которых считается балл, — живёт в подклассе.

    Пустой перечень значит «проверено и чисто», отсутствие ответа — «проверка не
    состоялась»; различает эти два случая `status`, и смешать их дороже всего:
    сорванные обращения к контуру дали бы чистый отчёт. Третьего случая эти два
    поля не различают: проверка отвечена, а находку выбросил разбор — за это
    отвечает `dropped`, иначе пустой отчёт ветки при живой модели выглядел бы её
    результатом.
    """

    source_key: str = Field(
        default="",
        description=(
            "Ключ источника: блок справочника или проверка пути — он же ключ "
            "записи в состоянии"
        ),
    )
    deviations: list[CXDeviation] = Field(
        default_factory=list,
        description=(
            "Найденные отклонения; пустой список, если проверять было не на что"
        ),
    )
    note: str = Field(
        default="", description="Что проверено и чем объяснена пустота"
    )
    status: CXNodeStatus = Field(
        default=CXNodeStatus.SUCCESS,
        validate_default=True,
        description="Чем закончился узел: ответ, пустой ответ или ошибка",
    )
    dropped: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Сколько вердиктов и отклонений разбор отбросил, по каждой причине "
            f"своей строкой ({DROP_UNKNOWN_CODE}, "
            f"{DROP_UNKNOWN_SUBCRITERION}, {DROP_NO_VERDICT})"
        ),
    )

    @property
    def unexplained(self) -> bool:
        """Находок нет, узел отвечен, а чем объяснена пустота — не сказано.

        Такой итог неотличим от непросмотренного пути: промпты обеих веток его
        запрещают, код его не наказывает, но обязан показать его в журнале.
        """
        return not self.deviations and self.answered and not self.note.strip()

    def emptiness_note(self) -> str:
        """Чем журнал объяснит пустой перечень и что в него не попало.

        Строка пуста, когда сказать нечего: находки есть, и разбор ничего не
        выбросил. Печатают её узлы обеих проверяющих веток — по ней видно, что
        ноль находок значит что-то конкретное, а не «на этом пути чисто».

        :return: пояснение модели (или отметка о его отсутствии) и потери разбора
        """
        parts: list[str] = []
        if not self.deviations:
            note = " ".join(self.note.split())
            if note:
                parts.append(f"пусто потому что: {note}")
            elif self.answered:
                parts.append("пустота не объяснена")
        if self.dropped:
            counts = ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(self.dropped.items())
            )
            parts.append(f"отброшено разбором: {counts}")
        return "; ".join(parts)


class CXBlockFindings(CXCheckFindings):
    """Ответ модели по одному блоку: вердикты подкритериев и отклонения.

    От блока к общему списку отклонений добавляется только `items`: из них
    считается балл, и только они решают, попал ли подкритерий в знаменатель.
    Отклонения блока — те же `CXDeviation`, что и у проверок проблем пути, с
    кодом и зоной из справочника.
    """

    items: list[CXSubVerdict] = Field(
        default_factory=list,
        description="Вердикты по подкритериям блока в порядке справочника",
    )

    @property
    def self_dismissed(self) -> list[str]:
        """Ключи подкритериев, снятых ссылкой на тип или сценарий пути.

        Балл их не видит — `na` вне знаменателя, — и без этого перечня прогон, в
        котором половину справочника не проверяли, выглядит как прогон, в котором
        всё проверено и чисто.
        """
        return [
            item.key for item in self.items if item.dismissed_by_attribution
        ]


class CXFrameObservation(NodeAnswer):
    """Разбор одного кадра проверкой проблем пути.

    Запись — единица проверки: пока кадр не разобран, проверенным он не считается,
    а пустой перечень находок без записей неотличим от непросмотренного пути. От
    описания поэкранного прохода (`CXScreenNote`) отличается тем, что держит итог
    по мерке своей проверки: величина с кадра и вывод о ней — то, по чему потом
    видно, искал ли кандидат вообще.
    """

    step: int = Field(
        description="Номер экрана из подписи `[кадр N]`, 1-based"
    )
    seen: str = Field(
        default="",
        description="Что на кадре видно: заголовок, доступные действия, "
        "названные величины",
    )
    values: list[str] = Field(
        default_factory=list,
        description="Дословно переписанные величины и условия кадра: суммы, "
        "сроки, проценты, границы, названия действий",
    )
    unit: str = Field(
        default="",
        description="Шифр меры проверки (`Р3`, `Н7`), по которой сведён кадр",
    )
    candidate: bool = Field(
        default=False,
        description="Есть ли на кадре кандидат на проблему этого предмета",
    )
    why: str = Field(
        default="",
        description="Чем кандидат является или чем объясняется норма этого пути",
    )


class CXUnitReview(NodeAnswer):
    """Разбор одной меры проверки по всему пути.

    Строка обязана быть по каждой мере: пока мера не разобрана, проверка считается
    сделанной наполовину, и ноль находок по этой мере значит «не смотрели». Кадр
    меру подписывает своим `unit`, но подпись показывает только итог экрана — здесь
    же итог меры, и по нему видно, что проверку не отделили первой попавшейся меркой.
    """

    unit: str = Field(
        default="", description="Шифр меры из раздела «Задание» (`Р7`, `Н3`)"
    )
    frames: list[int] = Field(
        default_factory=list,
        description="Номера кадров, где мера смотрелась, пустой — если не нашлась",
    )
    result: str = Field(
        default="",
        description="Итог по мере: finding | norm | not_applicable",
    )
    why: str = Field(
        default="",
        description="Что увидено на названных кадрах и почему такой итог",
    )


class CXFrictionFindings(CXCheckFindings):
    """Ответ одной проверки проблем пути.

    Отклонения — общего контейнера `CXCheckFindings`, и это не формальность:
    сведение получает находки проверки тем же перечнем, что и находки блоков.
    Своими полями идут разбор (`observations`) и перечень мер, которые проверка не
    назвала ни на одном кадре (`unchecked_units`): по ним видно, что ноль находок
    значит «перебрано и чисто», а не «меру не смотрели».
    """

    observations: list[CXFrameObservation] = Field(
        default_factory=list,
        description="Разбор кадров: одна запись на просмотренный экран",
    )
    unit_reviews: list[CXUnitReview] = Field(
        default_factory=list,
        description="Разбор по меркам проверки: одна строка на каждую меру",
    )
    unchecked_units: list[str] = Field(
        default_factory=list,
        description="Шифры мер проверки, не названные ни в одном разборе",
    )
    degraded: str = Field(
        default="",
        description="Чем плох прогон окна: batches_failed:i/j | no_answer",
    )

    @property
    def covered_steps(self) -> set[int]:
        """Номера кадров, до которых дошёл разбор этой проверки."""
        return {item.step for item in self.observations}

    @property
    def candidate_steps(self) -> list[int]:
        """Кадры, на которых проверка увидела кандидата."""
        return sorted(
            item.step for item in self.observations if item.candidate
        )

    @property
    def copied_steps(self) -> list[int]:
        """Кадры, описанные той же строкой, что и другой кадр: их не читали.

        Шесть записей «параметры вклада» на шести разных экранах значат, что окно
        переписало первый кадр и подписало им остальные: разбор по числу записей
        полный, а пути за ним не стоит. По этой строке журнала молчаливая пустота
        находок читается как «кадры не смотрели», а не как «смотрели, чисто».
        """
        first: dict[str, int] = {}
        copies: list[int] = []
        for item in self.observations:
            shape = " ".join(item.seen.casefold().split())
            if not shape:
                continue
            if first.setdefault(shape, item.step) != item.step:
                copies.append(item.step)
        return sorted(copies)

    @property
    def candidates_without_finding(self) -> list[int]:
        """Кадры с кандидатом без отклонения и без причины отказа.

        Контракт требует на каждого кандидата находку или причину, по которой это
        норма. Кандидат без того и другого — место, где проблема была признана и
        молча снята: пустой отчёт при живом разборе.
        """
        named = {found.step for found in self.deviations}
        named |= {
            item.step
            for item in self.observations
            if item.candidate and item.why.strip()
        }
        return [step for step in self.candidate_steps if step not in named]

    @property
    def units_with_finding(self) -> list[str]:
        """Шифры мер, по которым разбор сказал «находка».

        Мера с находкой и шифр отклонения — разные вещи: отклонение может лежать в
        блоке соседа, а названа при этом быть своя мера. По этому перечню видно,
        что ноль находок означает «меру проверили и сняли», а не «меры не было».
        """
        return sorted(
            item.unit
            for item in self.unit_reviews
            if item.result == UNIT_FINDING
        )

    @property
    def units_without_deviation(self) -> list[str]:
        """Меры, объявленные находкой, для которых ответа с отклонением нет.

        Сводка мер и перечень отклонений — два поля ответа, и оборванный ответ теряет
        второе, оставив первое: прогон печатает «мер с находкой: Р7» и «отклонений
        ноль», а это читается как чистый путь там, где модель нашла и не дописала.
        Мера считается потерянной, когда ни одно отклонение не стоит на её кадрах.
        """
        steps = {
            step
            for item in self.deviations
            for step in (item.step, item.related_step)
        }
        return sorted(
            item.unit
            for item in self.unit_reviews
            if item.result == UNIT_FINDING and not (set(item.frames) & steps)
        )

    @property
    def units_norm_with_shortfall(self) -> list[str]:
        """Меры, закрытые нормой там, где их же строка назвала нехватку.

        Контур переписывает обе величины и ставит знак сравнения правильно, но итог
        пишет `norm`: «остаток источника 108,59 ₽ < оформляемые 10 000 ₽» — и это
        проходит как чистый путь. Ответ противоречит сам себе, и прогон обязан видеть
        это здесь, а не молча штрафовать нолём.
        """
        unit_shortfall = re.compile(r"<|(?<!не )(?<![\w])меньше")

        return sorted(
            item.unit
            for item in self.unit_reviews
            if item.result == UNIT_NORM and unit_shortfall.search(item.why)
        )

    @property
    def units_without_frame(self) -> list[str]:
        """Меры, сведённые без названного кадра: отрицание, которое не на чём держать.

        «На маршруте нет X» без кадра — та же пустота, что и пустой ответ: прогон
        по этой строке не отличит меру, которую смотрели, от меры, о которой
        просто вспомнили.
        """
        return sorted(
            item.unit
            for item in self.unit_reviews
            if not item.frames
            and item.result != UNIT_FINDING
            and not any(char.isdigit() for char in item.why)
        )

    @property
    def units_reused_evidence(self) -> list[str]:
        """Меры, сведённые одним основанием: той же строкой или теми же кадрами.

        Модель, которой лень сводить семь мер по-разному, пишет всем «все шаги
        проходимы» — или подставляет всем один и тот же перечень кадров, — и
        считается сдавшей. Основание у такой сводки одно на несколько мер, и по
        прогону она стоит в одном ряду с неразобранными: мера отличается тем, что
        под неё ищется.
        """

        def mark(item: CXUnitReview) -> tuple:
            reason = item.why.strip().casefold()
            return (
                ("why", reason) if reason else ("frames", tuple(item.frames))
            )

        marks = [mark(item) for item in self.unit_reviews]
        twins = {
            value for value in marks if value[1] and marks.count(value) > 1
        }
        return sorted(
            item.unit for item in self.unit_reviews if mark(item) in twins
        )

    @property
    def dismissed_by_attribution(self) -> bool:
        """Пустой перечень объяснён тем, чем путь является, а не материалом.

        Такая пустота — не «проверено и чисто»: проверка по кадрам этого пути не
        состоялась, и шесть проверок идут по пути любого типа и сценария.
        """
        return not self.deviations and bool(PATH_EXCUSE.search(self.note))


class CXPinnedDeviation(CXDeviation):
    """Отклонение с пином: координаты указывают, на какой элемент экрана смотреть.

    Отклонение остаётся общим (`pipeline.schemas.nodes.CXDeviation`) — сведение читает
    находки блоков, проверок пути и обеих фокусных веток одним перечнем и
    дедуплицирует их между собой. Своими полями остаются `x` и `y`: их пишет только
    то, что судит по одному экрану и отвечает за место на нём — незаполненное поле
    автозаполнения или элемент поэкранной находки
    (`pipeline.schemas.result.IssuePin`, `pipeline.schemas.result.ScreenFinding`).

    У находки автозаполнения `evidence` пустой: дословной цитаты здесь нет — уликой
    служит подпись поля, названная в `text`. По той же причине зона жёлтая: ручной
    ввод известного банку значения мешает, но не блокирует путь.
    """

    x: float = Field(
        default=PIN_CENTER,
        description="Процент от левого края кадра до элемента, о котором находка",
    )
    y: float = Field(
        default=PIN_CENTER,
        description="Процент от верхнего края кадра до элемента, о котором находка",
    )


class CxAutofillFindings(CXCheckFindings):
    """Ответ фокусного прохода «лишний ручной ввод».

    Контейнер тот же, что у блоков и у шести проверок пути: у прохода нет канона
    подкритериев и балл он не размечает, — добавить нечего, кроме отметки о том,
    что часть обращений к контуру не состоялась.

    `degraded` держит счёт сорванных групп кадров (`batches_failed:2/7`): пустой
    перечень при живом контуре значит «предзаполнять нечего», а пустой перечень
    там, где половина групп упала, значит «мы не смотрели» — без счётчика эти два
    случая печатались бы одинаково, и молчание прохода выглядело бы чистотой пути.
    """

    deviations: list[CXPinnedDeviation] = Field(
        default_factory=list,
        description="Экраны, где поле можно было подставить; пустой список, "
        "если предзаполнять было нечего",
    )
    degraded: str = Field(
        default="",
        description="Чем плох проход: batches_failed:i/j | no_answer",
    )


class CXScreenNote(NodeAnswer):
    """Описание одного экрана поэкранного прохода: чем он является и как сделан.

    Описание обязано быть на каждом кадре, в том числе там, где находок нет:
    отсутствие находок — норма, отсутствие описания — нет. Держится отдельно от
    отклонений именно поэтому — отклонений на экране может не быть вовсе.
    """

    step: int = Field(
        description="Номер экрана из подписи `[кадр N]`, 1-based: по нему "
        "описание приклеивается к кадру прогона"
    )
    summary: str = Field(
        default="",
        description="Что это за экран и как он сделан с точки зрения CX; пустое "
        "только у кадра, который разобрать не удалось",
    )


class CXScreenFindings(CXCheckFindings):
    """Ответ поэкранного прохода: описания кадров и их находки с пинами.

    Отклонения — общего контейнера `CXCheckFindings`, и это не формальность:
    находка того же экрана, названная блоком справочника и этим проходом, должна
    схлопнуться в одну проблему и оштрафовать прогон один раз
    (`pipeline.utils.findings.dedup_issues`). Описания (`screens`) лежат рядом: они в
    сведение не идут, а питают секцию `screens` ответа прогона.

    `dropped` объясняет пустой перечень находок при живом контуре (кадр вне пути,
    чужой блок, находка без текста), `degraded` — что часть обращений к контуру не
    состоялась и сколько кадров осталось без описания.
    """

    deviations: list[CXPinnedDeviation] = Field(
        default_factory=list,
        description="Находки кадров, каждая с пином; пустой список, если видимых "
        "проблем на разобранных кадрах не было",
    )
    screens: list[CXScreenNote] = Field(
        default_factory=list,
        description="Описания разобранных кадров в порядке пути",
    )
    degraded: str = Field(
        default="",
        description="Чем плох проход: batches_failed:i/j | summaries:i/j",
    )

    @property
    def unexplained(self) -> bool:
        """Ни находок, ни единого описания кадра — проход неотличим от молчания.

        У поэкранного прохода пустой перечень находок — штатный ответ («на кадре
        нет явных проблем»), поэтому общая для веток проверка «пустота не
        объяснена» по `note` здесь бессмысленна: описания кадров и есть
        доказательство того, что их смотрели.
        """
        return not self.deviations and not self.screens and self.answered


class CXConsolidatedEvidence(NodeAnswer):
    """Улика сведённой находки: экран и дословная цитата с него."""

    screen: str = Field(
        default="",
        description="Номер экрана 1-based, как он указан в сырой находке",
    )
    quote: str = Field(
        default="",
        description="Дословная цитата улики сырой находки: пересказ не годится, "
        "придумать цитату нельзя — без неё оставляйте поле пустым",
    )


class CXConsolidatedFinding(NodeAnswer):
    """Одна уникальная находка сведения — элемент `issues` ответа модели.

    Значения мягкие (строка вместо перечислений): значение вне канона должно
    приводиться к допустимому в разборе, а не обрывать весь ответ сведения.
    """

    zone: str = Field(
        default="",
        description="Зона из сырой находки: black | red | yellow | green. "
        "Зону выбирает справочник по коду дефекта, не модель; black без "
        "дословной цитаты не ставят",
    )
    criterion: str = Field(
        default="",
        description="Блок находки — строго один из перечисленных ключей блоков",
    )
    check_key: str = Field(
        default="",
        description="Подкритерий этого блока, на который ложится находка; "
        "`none`, если находка сквозная и не ложится ни на один подкритерий",
    )
    defect_code: str = Field(
        default="", description="Код дефекта из сырой находки, без изменений"
    )
    title: str = Field(default="", description="Суть находки в 3–6 слов")
    text: str = Field(
        default="",
        description="Что видно на пути → как это трактуется → чем грозит клиенту",
    )
    rec: str = Field(
        default="",
        description="Конкретное изменение интерфейса или сценария; мета-советы "
        "вида «сделать понятнее» запрещены",
    )
    steps: list[str] = Field(
        default_factory=list,
        description="Номера экранов 1-based строго как в сырых находках, "
        "1–2 самых показательных",
    )
    evidence: list[CXConsolidatedEvidence] = Field(
        default_factory=list,
        description="Улики находки; пустой список, если дословной цитаты нет",
    )
    confidence: str = Field(
        default="",
        description="low | med | high — по лестнице доказательности; поле "
        "обязательное: без него находка штрафует блок с максимальным весом",
    )
    kind: str = Field(
        default="",
        description="ux | product_rule | noise; noise — придирка, такая находка "
        "в отчёт не идёт",
    )


class CXConsolidatedFindings(NodeAnswer):
    """Ответ сведения: единый список уникальных находок и сводка по пути."""

    issues: list[CXConsolidatedFinding] = Field(
        default_factory=list,
        description="Уникальные находки прогона; пустой список, если нарушений нет",
    )
    summary: str = Field(
        default="",
        description="2–4 предложения на ВЕСЬ путь: что за процесс и ключевые проблемы",
    )


class CXDedupGroup(NodeAnswer):
    """Одна прошлая находка и новые, которые говорят о том же самом.

    Новая находка здесь — дубль по смыслу: прошлая уже описывает эту проблему,
    и её формулировка прошла экспертизу. Группе полагается одно место в итоговом
    списке, поэтому новых в группе может быть несколько.
    """

    previous: Optional[int] = Field(
        default=None,
        description="Номер прошлой находки, 1-based, как в перечне [ПРОШЛЫЕ НАХОДКИ]",
    )
    current: list[int] = Field(
        default_factory=list,
        description="Номера новых находок о том же самом, 1-based, как в "
        "перечне [НАХОДКИ ТЕКУЩЕГО ПРОГОНА]",
    )


class CXDedupGroups(NodeAnswer):
    """Ответ сверки: что из нового дублирует прошлое и какие прошлые просить вернуть.

    `restored` — прошлые находки, которых сегодняшний прогон не нашёл: у них нет
    группы, и в итоговом списке они остаются прежними.
    """

    groups: list[CXDedupGroup] = Field(
        default_factory=list,
        description="Прошлые находки, у которых нашлись дубли среди новых",
    )
    restored: list[int] = Field(
        default_factory=list,
        description="Номера прошлых находок без дублей среди новых, 1-based",
    )
