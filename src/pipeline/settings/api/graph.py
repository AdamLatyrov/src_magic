import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Optional, Union

from langchain_classic.prompts import load_prompt
from langchain_classic.prompts.base import BasePromptTemplate
from langchain_classic.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import Field, PrivateAttr, model_validator

from src.pipeline.aef.models import ChatOptions, ChatRequest
from src.pipeline.schemas.nodes import (
    CXBlock,
    CXCriteria,
    CXCriteriaModel,
    CXDeviation,
    CXFrame,
    CXJourneyDetails,
    CXPinnedDeviation,
    CXScreenNote,
)
from src.pipeline.settings.core import CoreSettings
from src.pipeline.settings.params.nodes import CXJourneyType, CXZone
from src.pipeline.settings.paths import service_path, service_root
from src.pipeline.utils.frames import image_part

logger = logging.getLogger(__name__)


class BaseSummarizeSettings(CoreSettings):
    """Настройки узла, который спрашивает контур.

    Модель этапа названа прямо на поле `model` каждого узла: `alias` — имя его
    переменной в `.env`, значение по умолчанию — модель, отвечающая на этом контуре.
    Переменную вычитает источник конфига (`src/utils/external_sources.py`), там же и
    порядок: своя переменная, общая `LLM_MODEL` (только для поля `model`), строка
    `model:` из `configs/service_configs/agent_config.yaml`, значение поля. Имена
    переменных есть только здесь — список в другом месте пришлось бы синхронизировать
    вручную.

    Настройки правки ответа (`temperature`, `top_p`, `max_tokens` и остальные)
    объявлены здесь, в общем базовом классе, поэтому одна переменная с общим именем
    читается каждым этапом сразу. Право перебить строку конфига у неё только у
    `LLM_MODEL`: калибровка этапа живёт в конфиге, и прогон 047 показал цену такого
    перебивания — `.env` с `LLM_MAX_TOKENS=4096` снял лимит в 8192 знаков с ветки
    блоков, длинные ответы блоков обрезались на середине JSON, батчи считались
    сорванными, и прогон терял отклонения молча. Список `common_variables` — те
    самые общие имена; своё право дополнять молчащий раздел конфига они не теряют.

    Мерки кадровой нарезки (`max_images_per_request`, `min_frames_per_request`)
    объявлены здесь же, у общего базового класса: читает их каждая ветка, что режет
    путь обращениями с кадрами, а значение каждой называет секция этой ветки в
    `configs/service_configs/agent_config.yaml` — на стенде предел контура другой, и
    менять его правкой кода значило бы пересобирать образ.
    """

    prompts_dir: str
    system_prompt_filename: str = Field(default="system_prompt.yaml")
    user_prompt_filename: str = Field(default="user_prompt.yaml")
    model: str = Field(alias="LLM_MODEL", default="Qwen3.5-397b")
    temperature: Optional[float] = Field(alias="LLM_TEMPERATURE")
    top_p: Optional[float] = Field(ge=0, le=1, alias="LLM_TOP_P")
    max_tokens: Optional[int] = Field(alias="LLM_MAX_TOKENS")
    repetition_penalty: Optional[float] = Field(
        default=None, alias="LLM_REPETITION_PENALTY"
    )
    profanity_check: Optional[bool] = Field(
        default=False, alias="LLM_PROFANITY_CHECK"
    )
    reasoning_effort: str = Field(default="off", alias="LLM_REASONING_EFFORT")

    max_images_per_request: int = Field(
        default=10,
        ge=1,
        alias="GIGACHAT_MAX_IMAGES_PER_REQUEST",
        description=(
            "Документированный предел изображений одного обращения к контуру: "
            "кадры сверх него контур снимает молча"
        ),
    )

    min_frames_per_request: int = Field(
        default=4,
        ge=1,
        description=(
            "Сколько кадров минимум в последней группе: короткий хвост добирают "
            "кадрами предыдущей группы"
        ),
    )

    common_variables: ClassVar[frozenset[str]] = frozenset(
        {
            "LLM_TEMPERATURE",
            "LLM_TOP_P",
            "LLM_MAX_TOKENS",
            "LLM_REPETITION_PENALTY",
            "LLM_PROFANITY_CHECK",
            "LLM_REASONING_EFFORT",
        }
    )

    def __repr__(self) -> str:
        return (
            f"model: {self.model}, "
            f"temperature: {self.temperature}, "
            f"max_tokens: {self.max_tokens}, "
            f"repetition_penalty: {self.repetition_penalty} "
            f"profanity_check: {self.profanity_check}, "
            f"reasoning_effort: {self.reasoning_effort}"
        )

    @property
    def prompts_path(self) -> Path:
        return service_path(self.prompts_dir)

    @property
    def system_prompt_path(self) -> Path:
        return self.prompts_path.joinpath(self.system_prompt_filename)

    @property
    def user_prompt_path(self) -> Path:
        return self.prompts_path.joinpath(self.user_prompt_filename)

    @property
    def template_system_prompt(self) -> BasePromptTemplate:
        return load_prompt(path=self.system_prompt_path, encoding="utf-8")

    @property
    def template_user_prompt(self) -> BasePromptTemplate:
        return load_prompt(path=self.user_prompt_path, encoding="utf-8")

    @property
    def chat_prompt_template(self) -> Optional[ChatPromptTemplate]:
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate(
                    prompt=self.template_system_prompt
                ),
                HumanMessagePromptTemplate(prompt=self.template_user_prompt),
            ]
        )

    @property
    def stage_model(self) -> str:
        """Модель этапа — та, что лежит в настройках.

        Снимок исполнения задания накладывается на эти настройки снаружи, до
        вызова прогона (`integration/execution_snapshot.py`): прогон получает
        готовый объект и не читает никакого состояния извне. Здесь остаётся
        последнее слово того, кто настройки собрал.

        :return: модель этапа из настроек процесса
        """
        return self.model

    @property
    def chat_options(self) -> ChatOptions:
        """Параметры обращения к контуру — из полей этого же объекта.

        Калибровка этапа (`temperature`, `top_p`, `max_tokens` и остальные) может
        быть зафиксирована в задании: настройки процесса к моменту прогона могли
        смениться, и половина вопроса ушла бы не той, что при постановке. Снимок
        задания накладывается на объект до вызова — прогону остаётся прочитать
        поля. Пайплайн не знает, откуда взялись значения: из `.env`, из конфига
        или из снимка.

        :return: параметры генерации этапа
        """
        return ChatOptions(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            repetition_penalty=self.repetition_penalty,
            profanity_check=self.profanity_check,
            reasoning_effort=self.reasoning_effort,
        )

    @staticmethod
    def frames_content(frames: Sequence[CXFrame]) -> list[dict]:
        """
        Кадры частями содержимого сообщения: метка кадра, затем само изображение.

        Метка нужна, чтобы номер экрана в ответе модели совпадал с кадром: модель видит изображения
        подряд и без подписей считает их сама как ей вздумается.

        :param frames: кадры пути в порядке следования
        :return: части содержимого сообщения для контракта с изображениями
        """
        steps = ", ".join(str(frame.step) for frame in frames)
        parts: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"\n\n## Кадры пути\n\n"
                    f"Приложено кадров: {len(frames)} (номера: {steps}). "
                    f"Каждый подписан строкой `[кадр N]` — в поле step указывай этот номер."
                ),
            }
        ]
        for frame in frames:
            parts.append({"type": "text", "text": f"\n[кадр {frame.step}]"})
            parts.append(image_part(frame))
        return parts

    def window_frames(self, requested: int, total: int) -> int:
        """Сколько кадров уходит в одном обращении: больше лимита контур не видит.

        Ветка вправе просить весь путь одним вопросом (`0`), но транспорт этого не
        переживает: сверх документированного лимита контур снимает кадры молча, и
        «формы нет» оказывается выводом об экранах, до которых вопрос не дошёл —
        на пути из одиннадцати кадров так терялся последний экран. Лимит поэтому
        потолок и для окна, и для «всего пути»; сам он стоит в секции ветки
        (`max_images_per_request`), а не в памяти кода.

        :param requested: сколько кадров просит ветка; 0 — весь путь
        :param total: число кадров пути
        :return: размер окна обращения, всегда не меньше одного кадра
        """
        size = requested if requested > 0 else total
        return max(1, min(size, self.max_images_per_request))

    def group_slices(self, total: int, size: int) -> list[slice]:
        """Границы групп обращения: по максимуму, короткий хвост — с перехлёстом.

        Группы набиваются с начала пути до лимита, а не делятся ровно: ровный
        дележ на одиннадцати кадрах при лимите десять давал две группы по шесть,
        то есть второй вопрос платился за половину пути, уже спрошенную первым.
        Хвост короче `min_frames_per_request` отдельное обращение здесь не
        составляет: на одном-двух экранах модель отвечает «нет данных для
        проверки» — вердикт по всему пути, оплаченный кадром. Хвост поэтому
        дотягивается до `min_frames_per_request` кадрами, которые легли и в
        предыдущее обращение: пересечение двух батчей дешевле пустого вердикта, а
        повтор той же находки снимает сведение ветки
        (`pipeline.utils.block_findings.merge_batches`, `pipeline.agent.autofill.nodes._merge`).

        :param total: сколько элементов делят
        :param size: сколько элементов умещается в одну группу
        :return: срезы групп в порядке перечня; последний может пересекаться с
            предыдущим
        """
        if total <= 0:
            return []
        if total <= size:
            return [slice(0, total)]
        tail = min(self.min_frames_per_request, size)
        groups: list[slice] = []
        start = 0
        while total - start > size:
            groups.append(slice(start, start + size))
            start += size
        short = total - start < tail
        groups.append(
            slice(total - tail, total) if short else slice(start, total)
        )
        return groups

    @staticmethod
    def window_note(frames: Sequence[CXFrame], total: int) -> str:
        """Пояснение к окну кадров: часть это пути или весь путь.

        Без него ветка, которой показали десять экранов из сорока, ставит
        «этого в пути нет» там, где предмет просто остался за окном, и описывает
        маршрут короче, чем он есть. Транспорт здесь не помощник: сверх
        документированного лимита контур кадры снимает молча, и без этой строки
        обрезанный запрос неотличим от запроса про весь путь.

        :param frames: кадры этого обращения
        :param total: число кадров пути — по нему видно, что показано не всё
        :return: блок вопроса, пустая строка на целом пути
        """
        if total <= 0 or len(frames) >= total:
            return ""
        return (
            "\n\n[ЧАСТЬ ПУТИ]\n"
            f"Приложено кадров: {len(frames)} из {total} этого пути "
            f"(номера {frames[0].step}–{frames[-1].step}); остальная часть пути "
            "в это обращение не легла. Основание «этого экрана в пути нет» здесь "
            "запрещено: экран мог остаться за окном. Число кадров этого окна — не "
            "длина пути, и вывод о всём маршруте по нему не делается."
        )

    @staticmethod
    def with_frames(
        user: BaseMessage,
        frames: Sequence[CXFrame],
        total: int = 0,
    ) -> HumanMessage:
        """USER-сообщение становится содержимым с изображениями.

        :param user: собранное текстовое сообщение пользователя
        :param frames: кадры пути — уходят изображениями после текста сообщения
        :param total: число кадров пути — nonzero, когда приложено окно, а не весь
            путь
        :return: сообщение того же содержания, но с приложенными кадрами
        """
        content: list[dict] = [{"type": "text", "text": user.content}]
        note = BaseSummarizeSettings.window_note(frames, total)
        if note:
            content.append({"type": "text", "text": note})
        content += BaseSummarizeSettings.frames_content(frames)
        return HumanMessage(content=content)


class ScanJourneyFrameSettings(BaseSummarizeSettings):
    """Настройки узла «что на кадре»: окно `[i-2, i-1, i]`, ответ про один кадр.

    Кадр разбирается в компании двух предыдущих: они отвечают на «что было до», а
    ответ модель обязана дать про разбираемый кадр — он подписан отдельно. Кадров
    после разбираемого в окне нет: скан идёт по одному кадру и спрашивает вперёд
    незачем, а каждое лишнее изображение — это оплаченный вопрос и размытая
    привязка ответа к кадру. Отсюда и потолок ответа выше, чем у распознавания
    пути: кадр ещё и переписывается текстом дословно.
    """

    model: str = Field(default="Qwen3.5-397b", alias="LLM_SCAN_JOURNEY_FRAME")

    @staticmethod
    def window(frames: Sequence[CXFrame], index: int) -> list[CXFrame]:
        """Кадры окна разбираемого кадра `index` (с нуля): он сам и два до него.

        Окно берётся по тому перечню кадров, который скану вообще остался: кадр,
        снятый фильтром (`CXFrameReadout.need_filter`), в него не входит, и тогда
        соседом становится кадр через пропуск, а не снятый экран. На краю пути окно
        укорачивается, а не дополняется дублем: повторное изображение стоит токенов
        и ничего не добавляет к контексту.

        :param frames: кадры, оставшиеся после фильтра, в порядке пути
        :param index: положение разбираемого кадра в этом перечне, с нуля
        :return: разбираемый кадр и два предыдущих, в порядке пути
        """
        if not frames:
            raise ValueError("у пути нет кадров — сканировать нечего")
        if not 0 <= index < len(frames):
            raise IndexError(
                f"кадра {index} нет в пути из {len(frames)} кадров"
            )
        return list(frames[max(index - 2, 0) : index + 1])

    @staticmethod
    def window_content(
        frames: Sequence[CXFrame], center_step: int
    ) -> list[dict]:
        """Кадры окна частями содержимого: разбираемый отделён от соседей.

        Подпись при каждом изображении — как в `frames_content`: без неё номера
        кадров перестают совпадать с файлами пути, а без разделения ролей модель
        описывает всё окно сразу вместо одного кадра. Роли названы действиями
        («переписывай» / «не переписывать»): в прогоне 009 кадр вернулся текстом
        соседа, и нейтральное «сосед» этого не предотвратило. Последней идёт
        строка после всех изображений — напоминание, к кому из них вопрос.

        :param frames: кадры окна в порядке пути
        :param center_step: номер разбираемого кадра
        :return: части содержимого сообщения с подписанными кадрами
        """
        steps = ", ".join(str(frame.step) for frame in frames)
        parts: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"\n\n## Кадры окна\n\n"
                    f"Приложено кадров: {len(frames)} (номера: {steps}). "
                    f"Разбираемый кадр подписан `[кадр {center_step} — "
                    f"разбираемый]`; отвечай по нему, соседи только для "
                    f"контекста."
                ),
            }
        ]
        for frame in frames:
            if frame.step == center_step:
                role = "разбираемый — текст переписывай с него"
            else:
                role = "сосед — только контекст, с него не переписывай"
            parts.append(
                {"type": "text", "text": f"\n[кадр {frame.step} — {role}]"}
            )
            parts.append(image_part(frame))
        parts.append(
            {
                "type": "text",
                "text": (
                    f"\n\nИзображения окна кончились. Вопрос про кадр "
                    f"{center_step}: с него назови `heading` и перепиши "
                    f"`text_on_screen`. Предыдущие кадры нужны были ровно затем, "
                    f"чтобы понять, что было до; их надписи в ответ не входят."
                ),
            }
        )
        return parts

    def variables(
        self,
        material: str,
        context: Sequence[str] = (),
    ) -> dict:
        """Переменные вопроса: заметка эксперта и уже разобранные кадры.

        :param material: заметка эксперта к материалам пути
        :param context: описания кадров, разобранных до этого, в порядке пути
        :return: переменные шаблона
        """
        return {
            "material": material.strip()
            or "Заметки нет: путь описан только кадрами.",
            "context": "\n".join(context)
            or "Предыдущих кадров нет: разбираемый кадр открывает путь.",
        }

    @staticmethod
    def reask_text(missing: Sequence[str] = (), doubt: str = "") -> str:
        """Требование к повторному ответу: чего не хватило или что в ответе не так.

        Переспрос обязан отличаться от первого вопроса. При `temperature = 0`
        дословно тот же вопрос возвращает дословно тот же ответ, и весь смысл
        повторного обращения — в перемене вопроса: здесь в нём названо, что именно
        не принято и чем это грозит кадру.

        :param missing: имена пустых обязательных полей ответа
        :param doubt: чем неудовлетворён ответ при заполненных полях — текст
            переписан не с разбираемого кадра
        :return: текст требования, пустая строка, переспрашивать нечего
        """
        if not missing and not doubt:
            return ""
        parts = [
            "\n\n## Требование к ответу\n\nПервый ответ на этот вопрос не принят."
        ]
        if missing:
            names = ", ".join(f"`{name}`" for name in missing)
            parts.append(
                f"Пустыми остались {names}. Ответь заново по тому же окну кадров "
                "и заполни названные поля по разбираемому изображению — не по "
                "соседним кадрам и не по контексту пути. `text_on_screen` — весь "
                "текст кадра до нижнего края: мелкая сноска под кнопкой, подпись "
                "под полем и текст согласия входят в него, а сокращать их "
                "многоточием нельзя. Пустой строкой поле остаётся только там, "
                "где читать нечего или надпись не разбирается, — и тогда "
                "`confidence = low`. `need_filter = true` из-за пустого поля не "
                "ставится: кадр остаётся в прогоне при любом ответе, фильтр — "
                "только для кадра, где интерфейса нет вовсе."
            )
        if doubt:
            parts.append(
                f"Кроме того, {doubt} — значит, переписан соседний кадр окна, а "
                "не разбираемый. Начни с `heading`: назови самую крупную надпись "
                "того изображения, которое подписано как разбираемое, и перепиши "
                "текст именно с него, построчно и сверху вниз. Надписей соседей "
                "в ответе быть не должно, даже если они читаются лучше."
            )
        return "\n\n".join(parts)

    def messages(
        self,
        material: str,
        window: Sequence[CXFrame],
        center_step: int,
        context: Sequence[str] = (),
        missing: Sequence[str] = (),
        doubt: str = "",
    ) -> list[BaseMessage]:
        """Сообщения описания кадра: вопрос, контекст пути, затем кадры окна.

        :param material: заметка эксперта к материалам пути
        :param window: кадры окна в порядке пути
        :param center_step: номер разбираемого кадра
        :param context: описания разобранных ранее кадров, не более `context_last`
        :param missing: пустые поля предыдущего ответа — тогда к вопросу
            добавляется требование к повторному ответу
        :param doubt: чем неудовлетворён предыдущий ответ при заполненных полях
        :return: сообщения запроса
        """
        system, user = self.chat_prompt_template.format_messages(
            **self.variables(material, context)
        )
        content: list[dict] = [{"type": "text", "text": user.content}]
        reask = self.reask_text(missing, doubt)
        if reask:
            content.append({"type": "text", "text": reask})
        content += self.window_content(window, center_step)
        return [system, HumanMessage(content=content)]

    def chat_request(
        self,
        material: str,
        window: Sequence[CXFrame],
        center_step: int,
        context: Sequence[str] = (),
        missing: Sequence[str] = (),
        doubt: str = "",
    ) -> ChatRequest:
        """Запрос описания кадра: параметры модели, вопрос и окно кадров.

        :param material: заметка эксперта к материалам пути
        :param window: кадры окна в порядке пути
        :param center_step: номер разбираемого кадра
        :param context: описания разобранных ранее кадров
        :param missing: пустые поля предыдущего ответа — повод переспросить
            по существу, а не тем же вопросом
        :param doubt: чем неудовлетворён предыдущий ответ (переписан не тот кадр)
        :return: запрос для провайдера контура
        """
        return ChatRequest(
            options=self.chat_options,
            model=self.stage_model,
            messages=self.messages(
                material, window, center_step, context, missing, doubt
            ),
        )


class SummarizeJourneySettings(BaseSummarizeSettings):
    """Настройки узла сведения описаний кадров в атрибуцию пути.

    Кадры сюда не уходят: их уже описал скан, и узел работает с текстом — по
    описаниям он называет компанию, продукт, сценарий и тип пути. Вердикт
    «путь ли это» берётся не отсюда, а из описаний кадров, поэтому модель
    здесь описывает путь, а не решает, запускать ли оценку.
    """

    model: str = Field(default="Qwen3.5-397b", alias="LLM_SUMMARIZE_JOURNEY")

    @property
    def journey_values(self) -> str:
        """Допустимые типы пути: их перечисляет схема, а не текст промпта."""
        return ", ".join(item.value for item in CXJourneyType)

    def variables(
        self,
        material: str,
        frames_text: str,
        coverage: str,
    ) -> dict:
        """Переменные вопроса: текстовый след пути и сколько пути он покрывает.

        :param material: заметка эксперта к материалам пути
        :param frames_text: текстовый след кадров — описания, фазы и текст экранов
        :param coverage: сколько кадров описано из скольки, с причиной обрыва
        :return: переменные шаблона
        """
        return {
            "journey_values": self.journey_values,
            "material": material.strip()
            or "Заметки нет: путь описан только кадрами.",
            "frames_text": frames_text,
            "coverage": coverage,
        }

    def messages(
        self,
        material: str,
        frames_text: str,
        coverage: str,
    ) -> list[BaseMessage]:
        """Сообщения сведения: вопрос и текстовый след пути, без изображений.

        :param material: заметка эксперта к материалам пути
        :param frames_text: текстовый след кадров
        :param coverage: покрытие пути описаниями
        :return: сообщения запроса
        """
        return list(
            self.chat_prompt_template.format_messages(
                **self.variables(material, frames_text, coverage)
            )
        )

    def chat_request(
        self,
        material: str,
        frames_text: str,
        coverage: str,
    ) -> ChatRequest:
        """Запрос сведения пути: тексты кадров, кадров в нём нет.

        :param material: заметка эксперта к материалам пути
        :param frames_text: текстовый след кадров
        :param coverage: покрытие пути описаниями
        :return: запрос для провайдера контура
        """
        return ChatRequest(
            options=self.chat_options,
            model=self.stage_model,
            messages=self.messages(material, frames_text, coverage),
        )


class JourneyExplorationSettings(CoreSettings):
    """Сканирование пути кадр за кадром.

    Числа правил скана живут здесь, а не в узлах: досрочная остановка и длина
    контекста — это режим прогона (сколько платим и как далеко помним), а не
    параметры модели. `stop_after=0` выключает досрочную остановку совсем,
    `context_last=0` — спрашивать каждый кадр только по его окну.
    """

    stop_after: int = Field(
        default=5,
        ge=0,
        alias="JOURNEY_SCAN_STOP_AFTER",
        description=(
            "Сколько первых разобранных кадров не должны оказаться клиентским "
            "путём, чтобы оборвать скан; 0 — правило выключено"
        ),
    )
    context_last: int = Field(
        default=10,
        ge=0,
        alias="JOURNEY_SCAN_CONTEXT_LAST",
        description=(
            "Сколько описаний разобранных кадров едет в следующем вопросе; "
            "0 — контекста нет"
        ),
    )
    scan_frame: ScanJourneyFrameSettings


class CheckCxCriteriaSettings(BaseSummarizeSettings):
    """Настройки ветки блоков по кадрам: блок спрашивается изображениями пути.

    Здесь же живут справочник, канон подкритериев, зоны дефектов и разбор
    ответа: ответы блоков ложатся в одну форму `CXBlockFindings`, независимо от
    того, сколькими обращениями блок спрашивали.
    """

    model: str = Field(default="Qwen3.5-397b", alias="LLM_CHECK_CX_CRITERIA")

    criteria_model: str = Field(
        default="configs/service_configs/criteria_model_v3_2.json"
    )

    frames_per_request: int = Field(
        default=10,
        alias="LLM_MAX_IMAGES_PER_REQUEST",
        ge=0,
        description=(
            "Сколько кадров уходит блоку в одном обращении: путь длиннее батча "
            "спрашивается несколькими обращениями подряд; 0 — спрашивать весь "
            "путь одним вопросом. Меньше `min_frames_per_request` кадров в "
            "обращении не бывает: короткий хвост добирают кадрами предыдущего"
        ),
    )

    _block: Optional[CXBlock] = PrivateAttr(default=None)
    _criteria_model: Optional[CXCriteriaModel] = PrivateAttr(default=None)

    @property
    def block(self) -> Optional[CXBlock]:
        return self._block

    @property
    def criteria_model_path(self) -> str:
        """Подпись справочника прогона — для сообщений об отказе.

        Сам справочник приходит из базы и подкладывается сюда
        (`bind_criteria_model`): файла нет, и путь здесь только называет,
        какая редакция ожидалась.
        """
        return self.criteria_model

    def bind_criteria_model(self, model: CXCriteriaModel) -> None:
        """Подкладывает справочник из базы на место собранного из настроек.

        Настройки собираются моделью pydantic, а справочник читается
        асинхронно и только в базе, поэтому он приходит уже после сборки.
        """
        self._criteria_model = model

    def load_criteria_model(self) -> CXCriteriaModel:
        """Справочник прогона из базы.

        :raises ValueError: справочник не подложен — прогон запущен мимо
            приложения, которое читает его из базы
        """
        if self._criteria_model is None:
            raise ValueError(
                "справочник не загружен из базы: прогон, которому он нужен, "
                "собирается приложением, а не настройками"
            )
        return self._criteria_model

    def blocks(self) -> list[CXBlock]:
        return self.load_criteria_model().blocks

    def for_block(
        self, block: Union[CXBlock, str]
    ) -> "CheckCxCriteriaSettings":
        bound = self.model_copy()

        if isinstance(block, str):
            bound._block = self.load_criteria_model().block(block)
        else:
            bound._block = block

        return bound

    @property
    def subcriteria(self) -> dict[str, str]:
        """Подкритерия блока в порядке справочника: ключ → текст."""
        return self._required_block().subcriteria

    @property
    def required_keys(self) -> list[str]:
        """Канон-порядок подкритериев — его обязан повторить ответ блока."""
        return self._required_block().required_keys

    @property
    def defect_zones(self) -> dict[str, str]:
        """Зона каждого дефекта блока: держит её справочник, а не настройки."""
        return self._required_block().defect_zones

    MEASURE_IN_QUESTION: ClassVar[int] = 400
    MEASURE_CLAUSES: ClassVar[re.Pattern[str]] = re.compile(r"[;\n()]")
    MEASURE_DIGITS: ClassVar[re.Pattern[str]] = re.compile(r"\d")
    MEASURE_WORDS: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(один|одна|одно|два|две|три|четыре|пять|шесть|семь|восемь|девять|"
        r"десять|пара|несколько)\b"
    )

    @classmethod
    def _measure_clauses(cls, source: str) -> list[str]:
        """Места строки справочника, где названа величина, в порядке появления.

        Мера — то, чем меряют: «4 — лучший UX, 6 — средний, 8+ — плохо», «два–три
        чекбокса». Всё остальное — пересказ того, как выглядит порядок без порока, и
        модель возвращает его в `note` вместо проверки по кадрам.

        :param source: суть и текст подкритерия одним текстом
        :return: клаузы с числом или словесным счётом, без повторов
        """
        seen: set[str] = set()
        clauses: list[str] = []
        for raw in cls.MEASURE_CLAUSES.split(source):
            clause = " ".join(raw.split()).strip(" .,:-")
            key = clause.casefold()
            if not clause or key in seen:
                continue
            if not (
                cls.MEASURE_DIGITS.search(clause)
                or cls.MEASURE_WORDS.search(clause)
            ):
                continue
            seen.add(key)
            clauses.append(clause)
        return clauses

    @classmethod
    def criteria_measure(cls, essence: str, text: str = "") -> str:
        """Мера подкритерия из справочника: только места, где названа величина.

        Справочник пишет суть как описание порядка без порока, и целая строка
        такой сути становилась готовым ответом: модель возвращала её дословно в
        `note`, не глядя на кадры. Первая версия брала начало строки до потолка
        знаков и называла мерой то, что стоит в начале, — а шкала справочника
        живёт в скобках и в конце. В прогоне 047 счётная мера «8+ экранов —
        плохо» обрезалась и не доехала до вопроса: блоку было не с чем сравнить
        одиннадцать экранов пути, и он поставил "ok" пути вдвое длиннее меры.
        Поэтому мера собирается из сути И текста подкритерия целыми клаузами и
        режется по общему потолку знаков уже на них.

        :param essence: строка сути из справочника
        :param text: строка текста подкритерия — в нём шкала чаще всего и стоит
        :return: мера до `MEASURE_IN_QUESTION` знаков, пустая строка без числа
        """
        parts: list[str] = []
        left = cls.MEASURE_IN_QUESTION
        for clause in cls._measure_clauses(f"{essence}\n{text}"):
            if len(clause) > left:
                clause = f"{clause[:left].rstrip()}…"
            parts.append(clause)
            left -= len(clause) + 2
            if left <= 0:
                break
        return "; ".join(parts)

    @property
    def criteria_listing(self) -> str:
        """Подкритерии блока перечнем, сгруппированным по тяжести дефектов.

        Порядок здесь — не косметика: перечнем по зонам модель получает ту же
        градацию, которой меряет подкритерий человек («вот это блокер, вот это
        замечание»), и выбирает статус по увиденному, а не по общему впечатлению.
        Плоский перечень из сорока кодов читается одинаковым шумом, и блок
        отвечает "ok" на весь список.
        """
        block = self._required_block()
        lines: list[str] = []
        for criteria in block.criteria:
            lines += self._criteria_lines(criteria, block)
        return "\n".join(lines).strip()

    def _criteria_lines(
        self, criteria: CXCriteria, block: CXBlock
    ) -> list[str]:
        """Перечень одного подкритерия: мера и нарушения по тяжести.

        Порядок задаёт сама шкала (`CXZone.heaviest_first`), а пояснение зоны
        приходит из неё же (`CXZone.note`): лесенка тяжести в вопросе и линейка
        баллов не могут разъехаться, потому что у них один источник. В вопрос
        зона уходит заглавными буквами (`BLACK НАРУШЕНИЕ (…):`).

        :param criteria: подкритерий справочника
        :param block: блок подкритерия — его суть не повторяют мерой
        :return: строки перечня подкритерия, последняя — пустая
        """
        lines: list[str] = [f"### {criteria.key} — {criteria.text}"]
        essence = criteria.essence if criteria.essence != block.essence else ""
        measure = self.criteria_measure(essence, criteria.text)
        if measure:
            lines.append(f"Мера: {measure}")
        for zone in CXZone.heaviest_first():
            defects = [
                defect
                for defect in criteria.defects
                if CXZone.parse(defect.zone) is zone
            ]
            if not defects:
                continue
            lines.append(f"{zone.upper()} НАРУШЕНИЕ ({zone.note}):")
            lines += [
                (
                    f"- {defect.code} — {defect.defect}. "
                    f"Проявляется так: {defect.symptom}"
                )
                for defect in defects
            ]
        lines.append("")
        return lines

    def variables(
        self,
        material: str,
        details: Optional[CXJourneyDetails] = None,
    ) -> dict:
        block = self._required_block()
        return {
            "block_key": block.key,
            "block_name": block.name,
            "block_essence": block.essence,
            "criteria_count": len(block.criteria),
            "defect_count": len(self.defect_zones),
            "required_keys": json.dumps(
                self.required_keys, ensure_ascii=False
            ),
            "criteria": self.criteria_listing,
            "material": material,
            "journey": details.context() if details else "",
        }

    @staticmethod
    def batch_note(
        frames: Sequence[CXFrame],
        total: int,
        number: int,
        count: int,
        steps: Sequence[int] = (),
        route: str = "[СЧЁТ ПУТИ] и [ТЕКСТЫ КАДРОВ]",
    ) -> str:
        """Метка обращения к контуру: который батч из скольких и какие кадры в нём.

        Блок спрашивается последовательно — одним обращением на пачку кадров, — и
        без метки вопрос про десять экранов из сорока неотличим от вопроса про весь
        путь: по своей пачке модель ставила «этого в пути нет» там, где экран просто
        остался за батчем, и объявляла путь короче, чем он есть. Здесь счёт
        обращениям ведёт сама ветка блоков, поэтому номер стоит в вопросе, а не
        только в журнале: по нему ход проверки блока читают и лаборатория, и разбор
        прогона.

        :param frames: кадры этого обращения; пусто — когда изображений нет
        :param total: число кадров пути — по нему видно, что показано не всё
        :param number: который по счёту батч, от одного
        :param count: сколько батчей легло на этот блок
        :param steps: номера кадров текстом — на случай вопроса без изображений
        :param route: блоками вопроса, где лежат факты всего маршрута; ветка
            описаний называет свои блоки — ссылки на чужие ей не к чему
        :return: строка метки, пустой она не бывает
        """
        taken = [frame.step for frame in frames] or sorted(steps)
        gaps = len(taken) > 1 and taken[-1] - taken[0] + 1 > len(taken)
        if not taken:
            edges = "кадров нет"
        elif len(taken) == 1:
            edges = f"кадр {taken[0]}"
        elif gaps:
            edges = "кадры " + ", ".join(str(step) for step in taken)
        else:
            edges = f"кадры {taken[0]}–{taken[-1]}"
        whole = total > 0 and len(taken) >= total
        head = f"Батч {number} из {count}" if count > 1 else "Батч 1 из 1"
        if whole or count <= 1:
            return f"{head}: {edges} — весь путь в этом обращении."
        return (
            f"{head}: {edges} из {total} кадров пути. Остальная часть пути "
            "проверяется другими обращениями этого же блока, и их ответов здесь "
            "нет: основание «такого экрана в пути нет» по всему маршруту здесь "
            f"не работает — экран мог остаться за батчем. Вывод о всём маршруте "
            f"делают по {route}, а не по тому, что легло в это обращение."
        )

    @staticmethod
    def tally_block(tally: str) -> str:
        """Счёт пути телом блока `[СЧЁТ ПУТИ]`; пустого блока вопроса нет.

        Число считает код: на десяти приложенных кадрах модель счёт не вела и
        ставила "ok" пути из одиннадцати экранов. Пустая строка здесь значила бы
        «счёт есть, но пустой», и это неотличимо от непроверенного пути.

        :param tally: счёт пути (`pipeline.utils.journey.screen_tally`) — посчитан кодом
        :return: строка для блока вопроса
        """
        return tally.strip() or (
            "счёта нет: скан не переписал ни одной надписи экрана, длину пути "
            "по этому материалу не проверить"
        )

    @staticmethod
    def texts_block(trace: str) -> str:
        """Тексты кадров телом блока `[ТЕКСТЫ КАДРОВ]`; пустого блока вопроса нет.

        След отдаёт модели то, что скан уже прочитал с каждого экрана: по нему
        считают длину пути и число полей там, где кадров обращения не хватило, —
        уликой при этом остаётся надпись приложенного кадра.

        :param trace: перечень текстов кадров (`pipeline.utils.journey.screen_texts`)
        :return: строка для блока вопроса
        """
        return trace.strip() or (
            "текстов нет: скан не переписал ни одного кадра пути"
        )

    def frames_variables(
        self,
        material: str,
        frames: Sequence[CXFrame] = (),
        details: Optional[CXJourneyDetails] = None,
        total: int = 0,
        trace: str = "",
        tally: str = "",
        number: int = 1,
        count: int = 1,
    ) -> dict:
        """Переменные вопроса блока по кадрам: справочник плюс этот батч пути.

        :param material: текстовые материалы пути
        :param frames: кадры этого обращения
        :param details: атрибуция пути
        :param total: число кадров пути
        :param trace: тексты кадров всего пути
        :param tally: счёт пути, посчитанный кодом
        :param number: который по счёту батч
        :param count: сколько батчей на блок
        :return: переменные шаблона
        """
        return self.variables(material, details) | {
            "batch": self.batch_note(frames, total, number, count),
            "tally": self.tally_block(tally),
            "frames_text": self.texts_block(trace),
        }

    def messages(
        self,
        material: str,
        frames: Sequence[CXFrame] = (),
        details: Optional[CXJourneyDetails] = None,
        total: int = 0,
        trace: str = "",
        tally: str = "",
        number: int = 1,
        count: int = 1,
    ) -> list[BaseMessage]:
        """Сообщения блока; с кадрами USER-сообщение становится содержимым с изображениями.

        :param material: текстовые материалы пути
        :param frames: кадры этого обращения — уходят модели изображениями
        :param details: атрибуция пути — по ней блок решает, к какому домену путь
        :param total: число кадров пути — по нему батч подписан частью маршрута
        :param trace: тексты кадров всего пути (`pipeline.utils.journey.screen_texts`) —
            по ним блок считает экраны и поля там, где кадров обращения не хватило
        :param tally: счёт пути (`pipeline.utils.journey.screen_tally`) — число экранов,
            которое код посчитал сам, чтобы модель не отделывалась впечатлением
        :param number: который по счёту батч этого блока
        :param count: сколько батчей на блок
        :return: сообщения запроса
        """
        system, user = self.chat_prompt_template.format_messages(
            **self.frames_variables(
                material,
                frames,
                details,
                total,
                trace,
                tally,
                number,
                count,
            )
        )
        if not frames:
            return [system, user]
        return [system, self.with_frames(user, frames)]

    def chat_request(
        self,
        material: str,
        frames: Sequence[CXFrame] = (),
        details: Optional[CXJourneyDetails] = None,
        total: int = 0,
        trace: str = "",
        tally: str = "",
        number: int = 1,
        count: int = 1,
    ) -> ChatRequest:
        return ChatRequest(
            options=self.chat_options,
            model=self.stage_model,
            messages=self.messages(
                material,
                frames,
                details,
                total,
                trace,
                tally,
                number,
                count,
            ),
        )

    def batches(self, frames: Sequence[CXFrame]) -> list[list[CXFrame]]:
        """Кадры групп обращения к контуру: батчи по N кадров всего пути.

        Батчи идут подряд, по максимуму кадров, и только последний может
        пересекаться с предыдущим — так короткий хвост пути не становится
        обращением из одного экрана (см. `group_slices`). Номера подписываются
        глобально (`frames_content` читает номера самих кадров), поэтому вердикты
        батчей складываются без пересборки
        (`pipeline.utils.block_findings.merge_batches`). «Весь путь одним вопросом»
        здесь тоже режется лимитом изображений: молча обрезанный запрос
        неотличим от вопроса про весь путь.

        :param frames: кадры пути в порядке следования
        :return: непустые группы кадров в порядке пути
        """
        if not frames:
            return []
        size = self.window_frames(self.frames_per_request, len(frames))
        return [
            list(frames[group])
            for group in self.group_slices(len(frames), size)
        ]

    def _required_block(self) -> CXBlock:
        if self._block is None:
            raise ValueError(
                "Настройки check_cx_criteria не привязаны к блоку — "
                "возьмите копию через for_block()"
            )
        return self._block


class CheckJourneyFrictionSettings(BaseSummarizeSettings):
    """Одна проверка проблем пути: свой промпт, свои меры, окно кадров.

    Проверок шесть, и каждая отвечает только за свой предмет: длина маршрута,
    лишний ручной ввод, тупики, запоздалое раскрытие условий, понятность
    навигации и расхождение обещаний с делом. Различаются они промптом и мерами,
    а вопрос собирается одним кодом — тем же, что задаёт соседняя ветка кадров
    (`with_frames`).

    Справочника блоков в вопросе проверки нет нарочно: где именно описано
    найденное, заранее неизвестно, и решение об адресе принимает разметчик после
    того, как все шесть проверок ответили (`MapJourneyFrictionsSettings`). Модель,
    которой на шаге поиска показывают разделы каталога, начинает подгонять
    увиденное под них и снимать находку, для которой подходящий раздел не нашёлся.

    Путь проверка видит окнами: контур держит в поле зрения документированные
    десять изображений и молча снимает лишние, поэтому ответ на сорок кадров — это
    ответ по первым десяти (`batches`). Структура всего маршрута при этом остаётся
    в вопросе текстом — перечень экранов даёт `slice_note`.
    """

    check_key: str = Field(
        description=(
            "Ключ проверки: частью имени узла его делает сборщик ветки, "
            "плюс ключ состояния и часть журнала"
        )
    )
    name: str = Field(
        default="", description="Что проверяется — короткая подпись для лога"
    )
    units: list[str] = Field(
        default_factory=list,
        description=(
            "Шифры мер проверки — те же, что перечислены в разделе «Задание» её "
            "промпта: по ним разбор видит, какую меру проверка не назвала ни на "
            "одном кадре"
        ),
    )
    frames_per_request: int = Field(
        default=10,
        alias="LLM_MAX_IMAGES_PER_REQUEST",
        ge=0,
        description=(
            "Сколько кадров уходит в одном обращении: путь длиннее окна "
            "спрашивается несколькими обращениями; 0 — спрашивать весь путь "
            "одним вопросом"
        ),
    )
    model: str = Field(default="Qwen3.5-397b", alias="LLM_JOURNEY_FRICTIONS")

    def batches(self, frames: Sequence[CXFrame]) -> list[list[CXFrame]]:
        """Кадры групп обращения к контуру: весь путь или окна по N кадров.

        Окна идут подряд и не пересекаются: номера подписываются глобально
        (`frames_content` читает номера самих кадров), поэтому находка остаётся на
        своём экране без пересборки. Делит путь именно эта ветка, а не транспорт:
        ответ проверки — JSON с перечнем, и склеивать ответы частей на уровне
        сообщений значило бы чинить разбор там, где достаточно сложить находки.

        :param frames: кадры пути в порядке следования
        :return: непустые группы кадров в порядке пути
        """
        if not frames:
            return []
        size = self.window_frames(self.frames_per_request, len(frames))
        return [
            list(frames[i : i + size]) for i in range(0, len(frames), size)
        ]

    @staticmethod
    def slice_note(frames: Sequence[CXFrame], total: int, route: str) -> str:
        """Пояснение к окну кадров: часть это пути или весь путь.

        Без него проверка, которой показали десять экранов из сорока, отвечает о
        начале и конце маршрута по десяти — и снимает находку «путь длинный» тем,
        что дальше неё просто ничего не видно. Перечень экранов всего пути даёт
        структуру текстом: маршрут по нему виден целиком, а улика по-прежнему
        берётся только с приложенного кадра.

        :param frames: кадры этого обращения
        :param total: число кадров пути — по нему видно, что показано не всё
        :param route: экраны всего пути одной строкой, пустая — если след не собран
        :return: блок вопроса, пустая строка на целом пути без следа
        """
        whole = len(frames) >= total
        if whole and not route.strip():
            return ""
        head = (
            f"Приложено кадров: {len(frames)} из {total} этого пути "
            f"(номера {frames[0].step}–{frames[-1].step})."
            if not whole
            else f"Приложены все {total} кадров пути."
        )
        tail = (
            "\nВесь путь по экранам: "
            f"{route.strip()}\nДлину маршрута, начало и конец сверяй по этому "
            "перечню; улику бери только с приложенного кадра — экрана из "
            "перечня без изображения не существует."
            if route.strip()
            else ""
        )
        return f"\n\n[ЧАСТЬ ПУТИ]\n{head}{tail}"

    @staticmethod
    def values_note(values: str = "", here: str = "") -> str:
        """Величины пути: и приложенных кадров, и той части, что в окно не легла.

        Мерам вроде «продолжение открыто при значении, которое его исключает» нужны
        два экрана, а окно показывает десять: остаток на счёте остался в первом окне,
        сумма вклада — во втором, и проверка снимает меру, потому что пары не видела.
        Вторая половина блока нужна по другой причине: на окне в шесть экранов модель
        переписывает величины первого кадра всем остальным и сводит меру вслепую, не
        прочитав ни одного кадра. Здесь у каждого кадра его собственные числа — те же,
        что сняты узлом распознавания с этого же изображения.

        :param values: кадры вне окна с их величинами (`value_trace`)
        :param here: кадры этого окна с их величинами (`value_trace(..., only=)`)
        :return: блок вопроса; величин нет — блок говорит об этом прямо
        """
        outside = values.strip()
        inside = here.strip()
        if not outside and not inside:
            return (
                "\n\n[ВЕЛИЧИНЫ ПУТИ]\n"
                "Ни одной величины распознавание с кадров этого пути не переписало: "
                "ни сумм, ни ставок, ни процентов, ни сроков, ни границ «до/от». "
                "Мерам, которым нужны два числа с разных концов пути, здесь нечего "
                "мерить — так и напиши в `note`. Число, которого нет на приложенном "
                "кадре, выдумано, каким бы правдоподобным для этого продукта оно ни "
                "было."
            )
        parts = ["\n\n[ВЕЛИЧИНЫ ПУТИ]"]
        parts.append(
            "Величины переписаны узлом распознавания дословно с того же кадра:"
        )
        if inside:
            parts.append(
                "\nна приложенных кадрах (изображение этого кадра перед "
                f"тобой):\n{inside}"
            )
        if outside:
            parts.append(
                "\nна остальных кадрах пути (изображением ты их не видишь):"
                f"\n{outside}"
            )
        parts.append(
            "\n`values` разобранного кадра — величины этого кадра, а не всего "
            "окна: одинаковый набор величин у нескольких кадров значит, что "
            "кадры не читали.\n"
            "Сверяй величину по всему пути, а не внутри окна: одна и та же сумма "
            "на двух кадрах названа по-разному; продолжение открыто при значении, "
            "которое его исключает (остаток меньше оформляемой суммы); обещанное на "
            "одном кадре не совпало со сделанным на другом. Находка обязана назвать "
            "оба экрана: `step` — приложенный кадр, `related_step` — второй кадр "
            "пары, в том числе из перечня без изображения. Кадра из перечня без "
            "изображения для цитаты не существует: дословно переписывай надпись "
            "приложенного кадра."
        )
        return "".join(parts)

    def unit_note(self) -> str:
        """Перечень мер проверки в конце вопроса.

        Меры описаны в задании, но описание длинное, и на потоке кадров модель
        подписывает все экраны первой меркой: перечень шифров прямо перед кадрами —
        то, по чему разбор видит меру, а не догадывается о ней по одному символу.

        :return: блок вопроса, пустая строка, если мер у проверки нет
        """
        if not self.units:
            return ""
        return (
            "\n\n[МЕРЫ ПРОВЕРКИ]\n"
            f"{', '.join(self.units)} — определены в задании выше, "
            f"их {len(self.units)}.\n"
            f"В `units` верни ровно {len(self.units)} коротких строк: по одной на "
            "каждую меру (`unit`, `frames`, `result`, а `why` — только у находки и "
            "у «не применима»). Мера, о которой в ответе нет строки, разбором не "
            "считается — сколько их не названо, ровно столько мер этот путь не "
            "смотрел.\n"
            "Строка `units` говорит только о мере: `deviations` и `note` "
            "остаются полями ответа на уровне объекта и внутрь `units` не "
            "кладутся."
        )

    def variables(
        self,
        material: str,
        evidence_rules: str,
        friction_rules: str,
        details: Optional[CXJourneyDetails] = None,
    ) -> dict:
        """Переменные вопроса: данные пути плюс общие для ветки правила.

        Вопрос отвечает за то, что меняется от прогона к прогону: атрибуцию, заметку
        эксперта, правила доказательства и контракт ответа (`_shared` файлы ветки — в
        шести копиях они разошлись бы между проверками). Предмет проверки здесь менять
        нечем: он целиком лежит в её system-части, и различаются проверки именно ею.

        Ключа проверки в вопросе нет нарочно: он служил только подписью вызова, а
        подписью занимается лаборатория (`analysis-lab-src/lab/textscan.py`), которой
        для этого хватает заголовка той же system-части. Ключ остаётся там, где он и
        решает: в имени узла (`check_journey_friction_<ключ>`) и в ключе состояния.

        :param material: заметка эксперта к материалам пути
        :param evidence_rules: общие правила доказательства
        :param friction_rules: контракт ответа проверки
        :param details: атрибуция пути — по ней проверка решает, к чему путь
        :return: переменные шаблона
        """
        return {
            "evidence_rules": evidence_rules,
            "friction_rules": friction_rules,
            "material": material,
            "journey": details.context() if details else "",
        }

    def messages(
        self,
        material: str,
        frames: Sequence[CXFrame],
        evidence_rules: str,
        friction_rules: str,
        details: Optional[CXJourneyDetails] = None,
        route: str = "",
        total: int = 0,
        values: str = "",
        here: str = "",
    ) -> list[BaseMessage]:
        """Сообщения проверки: вопрос, затем кадры окна изображениями.

        Кадры подписаны глобальными номерами (`frames_content` читает номера самих
        кадров), поэтому окно пути отвечает о тех же экранах, что и весь путь. Перед
        кадрами встаёт пояснение части: без него проверка, увидевшая десять экранов
        из сорока, считает их всем маршрутом и снимает находку «путь длинный» тем,
        что дальше неё ничего не видно. Рядом встаёт перечень величин — без него мера,
        которой нужны два экрана с разных концов пути, снимается молча, а одинаковые
        величины у всех кадров окна перестают быть заметны.

        :param material: заметка эксперта к материалам пути
        :param frames: кадры этого обращения — уходят модели изображениями
        :param evidence_rules: общие правила доказательства
        :param friction_rules: контракт ответа проверки
        :param details: атрибуция пути
        :param route: экраны всего пути одной строкой — след скана
        :param total: число кадров пути; 0 — приложено всё, что есть
        :param values: величины кадров вне этого окна, пустая строка — если следа нет
        :param here: величины приложенных кадров — по ним видно, что кадр не прочитан
        :return: сообщения запроса
        """
        system, user = self.chat_prompt_template.format_messages(
            **self.variables(
                material,
                evidence_rules,
                friction_rules,
                details,
            )
        )
        if not frames:
            return [system, user]

        note = self.slice_note(frames, total or len(frames), route)
        note += self.values_note(values, here)
        note += self.unit_note()
        if note:
            user = HumanMessage(content=f"{user.content}{note}")
        return [system, self.with_frames(user, frames)]

    def chat_request(
        self,
        material: str,
        frames: Sequence[CXFrame],
        evidence_rules: str,
        friction_rules: str,
        details: Optional[CXJourneyDetails] = None,
        route: str = "",
        total: int = 0,
        values: str = "",
        here: str = "",
    ) -> ChatRequest:
        """Запрос проверки: параметры модели, вопрос и кадры окна пути.

        :param material: заметка эксперта к материалам пути
        :param frames: кадры этого обращения
        :param evidence_rules: общие правила доказательства
        :param friction_rules: контракт ответа проверки
        :param details: атрибуция пути
        :param route: экраны всего пути одной строкой
        :param total: число кадров пути
        :param values: величины кадров вне этого окна
        :param here: величины приложенных кадров
        :return: запрос для провайдера контура
        """
        return ChatRequest(
            options=self.chat_options,
            model=self.stage_model,
            messages=self.messages(
                material,
                frames,
                evidence_rules,
                friction_rules,
                details,
                route=route,
                total=total,
                values=values,
                here=here,
            ),
        )


class JourneyFrictionsSettings(CoreSettings):
    """Ветка проблем пути: шесть проверок по кадрам, каждая своим промптом.

    Ветка соседничает с проверкой блоков и запускается рядом с ней: блоки сверяют
    путь со справочником по блокам, здесь же ищут то, что справочник описывает
    целыми блоками, — длину маршрута, лишний ручной ввод, тупики, запоздалое
    раскрытие условий, непонятную навигацию и расхождение обещаний с делом.
    Проверки идут последовательно (каждая спрашивает контур один раз) и кладут
    ответы в собственные ключи состояния, поэтому их прогон сравним с прогоном
    блоков напрямую.

    Состав проверок — дело конфигурации сборки, а не кода: сколько проверок
    описано, столько узлов и встанет в граф.

    Справочника здесь нет нарочно: он живёт у этапа разметки
    (`map_journey_frictions`), который ставит находкам код, блок и зону уже после
    того, как их нашли.
    """

    checks: list[CheckJourneyFrictionSettings] = Field(default_factory=list)
    evidence_rules_file: str = Field(
        default="prompts/journey_frictions/_shared/evidence_rules.txt",
        description=(
            "Общие правила доказательства: их повторяет каждый из шести "
            "промптов, и в шести копиях они разошлись бы между ветками"
        ),
    )
    friction_rules_file: str = Field(
        default="prompts/journey_frictions/_shared/answer_rules.txt",
        description=(
            "Контракт ответа проверки: форму разбирает "
            "`pipeline.utils.friction_findings`, она одна на все шесть проверок"
        ),
    )

    def keys(self) -> list[str]:
        """Ключи проверок в порядке конфигурации — он же порядок узлов в цепи."""
        return [check.check_key for check in self.checks]

    def check(self, check_key: str) -> CheckJourneyFrictionSettings:
        """Настройки проверки по её ключу.

        :param check_key: ключ проверки из конфигурации
        :return: настройки этой проверки
        """
        for check in self.checks:
            if check.check_key == check_key:
                return check
        raise ValueError(
            f"проверки {check_key} нет среди описанных в "
            "graph.journey_frictions.checks"
        )

    @property
    def evidence_rules(self) -> str:
        """Общие правила доказательства — одни для шести проверок."""
        return self._shared_prompt(self.evidence_rules_file)

    @property
    def friction_rules(self) -> str:
        """Контракт ответа проверки — форма, которую разбирает разборщик."""
        return self._shared_prompt(self.friction_rules_file)

    @staticmethod
    def _shared_prompt(relative: str) -> str:
        """Общая часть шести промптов, прочитанная из файла рядом с ними.

        Файл обязанный: пустая переменная промпта молча оставила бы проверку без
        правил доказательства и без контракта ответа, а различие ответов потом
        объяснялось бы разбором, а не сборкой.

        :param relative: путь к файлу от корня репозитория
        :return: содержимое файла
        :raises ValueError: файла нет — иначе прогон собрался бы с пустой общей
            частью и не заметил бы этого
        """
        path = service_root().joinpath(relative)
        if not path.is_file():
            raise ValueError(f"общей части промпта проверок пути нет в {path}")
        return path.read_text(encoding="utf-8")

    def chat_request(
        self,
        check_key: str,
        material: str,
        frames: Sequence[CXFrame],
        details: Optional[CXJourneyDetails] = None,
        route: str = "",
        total: int = 0,
        values: str = "",
        here: str = "",
    ) -> ChatRequest:
        """Запрос одной проверки: её настройки плюс общие для ветки правила.

        :param check_key: ключ проверки, чей узел спрашивает контур
        :param material: заметка эксперта к материалам пути
        :param frames: кадры этого обращения — уходят модели изображениями
        :param details: атрибуция пути
        :param route: экраны всего пути одной строкой — по ним окно пути остаётся
            отрезком маршрута, а не всем маршрутом
        :param total: число кадров пути
        :param values: величины кадров вне этого окна — по ним окно сверяет сумму
            с остатком на экране, который ему не показывали
        :param here: величины приложенных кадров — по ним видно, что кадр окна
            разобран по себе, а не переписан с соседнего
        :return: запрос для провайдера контура
        """
        check = self.check(check_key)
        return check.chat_request(
            material=material,
            frames=frames,
            evidence_rules=self.evidence_rules,
            friction_rules=self.friction_rules,
            details=details,
            route=route,
            total=total,
            values=values,
            here=here,
        )


class AutofillSettings(BaseSummarizeSettings):
    """Фокусный проход «лишний ручной ввод»: один вопрос про автозаполнение.

    Отдельный проход нужен потому, что в общем поэкранном проходе такая находка
    тонет: на десятки кадров и общую тему «найди всё подряд» контур отвечает
    консервативно и не решает, знает ли банк значение поля. Узкая тема на весь
    путь ловит незаполненные поля надёжно — той же причиной у ядра стоит это
    плечо (`eval_agent.py::analyze_autofill`).

    Группы кадров стоят по документированному лимиту контура
    (`max_images_per_request`), а не одним вопросом на весь путь, как это
    было здесь раньше: сверх лимита контур кадры снимает молча, и «форма
    отсутствует» оказывалось выводом о тех экранах, до которых запрос вообще не
    дошёл. Ядро ходит группами по четыре — вопрос просит сверить поле с более
    ранним кадром того же пути, и чем меньше группа, тем чаще повтор ввода остаётся
    за её краем; окно в десять кадров держит и проверку повтора, и весь материал
    обращения. `batch_frames = 0` возвращает один вопрос на весь путь — он годен
    только на путях короче лимита.

    Гейт по типу пути (`journey_types`) по умолчанию открыт: политика прогона
    запрещает ссылаться на то, чем путь является, — основание «путь
    информативный, форм тут нет» снимал с проверки целые блоки. Пустой перечень
    значит «проверяем путь любого типа», и применимость решает сам проход по
    словам промпта (не-банк, гость, форм нет). Значения `conversion` и `service`
    включают гейт ядра: информативный путь перестаёт спрашиваться вовсе.

    Справочника ветка не знает: ни перечня подкритериев, ни блока-адреса у неё
    нет. Прежний `block` записывал весь проход в один раздел — повтор ввода
    ехал в `cxv3:4` даже там, где он разрыв пути или расхождение платформ, — и
    молча терял находки прогона при опечатке в ключе, потому что строка ни с
    чем не сверялась. Находка прохода остаётся внекаталожной, а адрес ей даёт
    сведение (`consolidate_findings`), которое видит весь справочник сразу; сумму
    списания адрес не меняет — она идёт из зоны находки
    (`pipeline.scoring.method.issue_factor`), адрес решает лишь то, какой блок её
    теряет.
    """

    model: str = Field(default="Qwen3.5-397b", alias="LLM_AUTOFILL")
    #: Окно по умолчанию равно пределу обращения (`max_images_per_request`).
    batch_frames: int = Field(
        default=10,
        ge=0,
        alias="AUTOFILL_BATCH",
        description=(
            "Сколько кадров уходит в одном обращении: путь длиннее окна "
            "спрашивается несколькими обращениями; 0 — весь путь одним вопросом "
            "(повтор ввода виден через весь маршрут)"
        ),
    )
    journey_types: list[str] = Field(
        default_factory=list,
        description=(
            "Типы путей, для которых проход запускается; пусто — любого типа"
        ),
    )

    def skip_reason(self, details: Optional[CXJourneyDetails] = None) -> str:
        """Чем объясняется отказ запускать проход; пустая строка — спрашиваем.

        Тип пути читается только при названной атрибуции: без распознавания прогон
        идёт вслепую, и молча снимать проверку по атрибуту, которого нет, нельзя.

        :param details: атрибуция пути — по ней и сверяется тип
        :return: причина пропуска либо пустая строка
        """
        if not self.journey_types or details is None:
            return ""
        kind = details.journey_type_value
        if not kind or kind in self.journey_types:
            return ""
        return f"тип пути {kind} не среди проверяемых ({', '.join(self.journey_types)})"

    def batches(self, frames: Sequence[CXFrame]) -> list[list[CXFrame]]:
        """Кадры группы обращения к контуру: батчи по N кадров всего пути.

        Мерка одна с веткой блоков (`group_slices`): группы набиваются до лимита
        изображений, а хвост короче `min_frames_per_request` отдельным вопросом не
        идёт — на одном экране проход отвечает «форм для разбора ручного ввода нет»,
        и этот ответ оплачен кадром. Хвост дотягивается кадрами предыдущей группы,
        а повтор находки с перехлёста снимает `_merge`. Срез «весь путь» упирается в
        тот же лимит: сверх него контур кадры снимает молча, и проход жаловался бы
        на отсутствие формы на экранах, которых в запросе не было.

        :param frames: кадры пути в порядке следования
        :return: непустые группы кадров в порядке пути
        """
        if not frames:
            return []
        size = self.window_frames(self.batch_frames, len(frames))
        return [
            list(frames[group])
            for group in self.group_slices(len(frames), size)
        ]

    @staticmethod
    def batch_note(
        frames: Sequence[CXFrame],
        total: int,
        number: int,
        count: int,
    ) -> str:
        """Метка обращения прохода: который батч из скольких и какие кадры в нём.

        Окно прохода — десять кадров, путь длиннее: без метки вопрос про десять
        экранов из сорока неотличим от вопроса про весь путь, и «формы нет»
        оказывалось выводом об экранах, до которых запрос не дошёл. Номер стоит в
        самом вопросе, а не только в журнале: по нему лаборатория подписывает
        вызов («автозаполнение батч 2 из 4»), иначе два обращения одного прохода
        различались бы порядком файлов.

        Одно обращение не подписывается: когда весь путь умещается в него, номер —
        шум в вопросе.

        :param frames: кадры этого обращения
        :param total: число кадров пути — по нему видно, что показано не всё
        :param number: который по счёту батч, от одного
        :param count: сколько батчей легло на проход
        :return: строка метки, пустая при одном обращении
        """
        taken = [frame.step for frame in frames]
        if count <= 1 or not taken:
            return ""
        gaps = taken[-1] - taken[0] + 1 > len(taken)
        if len(taken) == 1:
            edges = f"кадр {taken[0]}"
        elif gaps:
            edges = "кадры " + ", ".join(str(step) for step in taken)
        else:
            edges = f"кадры {taken[0]}–{taken[-1]}"
        return (
            "\n\n[БАТЧ]\n"
            f"Батч {number} из {count}: {edges} из {total} кадров пути. "
            "Остальная часть пути проверяется другими обращениями этого прохода, "
            "и их ответов здесь нет: экран с формой мог остаться за батчем, и "
            "вывод о всём маршруте по этому обращению не делается."
        )

    def variables(
        self,
        material: str,
        details: Optional[CXJourneyDetails] = None,
    ) -> dict:
        """Переменные вопроса: что это за путь и заметка эксперта.

        Атрибуция отдаётся тем же `context()`, что читает проверка блоков: мерка
        «гость или авторизованный» берётся оттуда же, откуда и продукт сценария,
        иначе проход судил бы о профиле клиента по другой мерке, чем блоки.

        :param material: заметка эксперта к материалам пути
        :param details: атрибуция пути
        :return: переменные шаблона
        """
        return {
            "journey": details.context() if details else "",
            "material": material.strip()
            or "Заметки нет: путь описан только кадрами.",
        }

    def messages(
        self,
        material: str,
        frames: Sequence[CXFrame],
        details: Optional[CXJourneyDetails] = None,
        total: int = 0,
        number: int = 1,
        count: int = 1,
    ) -> list[BaseMessage]:
        """Сообщения прохода: вопрос, затем кадры изображениями.

        Кадры подписываются своими номерами (`frames_content`): находка обязана
        назвать тот же экран, что лежит в `screenshot_keys` прогона. Перед кадрами
        встаёт номер обращения — без него два батча одного прохода давали бы два
        одинаковых вопроса, по которым непонятно, какой частью пути они отвечены.

        :param material: заметка эксперта к материалам пути
        :param frames: кадры этой группы обращения к контуру
        :param details: атрибуция пути
        :param total: число кадров пути — nonzero, когда группа есть окно, а не
            весь путь
        :param number: который по счёту батч
        :param count: сколько батчей легло на проход
        :return: сообщения запроса
        """
        system, user = self.chat_prompt_template.format_messages(
            **self.variables(material, details)
        )
        if not frames:
            return [system, user]

        note = self.batch_note(frames, total, number, count)
        if note:
            user = HumanMessage(content=f"{user.content}{note}")
        return [system, self.with_frames(user, frames, total)]

    def chat_request(
        self,
        material: str,
        frames: Sequence[CXFrame],
        details: Optional[CXJourneyDetails] = None,
        total: int = 0,
        number: int = 1,
        count: int = 1,
    ) -> ChatRequest:
        """Запрос прохода: параметры модели, вопрос и кадры группы.

        :param material: заметка эксперта к материалам пути
        :param frames: кадры этой группы обращения к контуру
        :param details: атрибуция пути
        :param total: число кадров пути — nonzero, когда группа есть окно, а не
            весь путь
        :param number: который по счёту батч
        :param count: сколько батчей легло на проход
        :return: запрос для провайдера контура
        """
        return ChatRequest(
            options=self.chat_options,
            model=self.stage_model,
            messages=self.messages(
                material, frames, details, total, number, count
            ),
        )


class CriteriaReferenceSettings(BaseSummarizeSettings):
    """Настройки этапа, которому нужен справочник блоков.

    Справочник читается с диска один раз на настройки: этап ходит к контуру
    несколькими обращениями, и перечитывать файл ради каждого значило бы мерить
    путь справочником, изменившимся на середине прогона, — находки одного прохода
    оказались бы размечены разными редакциями каталога.

    Пустой `blocks` значит «весь справочник»; перечень вне справочника считается
    сломанной сборкой конфигов, а не правом этапа ничего не найти.
    """

    criteria_model: str = Field(
        default="configs/service_configs/criteria_model_v3_2.json",
        description="Файл критериальной модели прогона",
    )
    blocks: list[str] = Field(
        default_factory=list,
        description=(
            "Блоки справочника, которых касается этап; пусто — весь справочник"
        ),
    )

    _criteria_model: Optional[CXCriteriaModel] = PrivateAttr(default=None)

    @property
    def criteria_model_path(self) -> str:
        """Подпись справочника этапа — для сообщений об отказе.

        Сам справочник приходит из базы (`bind_criteria_model`): файла нет, и
        настройка только называет, какая редакция ожидалась.
        """
        return self.criteria_model

    def bind_criteria_model(self, model: CXCriteriaModel) -> None:
        """Подкладывает справочник из базы на место собранного из настроек."""
        self._criteria_model = model

    def load_criteria_model(self) -> CXCriteriaModel:
        """Справочник прогона из базы.

        :raises ValueError: справочник не подложен — этап собран в обход
            приложения, которое читает его из базы
        """
        if self._criteria_model is None:
            raise ValueError(
                "справочник не загружен из базы: этап собирается приложением, "
                "а не настройками"
            )
        return self._criteria_model

    def scoped_blocks(self) -> list[CXBlock]:
        """Блоки справочника в его порядке, ограниченные настройкой `blocks`.

        Настройка перечисляет ключи, а порядок и сами имена берёт справочник:
        разделов вне него этапу знать не положено. Пустая настройка — весь
        справочник.

        :return: блоки справочника
        :raises ValueError: если настроенного ключа в справочнике нет — сборка
            конфигов сломана, и молча отдавать пустой перечень значило бы
            оставить этап без ориентира
        """
        model = self.load_criteria_model()
        if not self.blocks:
            return list(model.blocks)
        blocks: list[CXBlock] = []
        for key in self.blocks:
            try:
                blocks.append(model.block(key))
            except KeyError as ex:
                raise ValueError(
                    f"блока {key} из настроечного перечня нет в "
                    f"{self.criteria_model_path}"
                ) from ex
        return blocks

    def reference_listing(self) -> str:
        """Справочник целиком: блоки, критерии, коды дефектов с их зонами.

        Строка кода несёт и зону — по ней разметка отличает блокер от минорного
        замечания, когда под одну находку подходят два кода разных тяжестей.

        :return: перечень справочника
        """
        lines: list[str] = []
        for block in self.scoped_blocks():
            lines.append(f"### {block.key} — {block.name}")
            for criteria in block.criteria:
                lines.append(f"#### {criteria.key} — {criteria.text}")
                lines += [
                    (
                        f"- {defect.code} ({defect.zone}) — {defect.defect}. "
                        f"Проявляется так: {defect.symptom}"
                    )
                    for defect in criteria.defects
                ]
            lines.append("")
        return "\n".join(lines).strip()


class ScreenSettings(CriteriaReferenceSettings):
    """Поэкранный проход: окно кадров, ответ про один разбираемый экран.

    Отдельный проход нужен потому, что он судит каждый экран отдельно, а не путь
    целиком: блоки справочника сверяют путь с каталогом, проверки проблем пути
    ищут то, что живёт между экранами, — и те и другие отвечают на один вопрос о
    всём маршруте. Здесь вопрос другой: что это за экран и что на нём мешает
    клиенту; ответ даёт ещё и описания кадров, которых в остальных ветках нет.

    Путь проход проходит окном: разбираемый кадр и по `context_frames` соседей с
    каждой стороны, а отвечает модель строго про центральный. Всю раскадровку
    одним вопросом спрашивать нельзя: сверх документированного лимита контур
    снимает кадры молча, и вывод «значения нет вообще» оказывался выводом по
    первым десяти кадрам сорокаэкранного пути. Окно же из одного кадра судит
    экран в отрыве от пути — соседи держатся ровно затем, чтобы «чего-то не
    хватает» не рождалось там, где это показано рядом. У каждого кадра ровно одно
    обращение в роли разбираемого, поэтому находка экрана приходит один раз и с
    одной тяжестью.

    Обращения идут последовательно: одновременные вопросы складывались в очередь
    контура (429), а неотвеченная часть пути при параллельном ходе неотличима от
    пути без находок.

    Справочник в вопросе — перечнем критериев, без кодов дефектов и без их зон: он
    здесь затем, чтобы модель знала, что в банковском пути вообще считают проблемой
    («только видимое и только мешающее» задаёт рамку, критерии — круг). Коды и их
    зоны приходят на следующем этапе (`SummarizeScreenSettings`), когда находки уже
    есть: показать их на шаге поиска значило бы подгонять увиденное под справочник
    вместо того чтобы смотреть на кадр.

    Пустой `blocks` значит «весь справочник»: своего предмета, которым
    ограничивались бы находки, у прохода нет — его граница задана самим вопросом.
    """

    model: str = Field(default="Qwen3.5-397b", alias="LLM_SCREEN")
    context_frames: int = Field(
        default=2,
        ge=0,
        alias="SCREEN_CONTEXT_FRAMES",
        description=(
            "Сколько соседей едет в окне с каждой стороны разбираемого кадра: "
            "окно целиком обязано умещаться в лимит изображений обращения "
            "(`max_images_per_request`). Без соседей экран судится в отрыве от "
            "пути и рождает находку «не показали X» там, где X показан на "
            "соседнем кадре; больше двух — окно платит токенами за кадры, по "
            "которым ответа всё равно не будет"
        ),
    )

    @model_validator(mode="after")
    def _window_fits_request(self) -> "ScreenSettings":
        """Окно не бывает больше обращения: лишние кадры контур снял бы молча.

        Настройка стоит в конфиге, а снять кадры сверх лимита транспорт умеет без
        единого слова в ответе — прогон тогда молча терял бы окна, и «разобрали
        все экраны» читалось бы правдой о части пути.

        Raises:
            ValueError: окно из разбираемого кадра и соседей не влезает в одно
                обращение к контуру.
        """
        size = 1 + 2 * self.context_frames
        if size > self.max_images_per_request:
            raise ValueError(
                f"окно поэкранного прохода из {size} кадров не влезает в "
                f"обращение к контуру ({self.max_images_per_request} "
                "изображения) — уменьшите screen.context_frames"
            )
        return self

    def criteria_examples(self) -> str:
        """[ПРИМЕРЫ ОТКЛОНЕНИЙ] построчно: блок и его критерии, без кодов и зон.

        Названия читает справочник, а не память модели: иначе проход выдумывал бы
        разделы, которых в прогоне нет. Ни кодов дефектов, ни их зон здесь нет
        нарочно — коды решают штраф, и модель, видимая их на шаге поиска, начинала
        бы подгонять увиденное под справочник. Суть блока (`essence`) тоже не
        уходит: она описывает измерение целиком, и переписать её моделью дешевле,
        чем посмотреть кадр (`CheckCxCriteriaSettings.criteria_measure`).

        :return: перечень блоков с их критериями
        """
        lines: list[str] = []
        for block in self.scoped_blocks():
            lines.append(f"- {block.key} — {block.name}")
            lines += [
                f"    - {criteria.key} — {criteria.text}"
                for criteria in block.criteria
            ]
        return "\n".join(lines)

    def window(self, frames: Sequence[CXFrame], index: int) -> list[CXFrame]:
        """Окно разбираемого кадра `index` (с нуля): он сам и соседи с двух сторон.

        Окно берётся по тому перечню кадров, который прогону вообще остался: кадр,
        снятый фильтром (`CXFrameReadout.need_filter`), в него не входит, и тогда
        соседом становится кадр через пропуск, а не снятый экран. На краю пути окно
        укорачивается, а не дополняется дублем: повторное изображение стоит
        токенов и ничего не добавляет к контексту.

        :param frames: кадры, оставшиеся после фильтра, в порядке пути
        :param index: положение разбираемого кадра в этом перечне, с нуля
        :return: кадры окна в порядке пути, разбираемый — в середине
        :raises ValueError: если кадров нет вовсе либо индекса в них нет
        """
        if not frames:
            raise ValueError("у пути нет кадров — разбирать нечего")
        if not 0 <= index < len(frames):
            raise IndexError(
                f"кадра {index} нет в пути из {len(frames)} кадров"
            )
        edge = self.context_frames
        return list(frames[max(index - edge, 0) : index + edge + 1])

    def window_content(
        self, window: Sequence[CXFrame], target_step: int, total: int = 0
    ) -> list[dict]:
        """Кадры окна частями содержимого: разбираемый отделён от соседей.

        Подпись стоит при каждом изображении — как в окне скана
        (`ScanJourneyFrameSettings.window_content`): без неё номера экранов
        расходятся с путью, а без разделения ролей модель описывает всё окно сразу
        и клеит находку к чужому кадру. Роли названы действиями («отвечай по
        нему» / «с него ничего не возвращай»): нейтральное «сосед» скану не
        помогало, и проход обязана останавливать та же формулировка. Последней
        идёт строка после всех изображений — напоминание, к кому из них вопрос.

        :param window: кадры окна в порядке пути
        :param target_step: номер разбираемого кадра
        :param total: число кадров пути — nonzero, когда приложено окно, а не весь
            путь
        :return: части содержимого сообщения с подписанными кадрами
        """
        steps = ", ".join(str(frame.step) for frame in window)
        of_path = f" из {total} кадров пути" if total > len(window) else ""
        parts: list[dict] = [
            {
                "type": "text",
                "text": (
                    "\n\n## Кадры окна\n\n"
                    f"Приложено кадров: {len(window)}{of_path} "
                    f"(номера: {steps}). Разбираемый кадр подписан "
                    f"`[кадр {target_step} — разбираемый]`: отвечай по нему, "
                    "соседи приложены только для контекста."
                ),
            }
        ]
        for frame in window:
            if frame.step == target_step:
                role = "разбираемый — описание и находки только по нему"
            else:
                role = "сосед — только контекст, по нему не отвечать"
            parts.append(
                {"type": "text", "text": f"\n[кадр {frame.step} — {role}]"}
            )
            parts.append(image_part(frame))
        parts.append(
            {
                "type": "text",
                "text": (
                    "\n\nИзображения окна кончились. `screens` обязан содержать "
                    f"ровно один элемент — по кадру {target_step}. Соседи нужны "
                    "были ровно затем, чтобы понять, что было до и после; их "
                    "описаний и находок в ответе быть не должно."
                ),
            }
        )
        return parts

    @staticmethod
    def batch_note(
        window: Sequence[CXFrame],
        target: CXFrame,
        total: int,
        number: int,
        count: int,
    ) -> str:
        """Метка обращения: какой кадр разбираем и который это раз.

        Окно меньше пути, и без метки вопрос про пять экранов из сорока
        неотличим от вопроса про весь путь: «этого нет на всём маршруте»
        оказывалось выводом об экранах, до которых обращение не дошло. Номер стоит
        в самом вопросе, а не только в журнале: по номеру разбираемого кадра
        лаборатория и подписывает вызов («поэкранный проход кадр 7»), иначе
        обращения одного прохода различались бы порядком файлов.

        Одно обращение не подписывается: когда путь умещается в него, номер — шум
        в вопросе.

        :param window: кадры этого обращения
        :param target: разбираемый кадр окна
        :param total: число кадров пути — по нему видно, что показано не всё
        :param number: который по счёту батч, от одного
        :param count: сколько батчей легло на проход
        :return: строка метки, пустая при одном обращении
        """
        if count <= 1:
            return ""
        neighbors = [
            frame.step for frame in window if frame.step != target.step
        ]
        edges = (
            "ни одного"
            if not neighbors
            else "кадры " + ", ".join(str(step) for step in neighbors)
        )
        return (
            "\n\n[БАТЧ]\n"
            f"Батч {number} из {count}: разбираем кадр {target.step} из {total} "
            f"кадров пути, соседями приложено {edges}. Остальная часть пути "
            "проверяется другими обращениями этого прохода, и их ответов здесь "
            "нет: вывод о всём маршруте по этому обращению не делается, как и "
            "вывод «этого нет на всём пути»."
        )

    def variables(
        self,
        material: str,
        details: Optional[CXJourneyDetails] = None,
    ) -> dict:
        """Переменные вопроса: что тут считают проблемой, что это за путь, заметка.

        Атрибуция отдаётся тем же `context()`, что читает ветка блоков: мерка
        оценки берётся оттуда же, откуда продукт и сценарий, иначе проход судил бы
        путь по другой атрибуции, чем остальные ветки.

        :param material: текстовые материалы пути
        :param details: атрибуция пути
        :return: переменные шаблона
        """
        return {
            "criteria_examples": self.criteria_examples(),
            "journey": details.context() if details else "",
            "material": material.strip()
            or "Заметки нет: путь описан только кадрами.",
        }

    def messages(
        self,
        material: str,
        window: Sequence[CXFrame],
        target: CXFrame,
        details: Optional[CXJourneyDetails] = None,
        total: int = 0,
        number: int = 1,
        count: int = 1,
    ) -> list[BaseMessage]:
        """Сообщения прохода: вопрос, затем кадры окна изображениями.

        Кадры подписываются своими номерами и ролями (`window_content`): находка
        обязана назвать тот же экран, что лежит в `screenshot_keys` прогона, и не
        имеет права приклеиться к соседу. Перед кадрами встаёт номер обращения —
        без него окна одного прохода давали бы одинаковые вопросы, по которым
        непонятно, какой частью пути они отвечены.

        :param material: текстовые материалы пути
        :param window: кадры этого обращения
        :param target: разбираемый кадр окна
        :param details: атрибуция пути
        :param total: число кадров пути — по нему видно, что показано не всё
        :param number: который по счёту батч
        :param count: сколько батчей легло на проход
        :return: сообщения запроса
        """
        system, user = self.chat_prompt_template.format_messages(
            **self.variables(material, details)
        )
        if not window:
            return [system, user]

        note = self.batch_note(window, target, total, number, count)
        content: list[dict] = [
            {"type": "text", "text": f"{user.content}{note}"}
        ]
        content += self.window_content(window, target.step, total)
        return [system, HumanMessage(content=content)]

    def chat_request(
        self,
        material: str,
        window: Sequence[CXFrame],
        target: CXFrame,
        details: Optional[CXJourneyDetails] = None,
        total: int = 0,
        number: int = 1,
        count: int = 1,
    ) -> ChatRequest:
        """Запрос прохода: параметры модели, вопрос и кадры окна.

        :param material: текстовые материалы пути
        :param window: кадры этого обращения
        :param target: разбираемый кадр окна
        :param details: атрибуция пути
        :param total: число кадров пути — по нему видно, что показано не всё
        :param number: который по счёту батч
        :param count: сколько батчей легло на проход
        :return: запрос для провайдера контура
        """
        return ChatRequest(
            options=self.chat_options,
            model=self.stage_model,
            messages=self.messages(
                material, window, target, details, total, number, count
            ),
        )


class SummarizeScreenSettings(CriteriaReferenceSettings):
    """Разметка находок поэкранного прохода кодами справочника.

    Этап стоит после прохода и только там: проход ищет то, что мешает клиенту на
    экране, и кодов при этом не видит — показать их на шаге поиска значило бы
    подгонять увиденное под каталог. Здесь находки уже есть (текст, цитата, экран,
    зона), и у них появляется адрес: код дефекта, его блок и его зона. Зона
    берётся из справочника по тому же правилу, что у ветки блоков
    (`pipeline.utils.block_findings._deviations`) — код решает штраф, и одно и то же
    нарушение не должно штрафоваться по-разному из блока и с экрана.

    Чего здесь нет, так это изображений: разметка сверяет описания находок с
    каталогом, а не разглядывает путь заново, и второе обращение со всеми кадрами
    стоило бы дороже самого прохода. Координаты этап не получает и не возвращает:
    пин ставит тот, кто видел экран, а смоделированная заново координата уехала бы
    по кадру вместе с исправленным текстом.

    Незаматченная находка — не потеря: она остаётся в отчёте без кода и со своей
    зоной (тот же смысл, что служебное `uncatalogued` у ветки блоков), потому что
    «вижу проблему, кода под неё нет» — регулярный итог сверки с каталогом.

    Справочник уходит весь: 877 кодов с описаниями и симптомами — это десятки тысяч
    токенов текста на один вызов, и это цена полноты сверки. Изображений при этом
    нет, а обращение одно на все находки: резать справочник молча значило бы
    терять коды, которых модель просто не видела.
    """

    model: str = Field(default="Qwen3.5-397b", alias="LLM_SUMMARIZE_SCREEN")

    def findings_listing(
        self,
        deviations: Sequence[CXPinnedDeviation],
        notes: Sequence[CXScreenNote] = (),
    ) -> str:
        """Находки прохода перечнем с номерами: `[находка N] экран … зона …`.

        Номер — единственный способ увязать ответ разметки с находкой: тексты
        похожи («поле просят ввести заново» на трёх экранах), а экран и цитата
        различают их надёжно, поэтому они в строке стоят. Описания кадров идут
        той же строкой: без них «нет подсказки» неотличимо от подсказки, которая
        есть на соседнем экране. Координат здесь нет намеренно — см. докстринг
        класса.

        :param deviations: находки прохода в порядке пути
        :param notes: описания кадров — им находятся находки по номеру экрана
        :return: перечень находок либо пустая строка
        """
        if not deviations:
            return ""
        summaries = {note.step: note.summary.strip() for note in notes}
        lines: list[str] = []
        for number, deviation in enumerate(deviations, start=1):
            where = (
                f"экран {deviation.step}" if deviation.step else "уровень пути"
            )
            summary = summaries.get(deviation.step, "")
            screen = f"; на экране: {summary}" if summary else ""
            quote = (
                f"; улика {'; '.join(deviation.evidence)}"
                if deviation.evidence
                else ""
            )
            lines.append(
                f"[находка {number}] {where}, зона {deviation.zone}: "
                f"{deviation.text}{quote}{screen}"
            )
        return "\n".join(lines)

    def variables(
        self,
        deviations: Sequence[CXPinnedDeviation],
        notes: Sequence[CXScreenNote] = (),
        details: Optional[CXJourneyDetails] = None,
    ) -> dict:
        """Переменные вопроса: справочник, находки, что это за путь.

        :param deviations: находки прохода
        :param notes: описания кадров прохода
        :param details: атрибуция пути
        :return: переменные шаблона
        """
        return {
            "criteria_reference": self.reference_listing(),
            "findings": self.findings_listing(deviations, notes),
            "journey": details.context() if details else "",
        }

    def messages(
        self,
        deviations: Sequence[CXPinnedDeviation],
        notes: Sequence[CXScreenNote] = (),
        details: Optional[CXJourneyDetails] = None,
    ) -> list[BaseMessage]:
        """Сообщения разметки: вопрос и справочник, изображений нет.

        :param deviations: находки прохода
        :param notes: описания кадров прохода
        :param details: атрибуция пути
        :return: сообщения запроса
        """
        return list(
            self.chat_prompt_template.format_messages(
                **self.variables(deviations, notes, details)
            )
        )

    def chat_request(
        self,
        deviations: Sequence[CXPinnedDeviation],
        notes: Sequence[CXScreenNote] = (),
        details: Optional[CXJourneyDetails] = None,
    ) -> ChatRequest:
        """Запрос разметки: параметры модели и текстовый вопрос.

        :param deviations: находки прохода
        :param notes: описания кадров прохода
        :param details: атрибуция пути
        :return: запрос для провайдера контура
        """
        return ChatRequest(
            options=self.chat_options,
            model=self.stage_model,
            messages=self.messages(deviations, notes, details),
        )


class MapJourneyFrictionsSettings(CriteriaReferenceSettings):
    """Разметка находок проверок пути кодами справочника.

    Этап стоит после шести проверок и только там: проверки ищут то, что мешает на
    всём пути, и кодов при этом не видят — показать им справочник значило бы
    подгонять увиденное под каталог (см. `CheckJourneyFrictionSettings`). Здесь
    находки уже есть (текст, цитата, экран, зона), и у них появляется адрес: код
    дефекта, его блок и его зона. Зона берётся из справочника по тому же правилу,
    что у поэкранной разметки и у ветки блоков, — код решает штраф, и одно и то же
    нарушение не должно штрафоваться по-разному из разных веток.

    Порядок скопирован с пары `screen` → `summarize_screen` не ради сходства
    промптов, а потому что у проверок не могло быть своего перечня блоков: он
    привязывал бы каждую проверку к ревизии справочника и заставлял бы снимать
    находку, когда её адрес оказался в «чужом» разделе. Теперь адрес ищется по всему
    справочнику сразу, а шесть проверок остаются шестью способами смотреть на путь.

    Находки идут группой на проверку: по заголовку группы разметка видит, каким
    взглядом найдено отклонение, и это различает две похожие строки разных проверок.
    Номера при этом сквозные — ответ привязан к находке, а не к группе.

    Чего здесь нет, так это изображений: разметка сверяет описания находок с
    каталогом, а не разглядывает путь заново, и второе обращение со всеми кадрами
    стоило бы дороже самого поиска.

    Незаматченная находка — не потеря: она остаётся в отчёте без кода и со своей
    зоной, потому что «вижу проблему, кода под неё нет» — регулярный итог сверки с
    каталогом (`pipeline.utils.deviation_mapping`).
    """

    model: str = Field(
        default="Qwen3.5-397b", alias="LLM_MAP_JOURNEY_FRICTIONS"
    )

    def findings_listing(
        self, groups: Sequence[tuple[str, Sequence[CXDeviation]]]
    ) -> str:
        """Находки проверок перечнем с номерами, группой на проверку.

        Номер сквозной — одним ответом размечаются находки шести проверок, и резать
        их шестью обращениями значило бы шесть раз заплатить за справочник. Группа
        подписана тем, что проверяла: «шагов больше, чем шагов оплаты» без подписки
        неотличимо от того же наблюдения про навигацию.

        :param groups: группы `(подпись проверки, её находки)` в порядке ветки
        :return: перечень находок либо пустая строка
        """
        lines: list[str] = []
        number = 0
        for title, deviations in groups:
            if not deviations:
                continue
            lines.append(f"### проверка {title}")
            for deviation in deviations:
                number += 1
                where = (
                    f"экран {deviation.step}"
                    if deviation.step
                    else "уровень пути"
                )
                pair = (
                    f" (+{deviation.related_step})"
                    if deviation.related_step
                    else ""
                )
                quote = (
                    f"; улика {'; '.join(deviation.evidence)}"
                    if deviation.evidence
                    else ""
                )
                lines.append(
                    f"[находка {number}] {where}{pair}, зона {deviation.zone}: "
                    f"{deviation.text}{quote}"
                )
            lines.append("")
        return "\n".join(lines).strip()

    def variables(
        self,
        groups: Sequence[tuple[str, Sequence[CXDeviation]]],
        details: Optional[CXJourneyDetails] = None,
    ) -> dict:
        """Переменные вопроса: справочник, находки, что это за путь.

        :param groups: группы находок проверок
        :param details: атрибуция пути
        :return: переменные шаблона
        """
        return {
            "criteria_reference": self.reference_listing(),
            "findings": self.findings_listing(groups),
            "journey": details.context() if details else "",
        }

    def messages(
        self,
        groups: Sequence[tuple[str, Sequence[CXDeviation]]],
        details: Optional[CXJourneyDetails] = None,
    ) -> list[BaseMessage]:
        """Сообщения разметки: вопрос и справочник, изображений нет.

        :param groups: группы находок проверок
        :param details: атрибуция пути
        :return: сообщения запроса
        """
        return list(
            self.chat_prompt_template.format_messages(
                **self.variables(groups, details)
            )
        )

    def chat_request(
        self,
        groups: Sequence[tuple[str, Sequence[CXDeviation]]],
        details: Optional[CXJourneyDetails] = None,
    ) -> ChatRequest:
        """Запрос разметки: параметры модели и текстовый вопрос.

        :param groups: группы находок проверок
        :param details: атрибуция пути
        :return: запрос для провайдера контура
        """
        return ChatRequest(
            options=self.chat_options,
            model=self.stage_model,
            messages=self.messages(groups, details),
        )


class ConsolidateFindingsSettings(BaseSummarizeSettings):
    """Настройки узла сведения находок в итоговый ответ прогона."""

    model: str = Field(
        default="Qwen3.5-397b", alias="LLM_CONSOLIDATE_FINDINGS"
    )

    def variables(
        self,
        material: str,
        raw_findings: str,
        checks: str,
        block_keys: str,
        finding_count: int = 0,
        details: Optional[CXJourneyDetails] = None,
    ) -> dict:
        return {
            "material": material,
            "journey": details.context() if details else "",
            "raw_findings": raw_findings,
            "checks": checks,
            "block_keys": block_keys,
            "block_count": self._lines(block_keys),
            "finding_count": finding_count,
        }

    def messages(
        self,
        material: str,
        raw_findings: str,
        checks: str,
        block_keys: str,
        finding_count: int = 0,
        details: Optional[CXJourneyDetails] = None,
    ) -> list[BaseMessage]:
        """Сообщения сведения: модель сводит текстовые описания, кадров здесь нет.

        Формат ответа здесь не описывают: его задаёт схема структурированного вывода,
        а инструкцию формата дописывает в сообщение контур.

        :param material: текстовые материалы пути
        :param raw_findings: сырые находки блоков с зонами из справочника
        :param checks: подкритерии по блокам — источник допустимых `check_key`
        :param block_keys: ключи блоков, между которыми распределяются находки
        :param finding_count: сколько сырых находок сводится
        :param details: атрибуция пути — сводка обязана называть тот же продукт
        :return: сообщения запроса
        """
        return list(
            self.chat_prompt_template.format_messages(
                **self.variables(
                    material,
                    raw_findings,
                    checks,
                    block_keys,
                    finding_count,
                    details,
                )
            )
        )

    def chat_request(
        self,
        material: str,
        raw_findings: str,
        checks: str,
        block_keys: str,
        finding_count: int = 0,
        details: Optional[CXJourneyDetails] = None,
    ) -> ChatRequest:
        """Запрос сведения: тексты находок, кадров в нём нет.

        :param material: текстовые материалы пути
        :param raw_findings: сырые находки блоков с зонами из справочника
        :param checks: подкритерии по блокам — источник допустимых `check_key`
        :param block_keys: ключи блоков, между которыми распределяются находки
        :param finding_count: сколько сырых находок сводится
        :param details: атрибуция пути — сводка обязана называть тот же продукт
        :return: запрос для провайдера контура
        """
        return ChatRequest(
            options=self.chat_options,
            model=self.stage_model,
            messages=self.messages(
                material,
                raw_findings,
                checks,
                block_keys,
                finding_count,
                details,
            ),
        )

    @staticmethod
    def _lines(listing: str) -> int:
        """Сколько строк в построчном перечне — по нему считают число блоков."""
        return len([line for line in listing.splitlines() if line.strip()])


class DedupFindingsSettings(BaseSummarizeSettings):
    """Настройки этапа сведения находок текущего прогона с прошлыми.

    Этап отдельный, а не продолжение сведения: сведение сводит находки одного
    прогона между собой, а здесь на входе уже сведённый список за сегодня и
    список за прошлые прогоны того же продукта. Модель и калибровка нужны свои —
    сверка по смыслу капризнее отбора, и подменять её настройками сведения
    значило бы менять поведение обоих этапов одной правкой.
    """

    model: str = Field(default="Qwen3.5-397b", alias="LLM_DEDUP_FINDINGS")

    def variables(
        self,
        previous_findings: str,
        current_findings: str,
        previous_count: int = 0,
        current_count: int = 0,
    ) -> dict:
        """Значения шаблона этапа: два перечня и их счётчики.

        :param previous_findings: прошлые находки продукта построчно
        :param current_findings: находки текущего прогона построчно
        :param previous_count: сколько прошлых находок пришло
        :param current_count: сколько находок пришло в текущем прогоне
        :return: значения переменных шаблона
        """
        return {
            "previous_findings": previous_findings,
            "current_findings": current_findings,
            "previous_count": previous_count or self._lines(previous_findings),
            "current_count": current_count or self._lines(current_findings),
        }

    def messages(
        self,
        previous_findings: str,
        current_findings: str,
        previous_count: int = 0,
        current_count: int = 0,
    ) -> list[BaseMessage]:
        """Сообщения сверки: кадров нет, только два перечня находок.

        :param previous_findings: прошлые находки продукта построчно
        :param current_findings: находки текущего прогона построчно
        :param previous_count: сколько прошлых находок пришло
        :param current_count: сколько находок пришло в текущем прогоне
        :return: сообщения запроса
        """
        return list(
            self.chat_prompt_template.format_messages(
                **self.variables(
                    previous_findings,
                    current_findings,
                    previous_count,
                    current_count,
                )
            )
        )

    def chat_request(
        self,
        previous_findings: str,
        current_findings: str,
        previous_count: int = 0,
        current_count: int = 0,
    ) -> ChatRequest:
        """Запрос сверки находок текущего прогона с прошлыми.

        :param previous_findings: прошлые находки продукта построчно
        :param current_findings: находки текущего прогона построчно
        :param previous_count: сколько прошлых находок пришло
        :param current_count: сколько находок пришло в текущем прогоне
        :return: запрос для провайдера контура
        """
        return ChatRequest(
            options=self.chat_options,
            model=self.stage_model,
            messages=self.messages(
                previous_findings,
                current_findings,
                previous_count,
                current_count,
            ),
        )

    @staticmethod
    def _lines(listing: str) -> int:
        """Сколько строк в построчном перечне — счётчик для шаблона."""
        return len([line for line in listing.splitlines() if line.strip()])


class GraphSettings(CoreSettings):
    """Настройки прогона: узлы, проверка блоков и фокусные проходы.

    Блоки справочника настраивает `check_cx_criteria`; рядом с ней — три
    прохода, идущие своей веткой: проблемы пути (`journey_frictions`), лишний
    ручной ввод (`autofill`) и поэкранный разбор (`screen`). У двух из них есть
    второй узел — `summarize_screen` и `map_journey_frictions`: проход ищет
    отклонения без кодов, а эти узлы дают им адрес в справочнике. Настройки у них
    свои, потому что и модель на сверке с каталогом нужна другая, чем на осмотре
    экрана.
    """

    journey_exploration: JourneyExplorationSettings
    summarize_journey: SummarizeJourneySettings
    check_cx_criteria: CheckCxCriteriaSettings
    journey_frictions: JourneyFrictionsSettings
    map_journey_frictions: MapJourneyFrictionsSettings
    autofill: AutofillSettings
    screen: ScreenSettings
    summarize_screen: SummarizeScreenSettings
    consolidate_findings: ConsolidateFindingsSettings
    dedup_findings: DedupFindingsSettings

    def bind_criteria_model(self, model: CXCriteriaModel) -> None:
        """Подкладывает справочник всем этапам прогона, которые его спрашивают.

        Справочник живёт только в базе и читается один раз на процесс; этапы
        держат его в приватном атрибуте. Раздать его обязан один владелец —
        иначе этап, которому справочник не достался, падает на
        `load_criteria_model()` с `ValueError`, и прогон объявляется
        деградированным: разметка находок пути без каталога не имеет адреса.
        """
        self.check_cx_criteria.bind_criteria_model(model)
        for stage in (
            self.map_journey_frictions,
            self.screen,
            self.summarize_screen,
        ):
            stage.bind_criteria_model(model)

    @model_validator(mode="after")
    def _stage_model_variables(self) -> "GraphSettings":
        """Каждый этап обязан называть свою переменную модели.

        Проверка смотрит на описание полей, а не на окружение: значение этапа вычитает
        источник конфига (`src/utils/external_sources.py`) по `alias`, и узел без
        alias переключается только правкой конфига — замер «весь прогон на другой
        модели» молча оставил бы его прежним. Две разные ветки за одной переменной —
        та же ловушка с обратной стороны: возврат одной ветки прежней модели утянул бы
        за собой соседнюю. Проверка дешевле размётанного прогона, где этап отвечает не
        той моделью, что другие.

        Raises:
            ValueError: настройки этапа не подписаны на свою переменную либо
                делят переменную с настройкой другого этапа.
        """
        stages: list[BaseSummarizeSettings] = [
            self.journey_exploration.scan_frame,
            self.summarize_journey,
            self.check_cx_criteria,
            *self.journey_frictions.checks,
            self.map_journey_frictions,
            self.autofill,
            self.screen,
            self.summarize_screen,
            self.consolidate_findings,
            self.dedup_findings,
        ]
        declared: dict[str, Optional[str]] = {
            type(stage).__name__: type(stage).model_fields["model"].alias
            for stage in stages
        }
        if absent := sorted(name for name, env in declared.items() if not env):
            raise ValueError(
                f"у настроек {', '.join(absent)} поле model без alias с именем "
                "переменной этапа — из .env этот этап не переключается"
            )
        own = {env for env in declared.values() if env}
        shared = sorted(
            env
            for env in own
            if len({name for name, value in declared.items() if value == env})
            > 1
        )
        if shared:
            raise ValueError(
                "переменную " + ", ".join(shared) + " делят настройки разных "
                "этапов — вернуть один этап прежней модели, не трогая другой, "
                "было бы нельзя"
            )
        return self
