import operator
from typing import Annotated, Optional
from uuid import uuid4

from pydantic import Field

from src.pipeline.schemas.core import CoreSchema
from src.pipeline.schemas.nodes import (
    CxAutofillFindings,
    CXBlockFindings,
    CXCheckFindings,
    CXConsolidatedFindings,
    CXCriteriaModel,
    CXFrame,
    CXFrameReadout,
    CXFrictionFindings,
    CXJourneyDetails,
    CXScreenFindings,
)
from src.pipeline.settings.params.nodes import CXNodeStatus


class AgentStateSchema(CoreSchema):
    run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Идентификатор прогона — попадает в итоговый ответ агента",
    )
    screenshot_keys: list[str] = Field(
        default_factory=list,
        description=(
            "Ключи кадров прогона в хранилище — ответ агента возвращает их как есть"
        ),
    )
    material: str = Field(
        default="",
        description=(
            "Заметка эксперта к материалам пути — уходящий модели текст; продукт и "
            "сценарий в ней не приходят: их называет сведение описаний кадров"
        ),
    )
    frames: list[CXFrame] = Field(
        default_factory=list,
        description="Кадры клиентского пути — модель видит их изображениями",
    )
    frame_readouts: Annotated[list[CXFrameReadout], operator.add] = Field(
        default_factory=list,
        description=(
            "Описания кадров, добытые сканированием, в порядке пути: узел "
            "склада сюда по одному за проход цикла, поэтому их длина — это и "
            "номер разбираемого кадра. Кадр без описания остаётся записанным "
            "со статусом ошибки: молчание контура не должно выглядеть так, "
            "будто кадр не смотрели"
        ),
    )
    journey_details: Optional[CXJourneyDetails] = Field(
        default=None,
        description=(
            "Что за путь распознан по кадрам — вердикт узла распознавания; "
            "пусто, если распознавание в этом прогоне не запускалось. Чем "
            "считать ответ контура — вердиктом, пустотой или сбоем узла, — "
            "решает `pipeline.utils.journey.journey_details`"
        ),
    )
    cx_findings: Annotated[dict[str, CXBlockFindings], operator.or_] = Field(
        default_factory=dict,
        description=(
            "Находки по блокам критериальной модели: ключ блока — находки блока. "
            "Их пишут сами узлы блоков: блок проверяется несколькими обращениями "
            "по батчам кадров, и вердикты батчей сводит `merge_batches`"
        ),
    )
    criteria_model_status: Annotated[dict[str, CXNodeStatus], operator.or_] = (
        Field(
            default_factory=dict,
            description=(
                "Чем закончился узел блока там, где ответа нет: без этой отметки "
                "блок с сорванным обращением неотличим от блока без находок. "
                "ВНИМАНИЕ — поле никогда не заполняется: его писатель "
                "`block_status_update` не вызывается ни из одного узла, а статус "
                "блока фактически лежит в `cx_findings[<блок>].status`. Читатели "
                "получают пустой словарь. Разбор — "
                "`docs/project-spec/09-maintenance/improvements.md`, I-004"
            ),
        )
    )
    journey_frictions: Annotated[
        dict[str, CXFrictionFindings], operator.or_
    ] = Field(
        default_factory=dict,
        description=(
            "Проблемы пути по шести проверкам: ключ проверки — её находки. "
            "Ветка идёт рядом с проверкой блоков и пишет только свои ключи, "
            "поэтому состояние не спорит сам с собой, когда обе ветки отвечают "
            "в одном супершаге"
        ),
    )
    autofill_findings: Optional[CxAutofillFindings] = Field(
        default=None,
        description=(
            "Находки фокусного прохода автозаполнения: где банк просил набрать "
            "то, что о клиенте уже знает. Пусто, если проход не запускался; "
            "проход без находок отличается от пропуска своим статусом и `note` "
            "(`pipeline.utils.autofill_findings.autofill_update`)"
        ),
    )
    screen_findings: Optional[CXScreenFindings] = Field(
        default=None,
        description=(
            "Поэкранный разбор: описание каждого кадра и его находки с "
            "координатами пина. Пусто, если проход не запускался. Отклонения "
            "этого прохода идут в сведение наравне с находками блоков и "
            "проверок пути, а описания кадров — только в секцию `screens` "
            "ответа (`pipeline.utils.screen_findings.screens_out`)"
        ),
    )
    cx_consolidation: Optional[CXConsolidatedFindings] = Field(
        default=None,
        description=(
            "Сведённый ответ модели: единый список находок пути и сводка. Пусто, "
            "если сведение не состоялось — ответ прогона тогда собирают находки блоков"
        ),
    )
    criteria_model: Optional[CXCriteriaModel] = Field(
        default=None,
        description=(
            "Справочник блоков прогона — кладётся в состояние при входе, чтобы "
            "узлы и отчёт читали один и тот же справочник, а не каждый свой"
        ),
    )

    @property
    def block_keys_listing(self) -> str:
        """Блоки прогона построчно — по ним сведение считает число проверок."""
        return "\n".join(
            f"  - {block.key} — {block.name}"
            for block in self._required_criteria().blocks
        )

    @property
    def checks_listing(self) -> str:
        """Подкритерии по блокам — источник допустимых `check_key` для модели."""
        lines: list[str] = []
        for block in self._required_criteria().blocks:
            lines.append(f"### {block.key} — {block.name}")
            for criteria in block.criteria:
                lines.append(f"  - {criteria.key} — {criteria.text}")
            lines.append("")
        return "\n".join(lines).strip()

    @property
    def criteria_model_findings(self) -> list[CXBlockFindings]:
        """Находки блоков справочника — то, что дала проверяющая ветка блоков.

        Узел разбирает ответы батчей и складывает их в итог блока
        (`pipeline.utils.block_findings.merge_batches`), поэтому здесь лежит готовое:
        состояние больше не хранит сырой ответ блока, которого теперь несколько.

        Порядок — порядок справочника: по нему читают отчёт и сведение. Блок, о
        котором не известно ничего (ни находок, ни отметки о сбое), в перечень не
        входит: отсутствие блока прогон называет отдельно, по числу блоков
        справочника (`pipeline.dto.agent.result.AgentResult._unanswered_blocks`).
        """
        blocks = self.criteria_model.blocks if self.criteria_model else []
        found = [
            found
            for block in blocks
            if (found := self._block_findings(block.key)) is not None
        ]
        seen = {block.key for block in blocks}
        found += [
            findings
            for key, findings in self.cx_findings.items()
            if key not in seen
        ]
        return found

    @property
    def journey_findings(self) -> list[CXFrictionFindings]:
        """Находки шести проверок проблем пути в порядке конфига."""
        return list(self.journey_frictions.values())

    @property
    def autofill(self) -> list[CxAutofillFindings]:
        """Находки прохода автозаполнения — пустой перечень, если его не было."""
        if self.autofill_findings is None:
            return []
        return [self.autofill_findings]

    @property
    def screens(self) -> list[CXScreenFindings]:
        """Находки поэкранного прохода — пустой перечень, если его не было."""
        if self.screen_findings is None:
            return []
        return [self.screen_findings]

    @property
    def findings(self) -> list[CXCheckFindings]:
        """Все находки прогона одним перечнем — вход сведения.

        Порядок повторяет порядок фаз прогона: блоки справочника в его составе,
        затем поэкранный проход, затем проверки проблем пути в порядке конфига,
        затем проход автозаполнения. Сведение читает источники как соседние абзацы
        одного отчёта — только так одна проблема, найденная блоком, экраном и
        проверкой, видна дублем, а не тремя разделами.
        """
        return [
            *self.criteria_model_findings,
            *self.screens,
            *self.journey_findings,
            *self.autofill,
        ]

    @property
    def findings_by_source(self) -> dict[str, CXCheckFindings]:
        """Находки ключом по источнику — так их читает отчёт по своему блоку."""
        return {found.source_key: found for found in self.findings}

    @property
    def raw_findings_count(self) -> int:
        """Сколько сырых отклонений досталось сведению — по всем трём веткам."""
        return sum(len(source.deviations) for source in self.findings)

    def _block_findings(self, key: str) -> Optional[CXBlockFindings]:
        """Итог одного блока: находки ветки либо отметка о том, чем узел кончил.

        :param key: ключ блока справочника
        :return: итог блока либо `None`, если блок в прогоне не отвечал
        """
        written = self.cx_findings.get(key)
        if written is not None:
            return written

        status = self.criteria_model_status.get(key)
        if status is not None:
            return CXBlockFindings(source_key=key, status=status)
        return None

    @property
    def scan_evidence(self) -> str:
        """Текстовый след всего пути — тем же перечнем, что видела модель."""
        return self.evidence_of()

    def evidence_of(self) -> str:
        """Текстовый след пути: описание кадра, его фаза и дословный текст экрана.

        След читают там, где кадр не покажешь: сведение описаний
        (`pipeline.agent.nodes.summarize_journey`) называет по нему продукт, сценарий и
        сквозной путь. Здесь поэтому есть и `text_on_screen`: компактный
        `listing()` годится для контекста скана, но по нему нельзя ни сверить
        сумму между экранами, ни процитировать надпись.

        Кадр без описания остаётся строкой: молчание контура обязано выглядеть
        непроверенным, иначе сведение прочитает пустоту как «такого элемента нет».
        Отфильтрованный кадр (`need_filter`) остаётся в следе отметкой без
        описаний и с причиной: это кадр не клиентский путь (фото, скан, чужой
        экран) — не «экрана не было» и не «проверка молчала», а третий случай.
        Неполный ответ скана сюда попадает обычным блоком: кадр с текстом без
        описания (или наоборот) анализа не покидает.

        :return: след кадров в порядке пути
        """
        filter_mark = "кадр отфильтрован — он не участвует в анализе"
        suspect_mark = (
            "Текст экрана под вопросом: заголовок этого кадра в нём не найден, "
            "переписан другой экран. Опирайся на описание, а не на цитаты из текста."
        )
        chunks: list[str] = []
        for readout in self.frame_readouts:
            if not readout.answered:
                reason = readout.degraded or readout.status
                chunks.append(
                    f"### Кадр {readout.step}\n"
                    f"Описания не было ({reason}) — этот экран не проверен."
                )
                continue
            head = f"### Кадр {readout.step} — {readout.screen or 'экран не назван'}"
            if readout.need_filter:
                why = f"\n{readout.filter_note}" if readout.filter_note else ""
                chunks.append(f"{head} ({filter_mark}){why}")
                continue
            meta = [
                part
                for part in (
                    f"фаза: {readout.stage}" if readout.stage else "",
                    (
                        f"изменилось: {readout.changed}"
                        if readout.changed
                        else ""
                    ),
                    (
                        "кадр посторонний пути"
                        if readout.is_journey is False
                        else ""
                    ),
                )
                if part
            ]
            lines = [f"{head} ({'; '.join(meta)})" if meta else head]
            if readout.narrative:
                lines.append(readout.narrative)
            if readout.why_not_journey:
                lines.append(f"Почему посторонний: {readout.why_not_journey}")
            if readout.text_on_screen:
                lines.append(f"Текст экрана: {readout.text_on_screen}")
                if readout.suspect_text:
                    lines.append(suspect_mark)
            else:
                lines.append("Текст экрана не переписан.")
            chunks.append("\n".join(lines))
        return "\n\n".join(chunks)

    @property
    def filtered_steps(self) -> set[int]:
        """Номера кадров, снятых с анализа: там стоит `need_filter`.

        Флаг ставит модель и дописывает схема — по `is_journey = false`. Снимает
        только кадр, который не может быть клиентским путём; неполный ответ скана
        (пустой текст или пустое описание) фильтром не стал и кадр не трогает.
        Кадр остаётся в пути своим номером и своим описанием-отметкой, но модель
        больше его не показывает. Отсюда два перечня: `frames` — путь целиком, а
        `kept_frames` — то, что уходит в вопросы.
        """
        return {
            readout.step
            for readout in self.frame_readouts
            if readout.need_filter
        }

    @property
    def kept_frames(self) -> list[CXFrame]:
        """Кадры, которые прогон показывает модели: весь путь минус отфильтрованные.

        Снятый кадр выпадает отовсюду, где модель смотрит изображения: из окон
        скана, из вопроса блоков справочника, из проверок пути и обоих проходов —
        вместе со своей ссылкой на загруженный файл. Его изображение прогону больше
        не нужно ни в одном обращении.

        Номера оставшихся кадров при этом не пересчитываются: находка обязана
        остаться привязанной к тому экрану, который был на самом деле, поэтому
        верхнюю границу номера считают по `frames`, а не по этому перечню.

        Кадр без описания сюда входит: его никто не назвал чужим экраном, а
        молчание контура не имеет права снимать экран с проверки. Неполный ответ
        скана (текст без описания, описание без текста) — тоже: фильтром он не
        стал и остаётся проверяем по тому, что в нём есть.
        """
        filtered = self.filtered_steps
        return [frame for frame in self.frames if frame.step not in filtered]

    @property
    def scan_coverage(self) -> str:
        """Покрытие пути описаниями — чем оба узла объясняют «не видели».

        Снятые кадры в покрытие входят (скан их разобрал), но называются
        отдельно: «13 из 15, отфильтровано кадров: 2» значит, что 2 экрана — не
        клиентский путь, и судить можно по 11, а не по 13.
        """
        filtered = sum(
            1 for readout in self.frame_readouts if readout.need_filter
        )
        parts = [f"{len(self.frame_readouts)} из {len(self.frames)}"]
        if filtered:
            parts.append(f"отфильтровано кадров: {filtered}")
        if self.scan_truncated:
            parts.append("скан оборван досрочно")
        return ", ".join(parts)

    @property
    def scan_truncated(self) -> bool:
        """Скан оборвался раньше последнего кадра — описаны не все кадры пути.

        Пустой скан сюда не попадает: «не смотрели» и «смотрели да оборвались»
        для отчёта случаи разные, а первый описан кадр — уже начало скана.
        """
        return 0 < len(self.frame_readouts) < len(self.frames)

    @property
    def scan_verdict(self) -> Optional[bool]:
        """Вердикт «путь ли это» по описаниям кадров; `None` — вердикта нет.

        Считается только по отвеченным кадрам: молчание контура не имеет права
        сложиться в «пути нет» — та же политика, что в `verdict_of`. Кадров с
        ответом нет вовсе — вердикта нет, и прогон идёт вслепую, а не отклоняется.
        """
        answered = [
            readout for readout in self.frame_readouts if readout.answered
        ]
        if not answered:
            return None
        return any(readout.is_journey for readout in answered)

    def scan_context(self, last: int) -> list[str]:
        """Описания последних `last` разобранных кадров — контекст следующего.

        Отрезается только хвост: без предела список рос бы вместе с путём, и
        каждый следующий вопрос платил бы за всё прошлое пути, хотя фаза
        текущего кадра определяется ближайшими экранами. Отфильтрованные кадры
        в контекст не идут: следующему кадру не о чем судить по экрану, с
        которого нечего было переписать.

        :param last: сколько описаний помещать в вопрос; 0 — контекста нет
        :return: описания кадров в порядке пути, не длиннее `last`
        """
        if last <= 0:
            return []
        return [
            readout.listing()
            for readout in self.frame_readouts[-last:]
            if readout.usable
        ]

    def _required_criteria(self) -> CXCriteriaModel:
        """Справочник прогона; без него перечни строить не по чему."""
        if self.criteria_model is None:
            raise ValueError(
                "в состоянии нет критериальной модели — "
                "её обязан положить вход прогона"
            )
        return self.criteria_model
