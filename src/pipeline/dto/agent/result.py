"""
Ответ прогона: из состояния графа в формат результата анализа.

Граф занимается только разговором с моделью и складывает в состояние её ответы.
Здесь же — вся арифметика отчёта: разметка блоков по вердиктам, штраф за находки,
причины деградации и сводка. Ответ формируется вне графа, поэтому его можно
пересобрать по уже готовому состоянию, не трогая модель.
"""

from typing import Optional

from src.pipeline import scoring
from src.pipeline.schemas.nodes import (
    CXBlockFindings,
    CXCriteriaModel,
    CXJourneyDetails,
    CXModelInfo,
    CXSubVerdict,
)
from src.pipeline.schemas.result import (
    AnalyzeResult,
    CriterionOut,
    DetectedEntity,
    Friction,
    Issue,
    ScreenOut,
    SubCriterionOut,
)
from src.pipeline.schemas.state.agent import AgentStateSchema
from src.pipeline.settings.params.nodes import (
    NOT_ASKED_PREFIX,
    UNMATCHED_PREFIX,
    CXItemStatus,
    CXNodeStatus,
)
from src.pipeline.utils.findings import (
    consolidated_issues,
    covered_by,
    dedup_issues,
    findings_status,
    issues_from_findings,
)
from src.pipeline.utils.friction_findings import frictions_out
from src.pipeline.utils.journey import verdict_of
from src.pipeline.utils.screen_findings import screens_out


class AgentResult:
    """Ответ прогона по состоянию графа.

    Состояние — единственный вход: находки блоков и сводный ответ модели. Всё
    остальное (справочник блоков и имя модели, которой состоялось сведение) к
    ответу отношения к разговору не имеет и приходит явно.
    """

    def __init__(
        self,
        state: AgentStateSchema,
        criteria: CXCriteriaModel,
        analyzed_by: Optional[str] = None,
    ) -> None:
        """
        :param state: состояние графа после прогона: находки блоков и сведение
        :param criteria: критериальная модель — состав блоков и подкритериев
        :param analyzed_by: модель, отвечавшая на сведении, — попадает в ответ
        """
        self._state = state
        self._criteria = criteria
        self._analyzed_by = analyzed_by
        self._fallbacks: list[str] = []
        self._degradation: list[str] = []
        self._blocks: dict[str, CXBlockFindings] = {
            found.source_key: found for found in state.criteria_model_findings
        }

    def get_result(self) -> AnalyzeResult:
        """Итог прогона: балл из оценки блоков минус штраф за находки.

        Штраф считается по единому списку проблем — одна проблема штрафуется один
        раз, независимо от того, сколькими источниками она найдена.

        Отклонения всех проверяющих веток — блоков, поэкранного прохода и
        проверок проблем пути — сходятся в этот список через сведение
        (`pipeline.agent.nodes.consolidate_findings`), поэтому одна проблема на одном
        экране, названная двумя ветками, штрафует прогон один раз. Секции
        `frictions` и `screens` остаются отдельными: в первой трения показаны как
        вид пути, во второй находки держат координаты пина; повторно за них не
        штрафуя.

        Порядок внутри ответа жёсткий: блоки строятся по вердиктам, а `score_run`
        перещёлкивает их статусы по сведённым находкам и только тогда считает баллы
        — балл блока есть доля `ok` после перещёлкивания, и её уже перемножают доли
        зон тех находок, что на блок легли.

        `friction_penalty` отдаётся нулём: у этой методики списание каждой находки
        живёт внутри её блока, и отдельной строкой его вычесть второй раз — значит
        наказать прогон дважды за одну и ту же находку. Поле ответа при этом остаётся
        числом, а не пустотой: у прогонов аддитивной методики оно ненулевое.

        Версия справочника, которым мерили, едет в ответе отдельными полями
        (`eval_model_id`, `eval_model_version`): балл без неё несопоставим с
        другим прогоном — каталог перекалибровывают, и та же находка в новом
        выпуске стоит других баллов.

        :return: ответ прогона в формате ядра
        """
        self._same_catalog()
        details = self._state.journey_details
        if verdict_of(details) is False:
            return self._not_journey(details)
        self._journey_degradation(details)

        failed_blocks = self._unanswered_blocks()
        issues = self._issues()
        criteria = self._criteria_out(issues)
        frictions = self._frictions()
        self._frictions_degradation()
        self._autofill_degradation()
        self._screen_degradation()
        run = scoring.score_run(
            criteria, issues, excluded_blocks=frozenset(failed_blocks)
        )
        degraded = bool(self._degradation or failed_blocks or self._fallbacks)
        note = self._summary() or None
        return AnalyzeResult(
            run_id=self._state.run_id,
            model=self._analyzed_by,
            eval_model_id=self._criteria.info.id or None,
            eval_model_version=self._criteria.info.version or None,
            screenshot_keys=list(self._state.screenshot_keys),
            detected=self._detected(details),
            score=run.score,
            base_score=run.base_score,
            friction_penalty=run.penalty,
            rating=run.rating,
            critical_count=run.critical_count,
            stop_flag=run.stop_flag,
            criteria=criteria,
            screens=self._screens(),
            frictions=frictions,
            issues=issues,
            summary=note,
            feedback=note,
            analysis_status="partial" if degraded else "complete",
            is_degraded=degraded,
            degradation_reasons=list(self._degradation),
            failed_blocks=list(failed_blocks),
            fallbacks_used=list(self._fallbacks),
            pipeline=self._pipeline(),
        )

    def _same_catalog(self) -> None:
        """Снимок справочника у прогона один: тем, чем меряли, тем и размечают.

        Справочник кладёт в состояние вход прогона, а ответу его передают
        отдельно: пересборка ответа по записанному состоянию или вход со снятым с
        поддержки конфигом могут принести две редакции каталога. Балл тогда
        считается по линейке штрафов одной редакции, а блоки, коды и зоны в
        ответе подписаны другой — и в самом ответе это не различимо, потому что
        обе половины выглядят собранной моделью.

        Состояние без снимка — не разнобой: его заполняет вход прогона, а
        короткие прогоны одной ветки собирают состояние руками.
        """
        pinned = self._state.criteria_model
        if pinned is None:
            return

        def named(info: CXModelInfo) -> str:
            return f"{info.key}@{info.version} (id {info.id or 'не назван'})"

        mine, pinned_info = self._criteria.info, pinned.info
        if not (mine.version and pinned_info.version):
            return
        if (mine.id, mine.version) == (pinned_info.id, pinned_info.version):
            return
        raise ValueError(
            f"прогон измерялся справочником {named(pinned_info)}, а ответ "
            f"строится по {named(mine)}: снимок обязан быть одним"
        )

    def _pipeline(self) -> dict:
        """Сырьё каждой фазы прогона — шестью разделами по их порядку.

        Разделы названы и пронумерованы теми же именами, что и `trace.pipeline`
        ответа ядра, поэтому два прогона сравниваются раздел в раздел, а не
        «наше поле против их поля»:

        * `1_attribution` — распознавание пути (`pipeline.agent.nodes.summarize_journey`);
          поля идут за `CXJourneyDetails`, где сценарий пути называется `scenario`,
          а не `journey`, как у ядра;
        * `2_criteria` — ответы блоков (`pipeline.agent.criteria_model_frames`): здесь
          ответ блока таким, каким его дал контур, до разметки и до балла; блок,
          где контур молчал, остаётся со своим статусом и без вердиктов;
        * `3_screens` — поэкранный проход (`pipeline.agent.screen`): описания кадров
          и их отклонения с координатами пина. Пустой раздел значит, что проход
          не запускался вовсе (путь не подтверждён), — пропуск с отметкой
          `not_asked:` остаётся видимым;
        * `4_journey` — находки шести проверок проблем пути
          (`pipeline.agent.journey_frictions`), по проверке на элемент;
        * `5_autofill` — проход автозаполнения (`pipeline.agent.autofill`) его же
          контейнером: находки экранов, объяснение пустоты и отметка о плохом
          ответе. Пустой раздел значит, что проход не запускался вовсе (путь не
          подтверждён), — пропуск с отметкой `not_asked:` остаётся видимым;
        * `6_consolidation` — сведённый ответ модели
          (`pipeline.agent.nodes.consolidate_findings`), пустой, если сведения не было.

        Сырьё, а не итог: `criteria` и `issues` ответа — размеченные и оштрафованные
        производные, и по ним не видно, что именно сказал контур. Пустые разделы
        остаются в ответе нарочно: отсутствие фазы должно отличаться от фазы,
        которая отработала и ничего не нашла.

        :return: шесть разделов прогона в порядке его фаз
        """
        details = self._state.journey_details
        consolidated = self._state.cx_consolidation
        autofill = self._state.autofill_findings
        screen = self._state.screen_findings
        criteria: list[dict] = [
            {"key": key, **found.model_dump(mode="json")}
            for key, found in self._blocks.items()
        ]
        return {
            "1_attribution": (
                details.model_dump(mode="json") if details else {}
            ),
            "2_criteria": {"criteria": criteria},
            "3_screens": (screen.model_dump(mode="json") if screen else {}),
            "4_journey": {
                "checks": [
                    found.model_dump(mode="json")
                    for found in self._state.journey_findings
                ]
            },
            "5_autofill": (
                autofill.model_dump(mode="json") if autofill else {}
            ),
            "6_consolidation": (
                consolidated.model_dump(mode="json") if consolidated else {}
            ),
        }

    def _issues(self) -> list[Issue]:
        """Единый список проблем прогона.

        Сведение может и не состояться: без него прогон остаётся с сырыми
        отклонениями обеих веток — форма у них одна, и фолбэк не разбирается, кто
        их нашёл; разбор не прячем — прогон помечается частичным.

        Состоявшееся сведение тоже не имеет права терять находку: оно сливает
        дубли, а сырое отклонение, которому не нашлось места в сведённом списке,
        возвращается в отчёт сам — с экраном, кодом и цитатой автора. В прогоне
        047 сведение оставило четыре проблемы из тринадцати найденных, и девять
        названных экранов пропали из отчёта молча.
        """
        consolidated = self._state.cx_consolidation
        frames_count = len(self._state.frames)
        if consolidated is None:
            self._fallbacks.append("consolidation_fallback")
            self._degradation.append("consolidation:error")
            return issues_from_findings(
                self._criteria, self._state.findings, frames_count
            )
        issues = consolidated_issues(
            self._criteria, consolidated, frames_count
        )
        lost = [
            issue
            for issue in issues_from_findings(
                self._criteria, self._state.findings, frames_count
            )
            if not covered_by(issue, issues)
        ]
        if lost:
            reason = f"consolidation_dropped:{len(lost)}"
            self._fallbacks.append(reason)
            self._degradation.append(reason)
        return dedup_issues([*issues, *lost])

    def _frictions(self) -> list[Friction]:
        """Трения пути: находки шести проверок ветки проблем пути.

        Ветка пишет в свой ключ состояния; в балл её отклонения приходят через
        сведение — тем же путём, что и находки блоков, — а здесь они показываются
        отдельной секцией отчёта, повторно не штрафуясь. Разбирает их
        `pipeline.utils.friction_findings`: там же решается, какой кадр считать уликой
        и какой зоной покрасить находку.
        """
        return frictions_out(
            self._state.journey_frictions, len(self._state.frames)
        )

    def _screens(self) -> list[ScreenOut]:
        """Поэкранный разбор: описание каждого кадра и что на нём мешает.

        Единственная секция отчёта, где находка показывает, где она на экране:
        координаты пина модель отдаёт процентами от размеров кадра, и держит их
        проход отдельно от сведённых проблем. Кадр остаётся в перечне и без
        находок — «экран разобран, чисто» обязано отличаться от «экрана нет».
        """
        return screens_out(
            self._state.screen_findings, len(self._state.frames)
        )

    def _frictions_degradation(self) -> None:
        """Несостоявшаяся проверка пути — в причины деградации прогона.

        Пустой список трений значит «проверено, чисто» только при статусе
        проверки `success`: иначе шесть сорванных обращений к контуру дали бы
        чистый отчёт, и «проблем нет» было бы неотличимо от «проверки не было».

        Отметки разметки читаются теми же снятием, что у поэкранной ветки
        (`_screen_degradation`): `unmatched:` — штатный итог сверки с каталогом,
        находка осталась внекаталожной и штрафа за это нет; `mapping_failed:` —
        адрес не получила вся проверка, и её находки ушли в сведение без кодов,
        что прогоном частичным быть обязано.
        """
        for check_key, found in self._state.journey_frictions.items():
            status = CXNodeStatus(found.status)
            if status is not CXNodeStatus.SUCCESS:
                self._degradation.append(
                    f"friction:{check_key}:{status.value}"
                )
                continue
            marks = self._fault_marks(found.degraded)
            if marks:
                self._degradation.append(f"friction:{check_key}:{marks}")

    def _autofill_degradation(self) -> None:
        """Несостоявшийся проход автозаполнения — в причины деградации прогона.

        Пустой перечень находок значит «предзаполнять было нечего» только при
        статусе `success` и без счётчика сорванных групп: иначе молчание контура
        выглядело бы чистым путём. Отметка `not_asked:` деградацией не считается:
        ветка отказалась спрашивать по настройке или за отсутствием кадров — это
        её штатный ответ, а не сбой, и он остаётся только в разделе `5_autofill`.
        """
        found = self._state.autofill_findings
        if found is None:
            return
        status = CXNodeStatus(found.status)
        if status is not CXNodeStatus.SUCCESS:
            self._degradation.append(f"autofill:{status.value}")
            return
        if found.degraded and not found.degraded.startswith(NOT_ASKED_PREFIX):
            self._degradation.append(f"autofill:{found.degraded}")

    def _screen_degradation(self) -> None:
        """Несостоявшийся поэкранный проход — в причины деградации прогона.

        Счётчики прохода значат больше, чем у соседних веток: без части групп
        кадры пути не разобраны вовсе, а недобор описаний значит, что отчёт
        покажет не каждый экран. Молчание контура выглядело бы чистым путём —
        кадры-то приложены.

        Две отметки деградацией не считаются: `not_asked:` значит, что ветка не
        спрашивала, потому что кадров нет, а `unmatched:` — что у части находок
        не нашлось кода справочника и они остались внекаталожными, при своей зоне
        и пине. Обе остаются в разделе `3_screens`, но из-за них прогон не
        частичный: иначе каждая находка вне справочника обваливала бы статус
        прогона, хотя единственное её отличие — пустой код.
        """
        found = self._state.screen_findings
        if found is None:
            return
        status = CXNodeStatus(found.status)
        if status is not CXNodeStatus.SUCCESS:
            self._degradation.append(f"screen:{status.value}")
            return
        marks = self._fault_marks(found.degraded)
        if marks:
            self._degradation.append(f"screen:{marks}")

    @staticmethod
    def _fault_marks(degraded: str) -> str:
        """Сбойные отметки `degraded` — штатные сняты поштучно, а не вся строка.

        Узел склеивает отметки, и пропуск может стоять рядом со сбоем:
        «unmatched:1/3; mapping_failed:TimeoutError» обязан деградировать прогон,
        хотя начинается он со счётчика без кода.

        :param degraded: строка отметок ветки
        :return: сбойные отметки той же склейкой, пусто — если сбоев нет
        """
        parts = (part.strip() for part in degraded.split("; "))
        return "; ".join(
            part
            for part in parts
            if part
            and not part.startswith((NOT_ASKED_PREFIX, UNMATCHED_PREFIX))
        )

    def _not_journey(
        self, details: Optional[CXJourneyDetails]
    ) -> AnalyzeResult:
        """Ответ прогону, который распознавание пути не подтвердило.

        Ни блоков, ни балла здесь нет по делу: не-путь не оценивают, и это не
        деградация — это ответ. Причина — слова модели о материале, а не наша догадка;
        статус отдельный от `partial`, иначе «не путь» неотличим от «контур упал».

        Версия справочника в таком ответе не названа: им не измеряли ничего, и
        строчка прогона, где не было ни блока, ни балла, не имеет права выглядеть
        измеренной какой-то редакцией каталога (так же делает и контур, которым
        это измеряют).

        :param details: атрибуция пути с вердиктом «не путь»
        :return: короткий ответ без блоков и без балла
        """
        note = (
            details.note.strip() if details else ""
        ) or "Загруженное не похоже на клиентский путь."
        return AnalyzeResult(
            run_id=self._state.run_id,
            model=self._analyzed_by,
            screenshot_keys=list(self._state.screenshot_keys),
            criteria=[],
            issues=[],
            summary=note,
            feedback=note,
            analysis_status="rejected",
            is_journey=False,
            note=note,
            pipeline=self._pipeline(),
        )

    def _journey_degradation(
        self, details: Optional[CXJourneyDetails]
    ) -> None:
        """Чем распознавание пути не хватило — в причины деградации или в фолбэки.

        Блоки в этом случае всё равно проверяются: останавливать прогон имеет право
        только подтверждённый вердикт. Но отчёт обязан говорить, что продукт не назван,
        иначе «проверили вслепую» неотличимо от «проверили зная продукт».
        """
        if details is None:
            self._degradation.append("journey_details:absent")
            return
        if CXNodeStatus(details.status) is not CXNodeStatus.SUCCESS:
            self._degradation.append(f"journey_details:{details.status}")
        elif details.degraded:
            reason = f"journey_details:{details.degraded}"
            self._fallbacks.append(reason)
            self._degradation.append(reason)

    def _detected(self, details: Optional[CXJourneyDetails]) -> DetectedEntity:
        """Распознанная привязка пути: имена из распознавания, идентификаторов нет.

        Идентификаторов каталога у распознавания не бывает — имена уходят как
        предзаполнение ручной привязки. Без вердикта наружу не отдаём и имена:
        недобранное имя упавшего узла неотличимо от нераспознанного продукта.
        """
        if verdict_of(details) is None:
            return DetectedEntity()
        return DetectedEntity(
            org_name=details.organization or None,
            product_name=details.product or None,
            journey_name=details.scenario or None,
        )

    def _unanswered_blocks(self) -> list[str]:
        """Блоки без ответа и коды их деградации.

        Блок без ответа — это деградация прогона, а не «всё в порядке»: без причины
        пустой блок неотличим от блока без находок.
        """
        failed: list[str] = []
        for block in self._criteria.blocks:
            status = findings_status(self._blocks.get(block.key))
            if status == CXNodeStatus.SUCCESS:
                continue
            failed.append(block.key)
            self._degradation.append(f"block:{block.key}:{status.value}")
        return failed

    def _criteria_out(self, issues: list[Issue]) -> list[CriterionOut]:
        """Оценка блоков: здесь ещё вердикты модели, перещёлкивание — позже.

        Модель отвечает по каждому подкритерию, и только она знает, что здесь `na`:
        неприменимое к типу пути остаётся вне знаменателя балла, поэтому «не
        проверено» не стоит ни штрафных, ни защитных баллов. Подкритерий без
        вердикта — `na`: молчание модели не может стоить баллов.

        Статусы по привязанным находкам и баллы блоков считает
        `pipeline.scoring.run.score_run` — он вызывается после того, как эти блоки
        построены, потому что балл блока есть доля `ok` уже после перещёлкивания.
        Отсюда и пустые `score` с `rating` у возвращаемых блоков.

        :param issues: сведённые находки прогона
        :return: блоки оценки с вердиктными статусами, до балла
        """
        blocks: list[CriterionOut] = []
        for block in self._criteria.blocks:
            found = self._blocks.get(block.key)
            block_issues = [i for i in issues if i.criterion == block.key]
            skipped = findings_status(found) != CXNodeStatus.SUCCESS
            verdicts: dict[str, CXSubVerdict] = {
                v.key: v for v in (found.items if found else [])
            }
            items: list[SubCriterionOut] = []
            for criteria in block.criteria:
                hits = [i for i in block_issues if i.check_key == criteria.key]
                verdict = verdicts.get(criteria.key)
                items.append(
                    SubCriterionOut(
                        key=criteria.key,
                        name=criteria.text,
                        status=verdict.status if verdict else CXItemStatus.NA,
                        note=verdict.note.strip() if verdict else "",
                        improvement=hits[0].rec if hits else "",
                        zone=scoring.worst_zone(i.zone for i in hits),
                    )
                )
            blocks.append(
                CriterionOut(
                    key=block.key,
                    name=block.name,
                    hint=block.essence,
                    note=(
                        "Блок не удалось оценить полностью из-за сбоя распознавания; результат отчёта частичный."
                        if skipped
                        else (found.note if found else "")
                    ),
                    items=items,
                    zone=scoring.worst_zone(i.zone for i in items),
                )
            )
        return blocks

    def _summary(self) -> str:
        """Сводка по пути: текст сведения либо сводка по заметкам блоков.

        Ответа без сводки контур не принимает: готовность отчёта меряется в том
        числе по тексту. Поэтому фолбэк говорит ровно то, что известно: какие
        блоки отработали, что сказал каждый, и что сводного текста модель не дала.
        """
        consolidated = self._state.cx_consolidation
        if consolidated is not None and consolidated.summary.strip():
            return consolidated.summary.strip()

        lines = ["Сводного текста от модели нет, сводка — по заметкам блоков."]
        for block in self._criteria.blocks:
            found = self._blocks.get(block.key)
            if found is None or findings_status(found) != CXNodeStatus.SUCCESS:
                lines.append(f"— {block.name}: блок не оценён.")
                continue
            note = found.note.strip().rstrip(".")
            lines.append(
                f"— {block.name}: {note}."
                if note
                else f"— {block.name}: без замечаний."
            )
        return "\n".join(lines)
