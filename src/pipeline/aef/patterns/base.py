"""
Base REST API call pattern.
"""

import logging
from abc import ABC

import tenacity

from src.pipeline.aef.exceptions import RetryError, ServerError

logger = logging.getLogger(__name__)


class BaseConnector(ABC):
    """
    Base REST API call pattern.

    ВНИМАНИЕ — уровень повторов здесь НЕ РАБОТАЕТ. `async_call_retry` создаётся,
    но не вызывается ниоткуда: ни один класс не оборачивает в него обращение к
    контуру. Реальные повторы живут в двух других местах — политика узла
    (`llm_retry_policy` в `pipeline.aef.patterns.retry`) и цикл `repeat_recoverable`
    там же. Настроенный `retry_limit` (по умолчанию 3) ни на что не влияет.

    Держать в голове при чтении: транспорты контура падают сразу, без повтора,
    и узел уходит в деградацию. Разбор — `docs/project-spec/09-maintenance/`
    `improvements.md`, пункты I-003 и I-011.
    """

    def __init__(
        self,
        base_url: str,
        retry_limit: int,
        retry_increment: int,
        request_timeout: int = 10,
    ):
        """
        :param base_url: base URL for API requests
        :param retry_limit: maximum number of retries
        :param retry_increment: time increment for each retry
        :param request_timeout: timeout for API requests
        """
        self._base_url = base_url
        self.request_timeout = request_timeout

        # Мёртвый уровень повторов: см. пояснение в докстринге класса.
        # Ни один вызов не делает `async for attempt in self.async_call_retry`.
        self.async_call_retry = tenacity.AsyncRetrying(
            stop=tenacity.stop_after_attempt(retry_limit),
            after=tenacity.after_log(logger, logging.DEBUG),
            before=tenacity.before_log(logger, logging.DEBUG),
            retry=tenacity.retry_if_exception_type(
                (ServerError, TimeoutError, RetryError)
            ),
            wait=tenacity.wait_incrementing(
                start=retry_increment, increment=retry_increment
            ),
            reraise=True,
        )

    async def health_check(self):
        """
        Check the health of the API.
        """
        logger.info(f"Checking health of {self._base_url}")
