from collections.abc import Callable
from inspect import Parameter, iscoroutinefunction, signature
from typing import Optional, Type, Union, overload

from langchain_core.runnables import RunnableConfig
from langgraph.utils.runnable import RunnableLike
from pydantic import BaseModel


@overload
def node(fn: Callable) -> RunnableLike: ...


@overload
def node(fn: None = None) -> Callable[[Callable], RunnableLike]: ...


def node(
    fn: Optional[Callable] = None,
) -> Union[RunnableLike, Callable[[Callable], RunnableLike]]:
    if fn is None:
        # Allow use this decorator in the following form: `@node()`
        # This line may be changed to `functools.partial(node, ...)`
        # when `node` will accept other arguments.
        return node

    if not iscoroutinefunction(fn):
        raise NotImplementedError

    sig = signature(fn)
    accept_kwargs = any(
        p.kind == Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    async def wrapper(state: Type[BaseModel], config: RunnableConfig):
        possible_kwargs = {**config["configurable"], "config": config}
        if accept_kwargs:
            kwargs = possible_kwargs
        else:
            kwargs = {
                k: v for k, v in possible_kwargs.items() if k in sig.parameters
            }
        return await fn(state, **kwargs)

    # Имя узла графа langgraph берёт из `__name__`, поэтому декоратор обязан
    # его сохранить. `__wrapped__` не проставляем: по нему `inspect.signature`
    # вернул бы сигнатуру до обёртки, а граф подбирает аргументы узла именно по
    # сигнатуре обёртки — `(state, config)`.
    wrapper.__name__ = fn.__name__
    wrapper.__qualname__ = fn.__qualname__
    wrapper.__doc__ = fn.__doc__
    return wrapper
