from src.pipeline.schemas.nodes import CXFrame
from src.pipeline.settings.params.gigachat import FILE_REF


def image_part(frame: CXFrame) -> dict:
    """Кадр частью содержимого сообщения: ссылкой, если он уже в контуре.

    Байты кадра уходят модели ровно один раз — когда её про него спрашивают
    впервые. Дальше кадр живёт в контуре загруженным файлом, и его номер в вопросе
    стоит там же, где раньше стояли байты: подпись `[кадр N]` никуда не девается,
    модель по-прежнему видит, какой из кадров какой, а прогон не платит загрузку на
    каждый вопрос. Ссылку держит сам кадр (`CXFrame.ref`), поэтому ни одна
    собирающая вопрос сторона не должна знать про кэш ничего.

    :param frame: кадр пути — со ссылкой контура либо с data-URL
    :return: часть содержимого сообщения для контракта с изображениями
    """
    if frame.ref:
        return {"type": FILE_REF, "file_id": frame.ref}
    return {
        "type": "image_url",
        "image_url": {"url": frame.data_url, "detail": "high"},
    }
