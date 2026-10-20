"""
Совместимость speechbrain 1.0.x с новым huggingface_hub.

speechbrain 1.0.3 вызывает hf_hub_download(use_auth_token=...) и ждёт ValueError/HTTPError
для отсутствующих файлов; в новом huggingface_hub первого аргумента нет, а ошибка другая.
Подмена действует только внутри `with hf_compat():` — глобально она ломает transformers,
который рассчитывает на штатные исключения huggingface_hub.

    with hf_compat():
        model = HIFIGAN.from_hparams(...)
"""

from contextlib import contextmanager

import huggingface_hub


@contextmanager
def hf_compat():
    orig = huggingface_hub.hf_hub_download

    def wrapped(*args, use_auth_token=None, **kwargs):
        if use_auth_token is not None:
            kwargs.setdefault("token", use_auth_token)
        try:
            return orig(*args, **kwargs)
        except huggingface_hub.errors.EntryNotFoundError as e:
            # speechbrain ловит ValueError для необязательных файлов (например, custom.py)
            raise ValueError("File not found on HF hub") from e

    huggingface_hub.hf_hub_download = wrapped
    try:
        yield
    finally:
        huggingface_hub.hf_hub_download = orig
