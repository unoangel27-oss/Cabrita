"""UKJ v1: conversor GGUF y utilidades de contenedor."""

__version__ = "0.1.0"

from .format import (  # noqa: F401
    ALIGN,
    MAGIC,
    VERSION,
    UkjError,
    density_code,
    parse_density,
)
from .convert import convert_gguf  # noqa: F401
from .inspect import inspect_ukj, verify_ukj  # noqa: F401
from .subs import ensure_subs, extract_subs, verify_subs  # noqa: F401
