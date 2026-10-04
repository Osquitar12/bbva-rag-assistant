"""Jerarquía de excepciones de dominio para un manejo de errores explícito."""


class RAGError(Exception):
    """Error base del sistema."""


class ScrapingError(RAGError):
    pass


class VectorStoreError(RAGError):
    pass


class LLMError(RAGError):
    pass


class LLMConfigurationError(LLMError):
    pass
