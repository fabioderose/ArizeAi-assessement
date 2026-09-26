import os

from openinference.instrumentation import TracerProvider
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

DEFAULT_PROJECT_NAME = "travel-assistant"
DEFAULT_ENDPOINT = "http://localhost:6006"


def project_name() -> str:
    return os.getenv("PHOENIX_PROJECT_NAME", DEFAULT_PROJECT_NAME)


def collector_endpoint() -> str:
    return os.getenv("PHOENIX_COLLECTOR_ENDPOINT", DEFAULT_ENDPOINT)


def setup_tracing() -> TracerProvider:
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = collector_endpoint()
    tracer_provider = register(
        project_name=project_name(),
        auto_instrument=False,
        batch=True,
        verbose=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    return tracer_provider
