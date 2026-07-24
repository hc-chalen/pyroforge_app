from app.engines.optimization import OptimizationEngine
from app.engines.prediction import PredictionEngine
from app.engines.procurement import ProcurementEngine
from app.engines.registry_engine import RegistryEngine
from app.engines.visualization import VisualizationEngine


class EngineRegistry:
    """Registry for the five bio-computation engines."""

    def __init__(self) -> None:
        self.prediction = PredictionEngine()
        self.optimization = OptimizationEngine(prediction=self.prediction)
        self.visualization = VisualizationEngine()
        self.registry = RegistryEngine()
        self.procurement = ProcurementEngine()


_registry: EngineRegistry | None = None


def get_engine_registry() -> EngineRegistry:
    global _registry
    if _registry is None:
        _registry = EngineRegistry()
    return _registry
