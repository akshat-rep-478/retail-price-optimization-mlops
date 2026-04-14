import bentoml
import numpy as np
from constants import MODEL_NAME

# 1. Define the service using the class-based approach
# The decorator automatically handles service instantiation
@bentoml.service(
    name="retail_price_service",
    traffic={"timeout": 60},
)
class RetailPriceService:

    def __init__(self):
        # 3. Load the model directly using the integration's load_model
        self.model = bentoml.sklearn.load_model("retail_price:latest")

    @bentoml.api
    def predict(self, inp: np.ndarray) -> np.ndarray:
        # 4. Use standard sklearn predict since we loaded the model in __init__
        return self.model.predict(inp)

# CRITICAL: Do NOT manually instantiate the class (e.g., svc = RetailPriceService())
# ZenML/BentoML looks for the class decorated with @bentoml.service.