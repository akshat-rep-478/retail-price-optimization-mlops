from constants import MODEL_NAME

from zenml import __version__ as zenml_version
from zenml.integrations.bentoml.steps import bento_builder_step



bento_builder = bento_builder_step.with_options(
    parameters=dict(
        model_name=MODEL_NAME,
        model_type="sklearn",
        service="service.py:RetailPriceService",
        labels={
            "framework": "sklearn",
            "dataset": "retail",
            "zenml_version": "0.41.0",
        },
        exclude=["data"],
        python={
            "packages": ["zenml", "scikit-learn"],
        },
    )
)