# Databricks notebook source
# MAGIC %md
# MAGIC ## Example inference

# COMMAND ----------

"""
Databricks Unity Catalog Model Inference Pipeline
Supports batch and real-time inference with comprehensive error handling and monitoring.
"""

import mlflow
import mlflow.pyfunc
from mlflow.client import MlflowClient
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime
import json
from dataclasses import dataclass, asdict
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Data class for storing inference results with metadata."""

    predictions: Union[np.ndarray, List]
    model_version: str
    model_name: str
    inference_time_ms: float
    timestamp: str
    input_shape: tuple
    prediction_count: int
    metadata: Optional[Dict] = None

    def to_dict(self):
        """Convert to dictionary."""
        result = asdict(self)
        if isinstance(self.predictions, np.ndarray):
            result["predictions"] = self.predictions.tolist()
        return result


class UCModelInferencePipeline:
    """
    Inference pipeline for Databricks Unity Catalog models.
    Supports multiple model versions, aliases, and comprehensive monitoring.
    """

    def __init__(
        self,
        catalog: str,
        schema: str,
        model_name: str,
        version: Optional[str] = None,
        alias: Optional[str] = "champion",
    ):
        """
        Initialize the inference pipeline.

        Args:
            catalog: Unity Catalog name
            schema: Schema name
            model_name: Model name
            version: Specific model version (optional)
            alias: Model alias to use if version not specified (default: "champion")
        """
        self.catalog = catalog
        self.schema = schema
        self.model_name = model_name
        self.full_model_name = f"{catalog}.{schema}.{model_name}"
        self.version = version
        self.alias = alias
        self.client = MlflowClient()
        self.model = None
        self.model_uri = None
        self.model_metadata = {}

        logger.info(f"Initialized inference pipeline for {self.full_model_name}")

    def load_model(self, force_reload: bool = False) -> mlflow.pyfunc.PyFuncModel:
        """
        Load model from Unity Catalog.

        Args:
            force_reload: Force reload even if model is already loaded

        Returns:
            Loaded MLflow model
        """
        try:
            if self.model is not None and not force_reload:
                logger.info("Using cached model")
                return self.model

            # Determine model URI
            if self.version:
                self.model_uri = f"models:/{self.full_model_name}/{self.version}"
                logger.info(f"Loading model version: {self.version}")
            elif self.alias:
                self.model_uri = f"models:/{self.full_model_name}@{self.alias}"
                logger.info(f"Loading model with alias: {self.alias}")
            else:
                # Load latest version
                self.model_uri = f"models:/{self.full_model_name}/latest"
                logger.info("Loading latest model version")

            # Load the model
            start_time = time.time()
            self.model = mlflow.pyfunc.load_model(self.model_uri)
            load_time = (time.time() - start_time) * 1000

            # Get model metadata
            self._load_model_metadata()

            logger.info(f"Model loaded successfully in {load_time:.2f}ms")
            logger.info(
                f"Model version: {self.model_metadata.get('version', 'unknown')}"
            )

            return self.model

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _load_model_metadata(self):
        """Load and cache model metadata."""
        try:
            # Get model version details
            if self.version:
                version_info = self.client.get_model_version(
                    name=self.full_model_name, version=self.version
                )
            else:
                # Get version from alias
                model_version = self.client.get_model_version_by_alias(
                    name=self.full_model_name, alias=self.alias
                )
                version_info = model_version

            self.model_metadata = {
                "version": version_info.version,
                "run_id": version_info.run_id,
                "status": version_info.status,
                "creation_timestamp": version_info.creation_timestamp,
                "last_updated_timestamp": version_info.last_updated_timestamp,
                "tags": version_info.tags if hasattr(version_info, "tags") else {},
                "description": (
                    version_info.description
                    if hasattr(version_info, "description")
                    else ""
                ),
            }

        except Exception as e:
            logger.warning(f"Could not load model metadata: {str(e)}")
            self.model_metadata = {}

    def predict(
        self, data: Union[pd.DataFrame, np.ndarray, Dict], return_metadata: bool = True
    ) -> Union[InferenceResult, np.ndarray]:
        """
        Make predictions on input data.

        Args:
            data: Input data (DataFrame, array, or dict)
            return_metadata: Whether to return InferenceResult with metadata

        Returns:
            Predictions with optional metadata
        """
        try:
            # Ensure model is loaded
            if self.model is None:
                self.load_model()

            # Preprocess input
            processed_data = self._preprocess_input(data)

            # Make prediction
            start_time = time.time()
            predictions = self.model.predict(processed_data)
            inference_time = (time.time() - start_time) * 1000

            logger.info(
                f"Inference completed in {inference_time:.2f}ms "
                f"for {len(processed_data)} samples"
            )

            if return_metadata:
                return InferenceResult(
                    predictions=predictions,
                    model_version=str(self.model_metadata.get("version", "unknown")),
                    model_name=self.full_model_name,
                    inference_time_ms=inference_time,
                    timestamp=datetime.now().isoformat(),
                    input_shape=processed_data.shape,
                    prediction_count=len(predictions),
                    metadata=self.model_metadata,
                )
            else:
                return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def batch_predict(
        self, data: pd.DataFrame, batch_size: int = 1000, return_dataframe: bool = True
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Make predictions in batches for large datasets.

        Args:
            data: Input DataFrame
            batch_size: Number of samples per batch
            return_dataframe: Return results as DataFrame with original data

        Returns:
            Predictions as DataFrame or array
        """
        try:
            if self.model is None:
                self.load_model()

            total_samples = len(data)
            predictions_list = []

            logger.info(f"Starting batch prediction for {total_samples} samples")

            for i in range(0, total_samples, batch_size):
                batch = data.iloc[i : i + batch_size]
                batch_predictions = self.model.predict(batch)
                predictions_list.append(batch_predictions)

                if (i + batch_size) % (batch_size * 10) == 0:
                    logger.info(
                        f"Processed {min(i + batch_size, total_samples)}/{total_samples} samples"
                    )

            # Combine predictions
            all_predictions = np.concatenate(predictions_list)

            logger.info(f"Batch prediction completed for {total_samples} samples")

            if return_dataframe:
                result_df = data.copy()
                result_df["prediction"] = all_predictions
                result_df["model_version"] = self.model_metadata.get(
                    "version", "unknown"
                )
                result_df["prediction_timestamp"] = datetime.now().isoformat()
                return result_df
            else:
                return all_predictions

        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise

    def predict_from_table(
        self,
        spark,
        input_table: str,
        output_table: Optional[str] = None,
        batch_size: int = 10000,
    ) -> pd.DataFrame:
        """
        Make predictions on data from a Databricks table.

        Args:
            spark: Spark session
            input_table: Fully qualified input table name
            output_table: Optional output table name to save results
            batch_size: Batch size for processing

        Returns:
            DataFrame with predictions
        """
        try:
            logger.info(f"Reading data from table: {input_table}")

            # Read data from table
            df = spark.table(input_table).toPandas()

            # Make predictions
            results_df = self.batch_predict(
                data=df, batch_size=batch_size, return_dataframe=True
            )

            # Save to output table if specified
            if output_table:
                logger.info(f"Saving results to table: {output_table}")
                spark_df = spark.createDataFrame(results_df)
                spark_df.write.mode("overwrite").saveAsTable(output_table)
                logger.info(f"Results saved to {output_table}")

            return results_df

        except Exception as e:
            logger.error(f"Table prediction failed: {str(e)}")
            raise

    def _preprocess_input(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> pd.DataFrame:
        """
        Preprocess input data to appropriate format.

        Args:
            data: Input data in various formats

        Returns:
            Preprocessed DataFrame
        """
        try:
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, np.ndarray):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Handle single record or batch
                if isinstance(next(iter(data.values())), list):
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame([data])
            else:
                raise ValueError(f"Unsupported input type: {type(data)}")

        except Exception as e:
            logger.error(f"Input preprocessing failed: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model information
        """
        try:
            if not self.model_metadata:
                self._load_model_metadata()

            # Get registered model info
            model_info = self.client.get_registered_model(self.full_model_name)

            # Get all versions
            versions = self.client.search_model_versions(
                f"name='{self.full_model_name}'"
            )

            info = {
                "model_name": self.full_model_name,
                "current_version": self.model_metadata.get("version"),
                "description": model_info.description,
                "creation_timestamp": model_info.creation_timestamp,
                "last_updated_timestamp": model_info.last_updated_timestamp,
                "tags": model_info.tags,
                "total_versions": len(versions),
                "available_versions": [v.version for v in versions],
                "current_metadata": self.model_metadata,
            }

            # Get aliases
            try:
                aliases = {}
                for version in versions:
                    if hasattr(version, "aliases") and version.aliases:
                        for alias in version.aliases:
                            aliases[alias] = version.version
                info["aliases"] = aliases
            except Exception as e:
                logger.warning(f"Could not retrieve aliases: {str(e)}")
                info["aliases"] = {}

            return info

        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            raise

    def compare_versions(self, data: pd.DataFrame, versions: List[str]) -> pd.DataFrame:
        """
        Compare predictions across multiple model versions.

        Args:
            data: Input data for comparison
            versions: List of version numbers to compare

        Returns:
            DataFrame with predictions from each version
        """
        try:
            results = data.copy()

            for version in versions:
                logger.info(f"Getting predictions from version {version}")

                # Temporarily load specific version
                temp_pipeline = UCModelInferencePipeline(
                    catalog=self.catalog,
                    schema=self.schema,
                    model_name=self.model_name,
                    version=version,
                )
                temp_pipeline.load_model()

                predictions = temp_pipeline.predict(data, return_metadata=False)
                results[f"prediction_v{version}"] = predictions

            logger.info(f"Version comparison completed for {len(versions)} versions")
            return results

        except Exception as e:
            logger.error(f"Version comparison failed: {str(e)}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the inference pipeline.

        Returns:
            Health check results
        """
        health_status = {
            "status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "model_name": self.full_model_name,
            "checks": {},
        }

        try:
            # Check if model can be loaded
            start_time = time.time()
            self.load_model()
            load_time = (time.time() - start_time) * 1000

            health_status["checks"]["model_load"] = {
                "status": "pass",
                "load_time_ms": load_time,
            }

            # Check model metadata
            if self.model_metadata:
                health_status["checks"]["metadata"] = {
                    "status": "pass",
                    "version": self.model_metadata.get("version"),
                }
            else:
                health_status["checks"]["metadata"] = {
                    "status": "warning",
                    "message": "No metadata available",
                }

            # Test prediction with dummy data
            try:
                # Create dummy input based on model signature
                dummy_data = pd.DataFrame([[0] * 5])  # Adjust based on your model
                start_time = time.time()
                self.predict(dummy_data, return_metadata=False)
                pred_time = (time.time() - start_time) * 1000

                health_status["checks"]["prediction"] = {
                    "status": "pass",
                    "inference_time_ms": pred_time,
                }
            except Exception as e:
                health_status["checks"]["prediction"] = {
                    "status": "fail",
                    "error": str(e),
                }

            # Overall status
            failed_checks = [
                k
                for k, v in health_status["checks"].items()
                if v.get("status") == "fail"
            ]

            if failed_checks:
                health_status["status"] = "unhealthy"
                health_status["failed_checks"] = failed_checks
            else:
                health_status["status"] = "healthy"

        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status


# Convenience functions for common use cases


def quick_predict(
    catalog: str,
    schema: str,
    model_name: str,
    data: Union[pd.DataFrame, Dict],
    alias: str = "champion",
) -> np.ndarray:
    """
    Quick prediction function for simple use cases.

    Args:
        catalog: Unity Catalog name
        schema: Schema name
        model_name: Model name
        data: Input data
        alias: Model alias (default: "champion")

    Returns:
        Predictions array
    """
    pipeline = UCModelInferencePipeline(
        catalog=catalog, schema=schema, model_name=model_name, alias=alias
    )
    pipeline.load_model()
    return pipeline.predict(data, return_metadata=False)


def batch_score_table(
    spark,
    catalog: str,
    schema: str,
    model_name: str,
    input_table: str,
    output_table: str,
    alias: str = "champion",
    batch_size: int = 10000,
) -> pd.DataFrame:
    """
    Batch score a Databricks table and save results.

    Args:
        spark: Spark session
        catalog: Unity Catalog name
        schema: Schema name
        model_name: Model name
        input_table: Input table name
        output_table: Output table name
        alias: Model alias
        batch_size: Batch size for processing

    Returns:
        DataFrame with predictions
    """
    pipeline = UCModelInferencePipeline(
        catalog=catalog, schema=schema, model_name=model_name, alias=alias
    )
    pipeline.load_model()

    return pipeline.predict_from_table(
        spark=spark,
        input_table=input_table,
        output_table=output_table,
        batch_size=batch_size,
    )


# Example usage
if __name__ == "__main__":
    """
    Example usage of the UC Model Inference Pipeline
    """

    # Initialize pipeline
    pipeline = UCModelInferencePipeline(
        catalog="your_catalog",
        schema="your_schema",
        model_name="linear_regression_model",
        alias="champion",  # or specify version="1"
    )

    # Load model
    pipeline.load_model()

    # Get model information
    model_info = pipeline.get_model_info()
    print(json.dumps(model_info, indent=2, default=str))

    # Make single prediction
    sample_data = pd.DataFrame(
        {"feature1": [1.0, 2.0, 3.0], "feature2": [4.0, 5.0, 6.0]}
    )

    result = pipeline.predict(sample_data, return_metadata=True)
    print(f"\nPredictions: {result.predictions}")
    print(f"Inference time: {result.inference_time_ms:.2f}ms")
    print(f"Model version: {result.model_version}")

    # Health check
    health = pipeline.health_check()
    print(f"\nHealth Status: {health['status']}")
    print(json.dumps(health, indent=2, default=str))

    # Batch prediction example
    large_dataset = pd.DataFrame(
        {"feature1": np.random.randn(10000), "feature2": np.random.randn(10000)}
    )

    batch_results = pipeline.batch_predict(
        data=large_dataset, batch_size=1000, return_dataframe=True
    )
    print(f"\nBatch predictions completed: {len(batch_results)} rows")

    # Quick predict example
    quick_result = quick_predict(
        catalog="your_catalog",
        schema="your_schema",
        model_name="linear_regression_model",
        data={"feature1": [1.0], "feature2": [2.0]},
        alias="champion",
    )
    print(f"\nQuick prediction: {quick_result}")