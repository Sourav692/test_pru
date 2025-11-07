# Databricks notebook source
# MAGIC %md
# MAGIC ## Example training

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from mlflow.client import MlflowClient
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLFlowUnityModelTemplate:
    """
    MLFlow model template for Unity Catalog with automated model registration,
    aliasing, and tagging capabilities.
    """

    def __init__(self, catalog: str, schema: str, model_name: str):
        """
        Initialize the MLFlow Unity Catalog model template.

        Args:
            catalog: Unity Catalog name
            schema: Schema name within the catalog
            model_name: Model name
        """
        self.catalog = catalog
        self.schema = schema
        self.model_name = model_name
        self.full_model_name = f"{catalog}.{schema}.{model_name}"
        self.client = MlflowClient()

        # Model configuration
        self.tags = {"Division": "AI", "Billing": "AI"}
        self.aliases = ["champion", "challenger"]

    def setup_experiment(self, dbutils) -> str:
        """Setup MLFlow experiment and return parent run ID."""
        try:
            mlflow.end_run()
            parent_run_id = mlflow_utils.setup_experiment(dbutils)
            experiment = mlflow.set_experiment(
                experiment_id=mlflow.get_run(parent_run_id).info.experiment_id
            )
            logger.info(f"Experiment setup completed. Parent run ID: {parent_run_id}")
            return parent_run_id
        except Exception as e:
            logger.error(f"Failed to setup experiment: {str(e)}")
            raise

    def preprocess_data(self, data: pd.DataFrame, target_col: str) -> Tuple:
        """
        Preprocess data using the ModelPipeline.

        Args:
            data: Input DataFrame
            target_col: Target column name

        Returns:
            Tuple of processed data (X_train, X_test, y_train, y_test, scaler)
        """
        try:
            pipeline = ModelPipeline(model=LinearRegression())
            data_clean = pipeline.fill_nulls(data)

            X_train, X_test, y_train, y_test = pipeline.train_test_split(
                data_clean, target_col=target_col
            )

            scaler, X_train_scaled = pipeline.scale_data(data=X_train)
            X_test_scaled = pipeline.scale_data(data=X_test, scaler=scaler)

            logger.info("Data preprocessing completed successfully")
            return X_train_scaled, X_test_scaled, y_train, y_test, scaler, pipeline

        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise

    def train_and_log_model(
        self, X_train, X_test, y_train, y_test, pipeline, scaler, run_id: str
    ) -> Tuple:
        """
        Train model and log to MLFlow.

        Args:
            X_train, X_test, y_train, y_test: Training and test data
            pipeline: ModelPipeline instance
            scaler: Data scaler
            run_id: MLFlow run ID

        Returns:
            Tuple of (fitted_model, predictions)
        """
        try:
            # Train model
            fitted_model = pipeline.model.fit(X_train, y_train)
            y_pred = fitted_model.predict(X_test)

            # Log predictions
            metrics.log_predictions(y_test, y_pred)

            # Create model signature
            model_signature = infer_signature(X_test, y_pred)

            # Log model with artifacts
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=pipeline,
                artifacts={"scaler": "scaler.pkl"},
                code_paths=mlflow_utils.get_code_files(dbutils),
                signature=model_signature,
                registered_model_name=self.full_model_name,
            )

            # Log scaler artifact
            mlflow.log_artifact("scaler.pkl", artifact_path="artifacts")

            logger.info("Model training and logging completed")
            return fitted_model, y_pred

        except Exception as e:
            logger.error(f"Model training/logging failed: {str(e)}")
            raise

    def log_metrics_and_artifacts(
        self, y_test, y_pred, fitted_model, X_train, target_var: str
    ):
        """
        Log comprehensive metrics and visualization artifacts.

        Args:
            y_test: Test target values
            y_pred: Model predictions
            fitted_model: Trained model
            X_train: Training features
            target_var: Target variable name
        """
        try:
            # Ensure prediction shapes match
            if not y_pred.shape == y_test.values.shape:
                y_pred = y_pred.flatten()
                y_test = (
                    y_test[target_var].values if hasattr(y_test, target_var) else y_test
                )

            # Log accuracy metrics
            metrics.log_accuracy_metrics(y_test, y_pred)

            # Generate and log plots
            metrics.plot_residuals(y_test, y_pred)
            metrics.plot_actual_vs_predicted(y_test, y_pred)
            coefficients = metrics.plot_coefficients(fitted_model, X_train)

            # SHAP analysis
            shap_values = metrics.compute_shap_values(fitted_model, X_train)
            shap_df = (
                pd.DataFrame(shap_values, columns=X_train.columns)
                .mean()
                .reset_index()
                .rename(columns={0: "shap"})
            )

            # Log SHAP values as artifact
            shap_df.to_csv("shap_values.csv", index=False)
            mlflow.log_artifact("shap_values.csv", artifact_path="analysis")

            logger.info("Metrics and artifacts logged successfully")

        except Exception as e:
            logger.error(f"Failed to log metrics/artifacts: {str(e)}")
            raise

    def register_model_with_aliases_and_tags(self, run_id: str) -> str:
        """
        Register model in Unity Catalog with aliases and tags.

        Args:
            run_id: MLFlow run ID

        Returns:
            Model version
        """
        try:
            # Get the model URI
            model_uri = f"runs:/{run_id}/model"

            # Register model
            model_version = mlflow.register_model(
                model_uri=model_uri, name=self.full_model_name, tags=self.tags
            )

            version_number = model_version.version
            logger.info(f"Model registered as version {version_number}")

            # Set aliases
            for alias in self.aliases:
                try:
                    self.client.set_registered_model_alias(
                        name=self.full_model_name, alias=alias, version=version_number
                    )
                    logger.info(f"Alias '{alias}' set for version {version_number}")
                except Exception as alias_error:
                    logger.warning(f"Failed to set alias '{alias}': {str(alias_error)}")

            # Add additional tags to the registered model
            self.client.set_registered_model_tag(
                name=self.full_model_name, key="Environment", value="Production"
            )

            return version_number

        except Exception as e:
            logger.error(f"Model registration failed: {str(e)}")
            raise

    def run_complete_pipeline(
        self,
        data: pd.DataFrame,
        target_var: str,
        dbutils,
        dependency_dict: Optional[Dict] = None,
    ):
        """
        Execute the complete ML pipeline with Unity Catalog integration.

        Args:
            data: Input DataFrame
            target_var: Target variable name
            dbutils: Databricks utilities
            dependency_dict: Optional dependency dictionary
        """
        try:
            # Setup experiment
            parent_run_id = self.setup_experiment(dbutils)

            with mlflow.start_run(run_name=f"{self.model_name}_training") as run:
                run_id = run.info.run_id

                # Preprocess data
                X_train, X_test, y_train, y_test, scaler, pipeline = (
                    self.preprocess_data(data, target_var)
                )

                # Train and log model
                fitted_model, y_pred = self.train_and_log_model(
                    X_train, X_test, y_train, y_test, pipeline, scaler, run_id
                )

                # Log comprehensive metrics and artifacts
                self.log_metrics_and_artifacts(
                    y_test, y_pred, fitted_model, X_train, target_var
                )

                # Log dependencies and tags
                mlflow_utils.log_dependencies(parent_run_id, dependency_dict)
                mlflow_utils.log_tags()

                # Log custom tags
                for key, value in self.tags.items():
                    mlflow.set_tag(key, value)

                # Register model with Unity Catalog
                version_number = self.register_model_with_aliases_and_tags(run_id)

                logger.info(
                    f"Pipeline completed successfully. Model version: {version_number}"
                )

                return {
                    "run_id": run_id,
                    "model_version": version_number,
                    "model_name": self.full_model_name,
                    "fitted_model": fitted_model,
                    "predictions": y_pred,
                }

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise


# Usage Example
def main():
    """
    Example usage of the MLFlow Unity Catalog model template.
    """
    # Initialize the template
    model_template = MLFlowUnityModelTemplate(
        catalog="your_catalog",
        schema="your_schema",
        model_name="linear_regression_model",
    )

    # Run the complete pipeline
    # results = model_template.run_complete_pipeline(
    #     data=your_data,
    #     target_var="target",
    #     dbutils=dbutils
    # )

    print(f"Model will be registered as: {model_template.full_model_name}")
    print(f"With aliases: {model_template.aliases}")
    print(f"With tags: {model_template.tags}")


if __name__ == "__main__":
    main()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load and Predict from MLFLow Experiments

# COMMAND ----------

print("Testing model", parent_run_id)
logged_model_id = parent_run_id  # Change here for specific experiments run ID

data = {"feature1": [1, 2, 3, 4, 5], "feature2": [2, 3, 4, 5, 6]}
test_df = pd.DataFrame(data).astype("float64")
predictions = mlflow_utils.load_and_predict(logged_model_id, test_df)
display(predictions)

# COMMAND ----------

