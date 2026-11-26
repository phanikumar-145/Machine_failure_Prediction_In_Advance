"""
Main entry point - executes the complete predictive maintenance pipeline:

    python main.py              # run everything
    python main.py --step eda   # run a single stage
    python main.py --help

Stages: preprocess -> features -> eda -> train -> evaluate -> anomaly -> forecast -> sample
All outputs land in outputs/ and models/; progress is logged to
outputs/logs/training.log.
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src import (preprocessing, feature_engineering, eda, train_model,
                 evaluation, anomaly_detection, forecasting)

logger = logging.getLogger("pipeline")


def _configure_logging():
    config.LOGS_DIR and os.makedirs(config.LOGS_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(config.TRAINING_LOG, mode="w"),
                  logging.StreamHandler(sys.stdout)],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", default="all",
                        choices=["all", "preprocess", "features", "eda", "train",
                                 "evaluate", "anomaly", "forecast"],
                        help="Run a single stage instead of the full pipeline")
    args = parser.parse_args()
    _configure_logging()

    logger.info("=" * 78)
    logger.info("PREDICTIVE MAINTENANCE ML - pipeline start")
    logger.info("=" * 78)

    cleaned = engineered = trained = None

    if args.step in ("all", "preprocess"):
        logger.info("--- Stage 1/7: preprocessing")
        cleaned = preprocessing.clean_pipeline()
    if args.step in ("all", "features"):
        logger.info("--- Stage 2/7: feature engineering")
        cleaned = cleaned if cleaned is not None else preprocessing.clean_pipeline()
        engineered = feature_engineering.engineer_features(cleaned)
    if args.step in ("all", "eda"):
        logger.info("--- Stage 3/7: exploratory data analysis")
        cleaned = cleaned if cleaned is not None else preprocessing.clean_pipeline()
        eda.run_eda(cleaned)
    if args.step in ("all", "train"):
        logger.info("--- Stage 4/7: model training + evaluation")
        trained = train_model.train_all(engineered)
        evaluation.evaluate_models(trained["fitted"], trained["X_test"], trained["y_test"])
        evaluation.plot_feature_importance(trained["importance"])
    if args.step == "evaluate":
        logger.info("--- Stage 5/7: evaluation (standalone)")
        trained = train_model.train_all(engineered)
        evaluation.evaluate_models(trained["fitted"], trained["X_test"], trained["y_test"])
    if args.step in ("all", "anomaly"):
        logger.info("--- Stage 6/7: anomaly detection")
        cleaned = cleaned if cleaned is not None else preprocessing.clean_pipeline()
        anomaly_detection.run_anomaly_detection(cleaned)
    if args.step in ("all", "forecast"):
        logger.info("--- Stage 7/7: time-series forecasting (C-MAPSS FD001)")
        forecasting.run_forecasting()

    logger.info("Pipeline finished. See outputs/ and models/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
