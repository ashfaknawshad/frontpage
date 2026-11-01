import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd

from frontpage.features import INPUT_COLUMNS


class FrontpageModel(mlflow.pyfunc.PythonModel):
    """Ranking pipeline plus an optional Platt calibrator.

    Version 1 shipped without a calibrator; it still loads because the artifact is optional.
    """

    def load_context(self, context):
        self.pipeline = joblib.load(context.artifacts["pipeline"])
        path = context.artifacts.get("calibrator")
        self.calibrator = joblib.load(path) if path else None

    def predict(self, context, model_input, params=None):
        frame = pd.DataFrame(model_input)
        for col in INPUT_COLUMNS:
            if col not in frame:
                frame[col] = None
        frame = frame[INPUT_COLUMNS]
        if self.calibrator is None:
            prob = self.pipeline.predict_proba(frame)[:, 1]
        else:
            logit = np.asarray(self.pipeline.decision_function(frame), dtype=float).reshape(-1, 1)
            prob = self.calibrator.predict_proba(logit)[:, 1]
        return pd.DataFrame({"prob": prob})
