import joblib
import mlflow.pyfunc
import pandas as pd

from frontpage.features import INPUT_COLUMNS


class FrontpageModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.pipeline = joblib.load(context.artifacts["pipeline"])

    def predict(self, context, model_input, params=None):
        frame = pd.DataFrame(model_input)
        for col in INPUT_COLUMNS:
            if col not in frame:
                frame[col] = None
        prob = self.pipeline.predict_proba(frame[INPUT_COLUMNS])[:, 1]
        return pd.DataFrame({"prob": prob})
