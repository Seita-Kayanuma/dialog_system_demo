from .base import FacialExpressionRecognizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


class FacialExpressionRecognizer0001(FacialExpressionRecognizer):
    def __init__(self, feature_extractor=None):
        super().__init__(feature_extractor)
        self.clf = RandomForestClassifier(n_estimators=100)

    def fit_with_features(self, x, y, *args, validation_data=None, **kwargs):
        self.clf.fit(x, y)

    def predict_with_features(self, x):
        return self.clf.predict(x)

    def save_model(self, filename):
        joblib.dump(self.clf, filename)

    def load_model(self, filename):
        self.clf = joblib.load(filename)
