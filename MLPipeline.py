#!/usr/bin/env python3
# i
import sklearn
import numpy as np
import pandas as pd
import shap
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt


class Pipeline:
    def __init__(self):
        self.threshold = 0.5

    def set_data(
        self,
        features,
        target,
        split_percentage=0.2,
        split_state=33,
    ):
        self.corpus_features = features
        self.corpus_target = target
        self.split_percentage = split_percentage
        self.split_state = split_state

    def set_vectorizer(self, vectorizer):
        # vectorizer should have the functions fit_transform and get_feature_names_out
        self.update_vectorizer = True
        self.vectorizer = vectorizer

    def __vectorize__(self):
        corpus_features = self.vectorizer.fit_transform(self.corpus_features)
        if isinstance(corpus_features, np.ndarray):
            self.corpus_features = corpus_features
        else:
            self.corpus_features = corpus_features.toarray()

    def __split_data__(self):
        (
            self.train_features,
            self.test_features,
            self.train_target,
            self.test_target,
        ) = sklearn.model_selection.train_test_split(
            self.corpus_features,
            self.corpus_target,
            test_size=self.split_percentage,
            random_state=self.split_state,
        )

    def set_model(self, model):
        self.update_model = True
        self.model = model

    def __fit_model__(self):
        self.model.fit(np.array(self.train_features), np.array(self.train_target))

    def __predict__(self):
        self.predictions = self.model.predict(self.test_features)

    def set_prediction_threshold(self, threshold):
        self.update_prediction_threshold = True
        self.threshold = threshold

    def __get_binary_threshold__(self):
        return np.where(self.predictions >= self.threshold, 1, 0)

    def run(self):
        all_after = False

        if self.update_vectorizer:
            all_after = True
            self.__vectorize__()
            self.__split_data__()
            self.update_vecorizer = False

        if all_after or self.update_model:
            all_after = True
            self.__fit_model__()
            self.update_model = False

        if all_after or self.update_prediction_threshold:
            all_after = True
            self.__predict__()
            self.update_prediction_threshold = False

    def get_prediction_metrics(self):
        predictions_binary = self.__get_binary_threshold__()

        return sklearn.metrics.classification_report(
            self.test_target, predictions_binary
        )

    def get_confusion_matrix(self):
        predictions_binary = self.__get_binary_threshold__()

        return pd.DataFrame(
            sklearn.metrics.confusion_matrix(self.test_target, predictions_binary)
        )

    def get_percentage_confusion_matrix(self):
        cm = self.get_confusion_matrix()
        col_sums = cm.sum(axis=0)

        cm_percentages = cm.div(col_sums, axis=1) * 100

        return cm_percentages

    def run_lime(self, text):
        c = sklearn.pipeline.make_pipeline(self.vectorizer, self.model)

        self.lime_res = LimeTextExplainer().explain_instance(
            text_instance=text,
            classifier_fn=c.predict_proba,
            num_features=len(self.vectorizer.get_feature_names_out()),
        )

    def plot_lime(self):
        self.lime_res.as_pyplot_figure()
        plt.show()

    def get_lime_html(self):
        return self.lime_res.as_html()

    def get_lime_list(self):
        return self.lime_res.as_list()

    def run_shap(self):
        explainer = shap.Explainer(
            self.model,
            self.train_features,
            feature_names=self.vectorizer.get_feature_names_out(),
        )
        self.shap_explainer = explainer(self.test_features)
        self.shap_values = explainer.shap_values(self.test_features)

    def plot_shap_summary(self):
        return shap.summary_plot(
            self.shap_values,
            self.test_features,
            feature_names=self.vectorizer.get_feature_names_out(),
        )

    def plot_shap_scatter(self, index):
        return shap.plots.scatter(self.shap_explainer[index])
