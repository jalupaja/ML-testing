#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim
import wordcloud
from lime.lime_text import LimeTextExplainer
import shap
import sklearn

from MLPipeline import Pipeline
from Word2Vec import Word2VecTransformer

## PLOTS
def plot_bar(df):
    df = df.rank()
    x = np.array(df.keys())  # TODO maybe df.index?
    y = np.array(df.values)
    plt.bar(x, y)


def plot_default(df):
    plt.plot(df)


def create_wordcloud(col):
    # round mask
    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130**2
    mask = 255 * mask.astype(int)

    # TODO improve (a lot) or use orange
    return wordcloud.WordCloud(
        height=1000,
        width=1000,
        mask=mask,
        background_color="white",
        contour_width=3,
        contour_color="black",
    ).generate(str(col))


def show_wordcloud(col):
    wc = create_wordcloud(col)
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def shap_plot(shap_values):
    shap.plots.beeswarm(shap_values)


## FILE FUNCTIONS
def load_data(filepath):
    return pd.read_csv(filepath, sep=",", encoding="utf-8", encoding_errors="replace")


def remove_unreadable_text(col):
    return [text.replace("�", "") for text in col]


# VECTORIZERS
def vectorize_bow():
    return sklearn.feature_extraction.text.CountVectorizer()


def vectorize_tfidf():
    return sklearn.feature_extraction.text.TfidfVectorizer(min_df=10)


def vectorize_w2v():
    return Word2VecTransformer()


# MODELS
def linear_regression():
    return sklearn.linear_model.LinearRegression()


def logistic_regression():
    return sklearn.linear_model.LogisticRegression()


def decision_tree():
    return sklearn.tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=20)


filepath = "../data/McDonald_s_Reviews.csv"

print("Loading data:")
df = load_data(filepath)

# Fix columns
print("Fixing and Removing columns:")
df = (
    df.assign(rating=df.rating.astype(str).str[0].astype(int))
    .assign(review=remove_unreadable_text(df.review))
    .assign(rating_count=df.rating_count.astype(str).str.replace(",", "").astype(int))
    .drop(columns=["store_name", "category", "latitude ", "longitude"])
)

df = df.assign(target=(df.rating >= 4).astype(int))

print()

# TODO
df = df[0:1000]

# TODO maybe still have to remove stopwords?
pipeline = Pipeline()
pipeline.set_data(df.review, df.target, 0.2)
pipeline.set_vectorizer(vectorize_w2v())
pipeline.set_model(decision_tree())
pipeline.run()

print(pipeline.get_prediction_metrics())
print(pipeline.get_percentage_confusion_matrix())

# pipeline.run_shap()
# pipeline.get_shap_plot_scatter(1)
# pipeline.plot_shap_summary()

# pipeline.run_lime("I'd rather not eat here but it is the best around")
# pipeline.plot_lime()

# TODO distributions, ...
