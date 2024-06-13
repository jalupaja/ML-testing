#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(filepath):
    return pd.read_csv(filepath, sep=",", encoding="utf-8", encoding_errors="replace")


def use_integers():
    plt.xticks(list(range(-1000, 1000)))


def plot_bar(df):
    x = np.array(df.index)
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


def get_distribution(col):
    # TODO try return col.value_counts()
    return col.groupby(col).count()


def show_distribution(col):
    plot_bar(get_distribution(col))
    # TODO build nicer plot: axes, colors, ...
    plt.show()


# setup bigger plot fontsize
font = {"size": 22}
plt.rc("font", **font)


filepath = "../data/McDonald_s_Reviews.csv"

print("Loading data:")
df = load_data(filepath)

# format rating as integer
df = df.assign(rating=df.rating.astype(str).str[0].astype(int))

# RATING per store
df_gr_store = df.groupby(by="store_address")

print(f"average rating: {np.mean(df.rating)}")

average_rating_per_store = (
    pd.merge(df_gr_store.rating.mean(), df_gr_store.rating.count(), on="store_address")
    .query("rating_y >= 10")
    .sort_values("rating_x", ascending=False)
)
# TODO map this using size, color for amount_ratings, rating
map_rating_per_store = (
    pd.merge(
        average_rating_per_store,
        df[["store_address", "longitude", "latitude "]],
        how="left",
        on="store_address",
    )
    .drop(columns=["store_address"])
    .drop_duplicates()
)

# most common RATING
amount_ratings = df.groupby(by="review").review.count()
amount_ratings = pd.DataFrame(
    {"review": amount_ratings.keys(), "n": amount_ratings.values}
).sort_values("n", ascending=False)

# rating per common review
rating_per_review = (
    pd.merge(
        amount_ratings.iloc[0:20], df[["review", "rating"]], how="left", on="review"
    )
    .groupby("review")
    .rating.mean()
)


test = (
    amount_ratings.iloc[0:5]
    .assign(review="'" + amount_ratings.review + "'")
    .groupby(by="review")
    .n.max()
    .sort_values(ascending=False)
)

# plot_bar(
#     (
#         amount_ratings.iloc[0:5]
#         .assign(review="'" + amount_ratings.review + "'")
#         .groupby(by="review")
#         .n.max()
#         .sort_values(ascending=False)
#     )
# )
# plt.show()


def show_dist_of_review(review):
    use_integers()
    show_distribution(df[df.review == review].rating)


# show_distribution(df.rating)
# show_dist_of_review("Ok")
# show_dist_of_review("Nice")
# show_dist_of_review("Good")
# show_dist_of_review("It's McDonald's")
