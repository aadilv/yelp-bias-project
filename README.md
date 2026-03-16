# Detecting Systemic Bias in Hospitality Reviews (Yelp Open Dataset)

NLP pipeline that flags potential bias and discrimination in Yelp reviews across hospitality businesses (restaurants, bars, hotels, spas, cafes) in Edmonton, Nashville, and New Orleans. Built in the last two weeks of August 2025 as a technical assessment for an internship. The company gave me full ownership of the work, so here it is.

## What it does

Takes 1M+ Yelp reviews, filters to hospitality businesses, and tries to distinguish reviews describing actual discriminatory experiences from reviews that are just generally negative. The core idea is a dual-condition flag so a review only counts as "bias-flagged" if it contains a bias-related keyword AND has negative VADER sentiment. Just having one or the other isn't enough since people complain about cold food (negative, not bias) and people mention race/gender in perfectly neutral contexts (keyword, not bias).

From there, KMeans clustering on geographic coordinates groups flagged reviews into neighborhoods, with silhouette scores picking the cluster count. Folium heatmaps show where bias-flagged reviews concentrate across 10,900+ businesses in the three cities.

## How it works

1. Filter Yelp business data to hospitality categories
2. Preprocess review text (lowercase, tokenize, stopword removal)
3. Flag reviews containing bias-related keywords (regex patterns for race, gender, disability, religion, etc.)
4. Run VADER sentiment on all reviews
5. Dual-condition filter: negative sentiment + bias keyword = bias-flagged
6. KMeans clustering on lat/long of flagged businesses with dynamic k via silhouette optimization
7. Neighborhood-level analysis and Folium heatmap visualization

I did used ChatGPT for the regex patterns/matching because writing 50+ bias-detection regex expressions by hand sounded about as fun as having Vecna from Stranger Things gouge my eyeballs out.

## Tech

Python (pandas, numpy), NLTK + VADER, scikit-learn (KMeans), Matplotlib/Seaborn, Folium

## Running it

1. Clone the repo
2. Download the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) and drop `business.json` and `review.json` in the project folder
   - Note: Yelp periodically updates this dataset and rotates which cities are included, so results may not be exactly reproducible with a newer version. This was built on the version available in August 2025
4. Run the Jupyter notebook
