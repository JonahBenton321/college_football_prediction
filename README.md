## Project overview
The goal of this project is to use historical data from competing football team’s previous five games to predict the winner of a match up. This project employs an end-to-end data collection and refinement pipeline to scrape data from the NAIA's official website and train a model to predict match-up outcomes. This project was developed as a portfolio piece to demonstrate data collection, processing, feature engineering, and modeling skills

## Project Structure
    +---Data
    |   +---links
    |   |       links_to_games.npy
    |   |       
    |   +---processed
    |   |       processed_game_data_2021-26.csv
    |   |       
    |   \---raw
    +---models
    |       final_model.joblib
    |       
    +---src
    |   |   
    |   +---modeling
    |   |       train_model.py
    |   |       
    |   +---processing
    |   |   |   create_rolling_relative_features.py
    |   |   |   rolling_average.py
    |   |   |
    |   \---scraping
    |           scrape_games.py
    |           scrape_schedule.py
    +---README.MD
    +---requirements.txt

## Data sources
All data for the project stems from https://naiastats.prestosports.com/sports/fball/2025-26/schedule which hosts box scores for every football game played within the NAIA division from 2021 through 2025. 

Data collection details:
+ Data scraped using Playwright
+ Approximately 30 team-level features collected per game
+ Features include:
  + Team name
  + Final score
  + Total yards
  + Offensive yards
  + Rushing yards
  + Additional relevant game statistics

Scraped data is standardized and stored in CSV format for future processing and modeling.

## Data Processing & Feature Engineering
The project uses custom python scripts to ensure all scraped data is standardized. Data on the NAIA’s official website is often stored inconsistently, so regular expression matching is used to reliably extract relevant data across pages.

Key processing steps:
+ Grouping data by team
+ Computing an exponentially weighted rolling average over the previous five games for evey index
+ Shifting rolling averages by one game to prevent target leakage
+ Dropping rows with null values introduced by rolling windows

**Feature Engineering**\
To simplify modeling, relative features are created:

+ For each match-up, team-one features are subtracted from team-two features
+ Each game is represented as a single row in a panda’s DataFrame
+ The target variable is binary:
  + 1 if team two wins
  + 0 otherwise

The final dataset contains:
+ 28 relative numeric features
+ 1 binary target variable

## Modeling Approach
A Random Forest Classifier is used to predict the outcome of each match-up.
+ Target variable: Game outcome (win/loss)
+ Evaluation metric: Accuracy
+ Validation method: 5-fold cross-validation

The modeling pipeline includes:
1.	Train-test split (80% training, 20% testing)
2.	5-fold cross-validation on the processed data set
3.	Model training on processed features
4.	Performance comparison against a naïve baseline predictor (majority class)
## Results
Model successfully superseded a naïve baseline predictor

| Model                        | Accuracy |
|------------------------------|:--------:|
| Baseline (Majority Class)    |   54%    |
| 5-fold CV Score              |   76%    | 
| Test Score (20% of data)     |   77%    |
| Training Score (80% of data) |   86%    |

## How to Run
Clone the repository:

    git clone https://github.com/JonahBenton321/college_football_prediction.git
    cd college-football-predictor
Install dependencies:

	pip install -r requirements.txt
Run data scraping scripts:

    python src/scraping/scrape_schedule.py
(optional)

    python src/scraping/scrape_games.py
Run data processing script:

	python src/preprocessing/create_rolling_relative_features.py
Run data modeling script:

    python src/modeling/train_model.py
## Technologies Used
+ Python
+ Pandas
+ NumPy
+ scikit-learn
+ Playwright
+ Joblib

## Author
Jonah Benton\
Data Science Major\
Concordia University Nebraska

