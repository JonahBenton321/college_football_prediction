import asyncio
from playwright.async_api import async_playwright
import numpy as np
import pandas as pd
import re
import os

# Loads link to collage football games to be scraped
links = np.load('Data/links/links_to_games.npy')

output_dir = "Data/raw"

# Create directory structure if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

total_len = len(links)
# Columns often contain data the form '7--8', '7-8', '7 -8' among other variations
# This function converts all instances to the form ('7', '8')
def parse_stings(s):
    nums = re.findall(r"\d+(?:\.\d+)?", str(s))
    if len(nums) == 2:
        return nums[0], nums[1]
    elif len(nums) == 3:
        return nums[0]
    elif len(nums) == 6:
        return nums[0], nums[1], nums[2], nums[3], nums[4], nums[5]
    else:
        return s

# The average feature is often scraped in the form '30-5' or some other variation rather than an actual average
# This function replace instances of '30-5' with its average '6'
def parse_average(s):
    nums = re.findall(r"\d+(?:\.\d+)?", str(s))
    if len(nums) == 2:
        return int(nums[0])/int(nums[1])
    elif len(nums) > 2:
        return None
    return s

# Converts stringS of the form 'name score' such as 'Fort Lauderdale 32' to a tuple ('Fort Lauderdale', 32')
def parse_name_score(s):
    name = re.findall(r"^(.*?)\s+\d{1,2}", str(s))
    score = re.findall(r"\d{1,2}", str(s))
    return name[0], score[0]

async def scrape_urls():
    # Names of all the features to be scraped
    # The order of the data is consistent on the NAIA website and so column names and order is defined here
    expected_column_order = ['Date', 'Name', 'Score', 'FIRST DOWNS', 'THIRD DOWN EFFICIENCY', 'FOURTH DOWN EFFICIENCY',
               'TOTAL OFFENSE', 'NET YARDS PASSING', 'completionAttemptsNumber', 'completionAttemptsYards', 'NetYards',
               'SackedNumber', 'SackedYards', 'intercepted', 'NET YARDS RUSHING', 'Rushing Attempts',
               'Average gain per rush', 'PUNTS: Number', 'PUNTS: Yards', 'Average', 'TOTAL RETURN YARDS',
               'PENALTIES: Number', 'PENALTIES: Yards', 'FUMBLES: Number', 'FUMBLES: Lost', 'SACKS: Number',
               'SACKS: Yards', 'INTERCEPTIONS: Number', 'INTERCEPTIONS: Yards', 'TIME OF POSSESSION']

    # Instantiation of the data frame to be filled scraped values
    df_master = pd.DataFrame(columns=expected_column_order)

    # Use of async_playwright to scrape web pages
    async with async_playwright() as p:
        current_index=0

        # Loops over every url to be scraped
        for url in links:
            # Tracks the current url being scraped
            current_index+=1
            try:
                # Establishes the browser and context
                browser = await p.firefox.launch()
                page = await browser.new_page()
                context = await browser.new_context()

                # Connects to web page and waits for it to load
                await page.goto('https://naiastats.prestosports.com'+url)
                await page.wait_for_load_state("load")

                # Pulls the html and collects all the table data
                html = await page.content()
                all_td = await page.query_selector_all('td')

                # Converts html data to pandas data frames
                list_of_df = pd.read_html(html)

                # The index of 3 and 0 store the tables which contain relevant data
                df_major_features = list_of_df[3]
                df_score_name_date = list_of_df[0]

                # Pulls the date feature
                df_score_name_date['Date'] = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", df_score_name_date[0].iloc[0]).group(1)

                # The name and score feature are store left to right in single row
                # This code converts the row which stores the name and date to a vertical column and adds the data to the column 'Name/Score'
                df_score_name_date['Name/Score'] = df_score_name_date.iloc[1, :-1].T

                # Converts the string 'name score' to a tuple ('name', 'score')
                df_score_name_date['Name/Score'] = df_score_name_date['Name/Score'].apply(parse_name_score)

                # Converts the column 'Name/Score' to two columns 'Name' 'Score'
                name_score = pd.DataFrame(df_score_name_date['Name/Score'].to_list(), columns=['Name', 'Score'])
                df_score_name_date = pd.concat([df_score_name_date, name_score], axis=1)

                # This also drops columns 0, and 1 which contain extra relevant data pulled from the html
                df_score_name_date = df_score_name_date.drop([0, 1, 'Name/Score'], axis=1)

                # Sets the 'statistics' column to be the index
                df_major_features = df_major_features.set_index(df_major_features.columns[1])

                # Changes the name of the columns to 'team_1' and 'team_2' for ease of reference
                df_major_features.columns=['team_1', 'team_2']

                # Converts strings like '7-3' into tuples such as (7, 3)
                df_major_features['team_1'] = df_major_features['team_1'].apply(parse_stings)
                df_major_features['team_2'] = df_major_features['team_2'].apply(parse_stings)

                # Drops data unnecessary for model training
                # The dropped data consists of sub statistics which breaks data like 'Total Offensive Plays' into multiple subcategories
                # These subcategories are unnecessary for module training and dropped here as a result
                df_major_features = df_major_features.drop(['Punt Returns: Number-Yards  Kickoff Returns:  Number-Yards Interception Returns: Number-Yards'])
                df_major_features = df_major_features.drop(['Passing Rushing Penalty'])
                df_major_features = df_major_features.drop(['Total Offensive Plays Average gain per play'])

                # Transposes the data frame
                df_major_features = df_major_features.T

                # Converts the 'TIME OF POSSESSION' from (minutes, seconds) to the game length in seconds
                df_major_features['TIME OF POSSESSION'] = df_major_features['TIME OF POSSESSION'].apply(lambda x: int(x[0]) * 60 + int(x[1]))

                # Converts all the columns which contain tuples into multiple columns one per each element of the tuple
                for col in df_major_features.columns:
                    if df_major_features[col].apply(lambda x: isinstance(x, tuple)).all():
                        values = df_major_features[col].apply(pd.Series)
                        col_index = df_major_features.columns.get_loc(col)
                        df_before = df_major_features.iloc[:, :col_index]
                        df_after = df_major_features.iloc[:, col_index+1:]
                        df_major_features = pd.concat([df_before, values, df_after], axis=1)

                # Rests the index to be 0, 1 ... for ease of reference
                df_major_features = df_major_features.reset_index(drop=True)

                # Creates a data frame containing all features needed for modeling
                df_complete_features = pd.concat([df_score_name_date, df_major_features], axis=1)

                # Set the name of the columns to the expected order
                df_complete_features.columns = expected_column_order

                # Adds the complete features to a master data frame
                df_master = pd.concat([df_master, df_complete_features], ignore_index=True)

                print(f'scrape {current_index} of {total_len} successful')

            except Exception as e:

                print(f'scraping url {url} failed {e}')

            # close context and browser
            await context.close()
            await browser.close()
        # Saves to CSV
        df_master.to_csv(os.path.join(output_dir, 'games_data_2021_26.csv'), index=False)


# Runs async scraper
asyncio.run(scrape_urls())
