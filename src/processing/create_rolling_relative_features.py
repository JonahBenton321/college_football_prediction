import pandas as pd
from rolling_average import rolling_average

# Reads data from the years 2021-25 into a single data frame called df
df =  pd.read_csv('Data/raw/games_data_2021_26.csv')

output_dir = "Data"

# Create directory structure if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
# The NAIA Average statistic is often empty on the NAIA Stats website
# This code replaces the nan values with the mean of the average column
df = df.fillna(df['Average'].mean())

# Converts team data to rolling averages over the last 5 games
df_rolling = df.copy()
rolling_average(df_rolling)
df_rolling.dropna()

# The Score will be used to create the target variable which the model predicts
# Score is split into even and odd index so the odd index can be subtracted from the even indexes. A negative result is converted to 1 and positive to 0
y_even = df['Score'].iloc[::2].reset_index(drop=True)
y_odd = df['Score'].iloc[1::2].reset_index(drop=True)

# splits the data into even and odd indexes which allow team two's feature to be subtracted from teams one's features creating relative features
df_even_rows = df_rolling.iloc[::2].reset_index(drop=True)
df_odd_rows = df_rolling.iloc[1::2].reset_index(drop=True)

# The rolling_average function creates nan values which need to be removed
# In some cases the team ones feature will be replaced with nan values but team twos features are not or vice versa
# The following code insures only compete games are present in the data set by dropping nan values from the data frames and their accompanying indexes from the opposite data frame
# All indexes dropped are also dropped from the y data frames

# Finds nan indexes
even_indices_to_remove = df_even_rows[df_even_rows.isna().any(axis=1)].index
odd_indices_to_remove = df_odd_rows[df_odd_rows.isna().any(axis=1)].index

# Removes even nan values from old rows and visa versa
df_even_rows = df_even_rows.drop(index=odd_indices_to_remove)
df_odd_rows = df_odd_rows.drop(index=even_indices_to_remove)

# Remove nan values from y data
y_even = y_even.drop(index=odd_indices_to_remove)
y_odd = y_odd.drop(index=even_indices_to_remove)

# Finds remaining nan values
even_indices_to_remove = df_even_rows[df_even_rows.isna().any(axis=1)].index
odd_indices_to_remove = df_odd_rows[df_odd_rows.isna().any(axis=1)].index

# Removes remaining nan values from y data
y_even = y_even.drop(index=even_indices_to_remove)
y_odd = y_odd.drop(index=odd_indices_to_remove)

# Removes even nan values from even row and visa versa
df_even_rows =df_even_rows.dropna()
df_odd_rows = df_odd_rows.dropna()

# Removes now unnecessary data and name columns
df_even_rows = df_even_rows.drop(columns=['Name', 'Date'])
df_odd_rows = df_odd_rows.drop(columns=['Name', 'Date'])

# Creates relative features by subtracting odd rows from even rows
relative_features = df_even_rows - df_odd_rows
y_relative = y_even - y_odd

# Converts y_relative to 1 or 0 based on weather the index is negative
def convert_relative_score_to_categorical(x):
    if x <=0:
        return 1
    else:
        return 0

y = y_relative.apply(convert_relative_score_to_categorical)

relative_features['Target Data'] = y

# Rounds relative_features to simply data
for col in relative_features:
    relative_features[col] = relative_features[col].apply(lambda x: round(x, 4))

# Removes any remaining nan values
relative_features = relative_features.dropna()

# Saves data frames to csv
relative_features.to_csv(os.path.join(output_dir, 'processed_game_data_2021-26.csv'), index=False)










