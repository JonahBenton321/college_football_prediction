# This function takes a data frame and groups it by all the team names than the features for each team are replaced with rolling averages
# over a window of 5 games shifted by one to prevent target leakage

def rolling_average(dframe):
    window_size = 5
    for col in dframe:
        if col not in ['Name', 'Date']:
            dframe[col] = dframe.groupby('Name')[col].transform(lambda x: x.ewm(span=window_size, adjust=False ).mean())
            dframe[col] = dframe.groupby('Name')[col].shift(1)

    return  dframe


