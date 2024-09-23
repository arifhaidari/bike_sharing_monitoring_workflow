import requests
import zipfile
import io
import pandas as pd
import datetime

# Function to fetch the dataset
def fetch_data():
    """
    Fetches the Bike Sharing dataset from the UCI website.
    Returns a pandas DataFrame containing the raw data.
    return  -> pd.DataFrame
    """
    url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
    response = requests.get(url, verify=False)
    
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            # Read the 'hour.csv' file from the archive
            raw_data = pd.read_csv(archive.open("hour.csv"), parse_dates=['dteday'])
        return raw_data
    else:
        raise Exception(f"Failed to fetch data. Status code: {response.status_code}")

# Function to process the dataset
def process_data(raw_data: pd.DataFrame):
    """
    Processes the raw Bike Sharing dataset and creates a DateTime index based on the 'dteday' and 'hr' columns.
    Returns a pandas DataFrame with the processed data.
    return  -> pd.DataFrame
    """
    # Combine the 'dteday' date and 'hr' hour columns into a single datetime index
    raw_data['datetime'] = raw_data.apply(lambda row: datetime.datetime.combine(row['dteday'].date(), datetime.time(row['hr'])), axis=1)
    
    # Set the 'datetime' as the index
    raw_data.set_index('datetime', inplace=True)
    
    # Drop the 'dteday' and 'hr' columns as they are no longer needed
    processed_data = raw_data.drop(columns=['dteday', 'hr'])
    
    return processed_data

# Main function to execute data ingestion and processing
def load_bike_sharing_data():
    """
    Orchestrates the fetching and processing of the Bike Sharing dataset.
    Returns a pandas DataFrame with the processed data.
    return  -> pd.DataFrame
    """
    print("Fetching the dataset...")
    raw_data = fetch_data()
    
    print("Processing the dataset...")
    processed_data = process_data(raw_data)
    
    print("Data ingestion and processing completed.")
    return processed_data

if __name__ == "__main__":
    data = load_bike_sharing_data()
    print("Sample of the processed data:")
    print(data.head())
