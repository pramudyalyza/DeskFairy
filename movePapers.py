import os
import shutil
import logging
import pandas as pd

def main():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
    INPUT_CSV_PATH = "data/paper_data.csv"

    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        logging.info(f"Successfully loaded data from {INPUT_CSV_PATH}. Found {len(df)} entries")
    except FileNotFoundError:
        logging.error(f"Error: {INPUT_CSV_PATH} not found. Ensure 1_extract_cluster_name.py ran successfully")
        exit()
    except pd.errors.EmptyDataError:
        logging.info(f"No data to process in {INPUT_CSV_PATH}. It might be empty or only headers")
        exit()
    except Exception as e:
        logging.error(f"An error occurred while reading {INPUT_CSV_PATH}: {e}")
        exit()

    if df.empty:
        logging.info("DataFrame is empty. No papers to organize")
        exit()

    for index, row in df.iterrows():
        file_name = row['fileName']
        cluster_name = row['clusterName']

        source_path = os.path.join(DESKTOP_PATH, file_name)
            
        sanitized_cluster_name = "".join(x for x in cluster_name if x.isalnum() or x in " -_").strip()
        if not sanitized_cluster_name:
            sanitized_cluster_name = "Uncategorized_Papers"
            logging.warning(f"Cluster name '{cluster_name}' resulted in an empty sanitized name. Using '{sanitized_cluster_name}'")

        destination_folder = os.path.join(DESKTOP_PATH, sanitized_cluster_name)
        destination_path = os.path.join(destination_folder, file_name)

        os.makedirs(destination_folder, exist_ok=True)

        if os.path.exists(source_path):
            if not os.path.exists(destination_path):
                try:
                    shutil.move(source_path, destination_path)
                    logging.info(f"Moved '{file_name}' to '{destination_folder}'")
                except Exception as e:
                    logging.error(f"Error moving '{file_name}' to '{destination_folder}': {e}")
            else:
                logging.warning(f"'{file_name}' already exists in '{destination_folder}'. Skipping move to avoid overwrite")
        else:
            logging.warning(f"Source file '{file_name}' not found on Desktop. Skipping this file")

    logging.info("Processing complete for 2-movePapers.py")

if __name__ == "__main__":
    main()