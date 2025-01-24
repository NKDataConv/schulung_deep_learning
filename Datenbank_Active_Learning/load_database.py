from datasets import load_dataset
import sqlite3
import sys
from typing import List, Dict


# Constants
DATABASE_NAME = "rotten_tomatoes.db"
DATASET_NAME = "rotten_tomatoes"
BATCH_SIZE = 1000

def create_database_schema(cursor: sqlite3.Cursor) -> None:
    """Create the database schema for the Rotten Tomatoes dataset."""
    try:
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            label INTEGER NOT NULL
        )
        ''')
    except sqlite3.Error as e:
        print(f"Error creating database schema: {str(e)}")
        sys.exit(1)

def load_huggingface_dataset() -> Dict:
    """Load the dataset from Hugging Face."""
    try:
        print("Loading Rotten Tomatoes dataset from Hugging Face...")
        dataset = load_dataset(DATASET_NAME)
        return dataset
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {str(e)}")
        sys.exit(1)

def insert_data_batch(cursor: sqlite3.Cursor, data: List[Dict]) -> None:
    """Insert a batch of data into the database."""
    try:
        cursor.executemany(
            "INSERT INTO reviews (text, label) VALUES (?, ?)",
            [(item['text'], item['label']) for item in data]
        )
    except sqlite3.Error as e:
        print(f"Error inserting batch into database: {str(e)}")
        raise

def get_database_connection():
    """Get a connection to the database."""
    try:
        return sqlite3.connect(DATABASE_NAME)
    except sqlite3.Error as e:
        print(f"Error connecting to database: {str(e)}")
        raise

def close_database_connection(conn: sqlite3.Connection):
    """Safely close the database connection."""
    try:
        if conn:
            conn.close()
    except sqlite3.Error as e:
        print(f"Error closing database connection: {str(e)}")
        raise

def main():
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        
        # Create schema
        create_database_schema(cursor)
        
        # Load dataset
        dataset = load_huggingface_dataset()
        
        # Process each split (train, validation, test)
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            total_items = len(split_data)
            total_batches = total_items // BATCH_SIZE + (1 if total_items % BATCH_SIZE else 0)
            
            print(f"\nProcessing {split_name} split ({total_items} items)...")
            
            # Process data in batches
            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min((batch_idx + 1) * BATCH_SIZE, total_items)
                
                # Convert dataset items to list of dictionaries
                batch_data = []
                for idx in range(start_idx, end_idx):
                    item = split_data[idx]
                    batch_data.append({
                        'text': item['text'],
                        'label': item['label']
                    })
                
                insert_data_batch(cursor, batch_data)
                conn.commit()
                
                # Print progress
                progress = (batch_idx + 1) / total_batches * 100
                print(f"\rProgress: {progress:.1f}%", end="", flush=True)
            
            print()  # New line after progress bar

        print("\nDatabase creation completed successfully!")
        
        # Print statistics
        cursor.execute("SELECT COUNT(*) FROM reviews")
        total_reviews = cursor.fetchone()[0]
        print(f"Total reviews in database: {total_reviews}")
        
        cursor.execute("SELECT label, COUNT(*) FROM reviews GROUP BY label")
        label_distribution = cursor.fetchall()
        print("\nLabel distribution:")
        for label, count in label_distribution:
            print(f"Label {label}: {count} reviews")
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)
        
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main() 