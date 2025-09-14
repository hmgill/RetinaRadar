from pathlib import Path
import pandas as pd
from typing import Optional

def extract_source_from_filename(filename: str) -> str:
    """
    Extract source identifier from CSV filename by removing the last underscore-separated part.
    
    Args:
        filename: The CSV filename 
        
    Returns:
        str: The source identifier (everything before the last underscore)
        
    Examples:
        'train_labels.csv' -> 'train'
        'test_val_labels.csv' -> 'test_val'
        'data_preprocessing_results.csv' -> 'data_preprocessing'
        'model_v2_final_outputs.csv' -> 'model_v2_final'
        'simple.csv' -> 'simple' (no underscore, returns whole stem)
    """
    # Remove the file extension first
    stem = Path(filename).stem
    
    # Split by underscore and take all parts except the last one
    parts = stem.split('_')
    
    if len(parts) <= 1:
        # No underscore found, return the whole stem
        return stem
    
    # Join all parts except the last one
    return '_'.join(parts[:-1])

def clean_name_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove file extensions from the 'name' column using pathlib.
    
    Args:
        df: DataFrame with a 'name' column that may contain filenames with extensions
        
    Returns:
        pd.DataFrame: DataFrame with cleaned 'name' column (extensions removed)
    """
    if 'name' in df.columns:
        # Apply pathlib.Path.stem to remove extensions
        df = df.copy()  # Avoid modifying the original dataframe
        df['name'] = df['name'].apply(lambda x: Path(str(x)).stem if pd.notna(x) else x)
    
    return df

def merge_csvs(folder_path: str | Path) -> pd.DataFrame:
    """
    Merge all CSV files from a folder with normalized headers, source tracking, and optional name indexing.
    
    Args:
        folder_path: Path to the folder containing CSV files (string or Path object)
        
    Returns:
        pd.DataFrame: The merged dataframe with lowercase headers, 'source' column, and 'name' index if present
        
    Raises:
        FileNotFoundError: If the folder doesn't exist
        ValueError: If no CSV files are found in the folder
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    # Get all CSV files in the folder
    csv_files = list(folder.glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in folder: {folder}")
    
    # Read and process each CSV file
    dataframes: list[pd.DataFrame] = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower()  # Lowercase headers
        
        # Clean the name column if it exists (remove extensions)
        df = clean_name_column(df)
        
        # Add source column based on filename
        source = extract_source_from_filename(file_path.name)
        df['source'] = source
        
        dataframes.append(df)
        print(f"Loaded: {file_path.name} -> source: '{source}' ({len(df)} rows)")
    
    # Concatenate all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True, sort=False)
    
    # Set 'name' as index if present (after cleaning and adding source column)
    if 'name' in merged_df.columns:
        merged_df = merged_df.set_index('name')
        print("Set cleaned 'name' column as index (extensions removed).")
    
    # Save to merged_labels.csv
    output_path = folder / "merged_labels.csv"
    merged_df.to_csv(output_path)
    
    print(f"Merged {len(csv_files)} files into {len(merged_df)} rows")
    print(f"Saved to: {output_path}")
    
    return merged_df

def get_dataframe_info(df: pd.DataFrame) -> dict[str, int]:
    """
    Get basic information about a dataframe.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        dict: Dictionary containing 'rows' and 'columns' counts
    """
    return {
        'rows': len(df),
        'columns': len(df.columns)
    }

def get_source_distribution(df: pd.DataFrame) -> dict[str, int]:
    """
    Get the distribution of rows by source.
    
    Args:
        df: The pandas DataFrame with a 'source' column
        
    Returns:
        dict: Dictionary mapping source names to row counts
    """
    if 'source' not in df.columns and 'source' not in df.index.names:
        # Check if source is in the columns (not index)
        if hasattr(df, 'reset_index'):
            temp_df = df.reset_index()
            if 'source' in temp_df.columns:
                return temp_df['source'].value_counts().to_dict()
        return {}
    
    if 'source' in df.columns:
        return df['source'].value_counts().to_dict()
    else:
        # If df is indexed by name, source info might be lost, 
        # but this shouldn't happen with our implementation
        return {}

def process_folder_csvs(folder_path: str | Path, verbose: bool = True) -> tuple[pd.DataFrame, int]:
    """
    Complete workflow to merge CSV files and return the result with row count.
    
    Args:
        folder_path: Path to the folder containing CSV files
        verbose: Whether to print detailed information
        
    Returns:
        tuple: (merged_dataframe, row_count)
    """
    try:
        merged_df = merge_csvs(folder_path)
        row_count = len(merged_df)
        
        if verbose:
            info = get_dataframe_info(merged_df)
            print(f"\nFinal dataframe: {info['rows']} rows, {info['columns']} columns")
            print(f"Columns: {list(merged_df.columns)}")
            
            # Show a sample of the index if it's the name column
            if merged_df.index.name == 'name':
                sample_names = merged_df.index[:5].tolist()
                print(f"Sample names (extensions cleaned): {sample_names}")
            
            # Show source distribution
            if 'source' in merged_df.columns:
                source_dist = get_source_distribution(merged_df)
                print(f"Source distribution: {source_dist}")
            elif hasattr(merged_df, 'reset_index'):
                # If indexed by name, temporarily reset to show source distribution
                temp_df = merged_df.reset_index()
                if 'source' in temp_df.columns:
                    source_dist = temp_df['source'].value_counts().to_dict()
                    print(f"Source distribution: {source_dist}")
        
        return merged_df, row_count
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        raise

# Usage example
if __name__ == "__main__":
    # Example usage
    folder_path = Path("labels")  # Change this to your folder
    
    try:
        merged_data, total_rows = process_folder_csvs(folder_path)
        print(f"\nProcess completed successfully!")
        print(f"Total rows in merged data: {total_rows}")
        
        # Example: Access data by source
        if 'source' in merged_data.columns:
            print(f"\nUnique sources: {merged_data['source'].unique().tolist()}")
        elif hasattr(merged_data, 'reset_index'):
            temp_df = merged_data.reset_index()
            if 'source' in temp_df.columns:
                print(f"\nUnique sources: {temp_df['source'].unique().tolist()}")
        
    except Exception as e:
        print(f"Failed to process CSV files: {e}")
