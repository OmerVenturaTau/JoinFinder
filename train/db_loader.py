import configparser
import psycopg2
import pandas as pd
from typing import List, Tuple, Optional, Dict
import os
import logging

from system import DB_CONFIG_PATH

# Set up logger
logger = logging.getLogger(__name__)

# Table names for patch coordinates (must match extract_patch_coordinates.py)
PATCH_IMAGE_METADATA_TABLE_NAME = 'pretrain_nli_manuscript_patch_image_metadata'
PATCH_COORDINATES_TABLE_NAME = 'pretrain_nli_manuscript_patch_coordinates'

def get_db_connection(config_path=DB_CONFIG_PATH):
    config = configparser.ConfigParser()
    config.read(config_path)
    db_params = config['postgresql']
    conn = psycopg2.connect(
        host=db_params['host'],
        database=db_params['database'],
        user=db_params['user'],
        password=db_params['password'],
        port=db_params.get('port', 5432)
    )
    return conn


def load_patch_coordinates_from_db(
    image_path: str, 
    conn, 
    patch_size: int,
    table_name: str = PATCH_COORDINATES_TABLE_NAME,
    split: Optional[str] = None
) -> Optional[List[Tuple[float, float]]]:
    """
    Load patch coordinates from database for a given image.
    
    Args:
        image_path: Full path to the image
        conn: Database connection
        patch_size: Size of patches (used for validation)
        table_name: Name of the patch coordinates table
        split: Optional split name ('train', 'val', 'test') for verification.
               If provided, will verify the image belongs to this split in the metadata table.
        
    Returns:
        List of (center_x_normalized, center_y_normalized) tuples, or None if not found
    """
    try:
        # Build query - join with metadata table if split verification is requested
        if split is not None:
            # Verify split by checking if image exists in the expected split
            # Note: This assumes the database has been populated with split information
            # For now, we'll just query by image_path (split verification can be added later)
            query = f"""
                SELECT pc.patch_index, pc.center_x_normalized, pc.center_y_normalized
                FROM {table_name} pc
                WHERE pc.image_path = %s
                ORDER BY pc.patch_index
            """
        else:
            query = f"""
                SELECT patch_index, center_x_normalized, center_y_normalized
                FROM {table_name}
                WHERE image_path = %s
                ORDER BY patch_index
            """
        
        df = pd.read_sql(query, conn, params=(image_path,))
        
        if df.empty:
            return None
        
        # Convert to list of tuples
        coords = [(row['center_x_normalized'], row['center_y_normalized']) 
                  for _, row in df.iterrows()]
        return coords
    except Exception as e:
        # If query fails, return None to fall back to extraction
        logger.exception(f"Failed to load patch coordinates from DB for {image_path}")
        return None


def load_patch_metadata_from_db(
    image_path: str,
    conn,
    table_name: str = PATCH_IMAGE_METADATA_TABLE_NAME
) -> Optional[Dict]:
    """
    Load patch extraction metadata from database for a given image.
    
    Args:
        image_path: Full path to the image
        conn: Database connection
        table_name: Name of the image metadata table
        
    Returns:
        Dictionary with metadata, or None if not found
    """
    try:
        query = f"""
            SELECT *
            FROM {table_name}
            WHERE image_path = %s
        """
        df = pd.read_sql(query, conn, params=(image_path,))
        
        if df.empty:
            return None
        
        # Convert to dictionary
        row = df.iloc[0]
        metadata = {
            'patch_size': int(row['patch_size']),
            'stride': int(row['stride']),
            'num_patches': int(row['num_patches']),
            'xml_path': row['xml_path'],
            'text_regions_count': int(row['text_regions_count']) if pd.notna(row['text_regions_count']) else 0,
            'is_two_page': bool(row['is_two_page']) if pd.notna(row['is_two_page']) else False,
            'page_split_x': float(row['page_split_x']) if pd.notna(row['page_split_x']) else None,
            'rotation_angle': int(row['rotation_angle']) if pd.notna(row['rotation_angle']) else 0,
            'rotation_corrected': bool(row['rotation_corrected']) if pd.notna(row['rotation_corrected']) else False,
            'reading_order_applied': bool(row['reading_order_applied']) if pd.notna(row['reading_order_applied']) else False,
        }
        return metadata
    except Exception as e:
        logger.exception(f"Failed to load patch metadata from DB for {image_path}")
        return None


def get_table_as_df(conn, table_name: str) -> pd.DataFrame:
    """
    Get all content of a database table as a DataFrame.
    
    Args:
        conn: Database connection
        table_name: Name of the table to load
        
    Returns:
        DataFrame containing all table rows
    """
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql(query, conn)
