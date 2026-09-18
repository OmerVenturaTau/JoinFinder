"""
Extract and store patch coordinates for pretraining dataset.

This script processes all images in the pretraining dataset, extracts patch
coordinates using XML-based extraction, and stores them in a database table.
"""

import os
import sys
from typing import List, Tuple, Dict
from PIL import Image
import torch
from tqdm import tqdm
import psycopg2
from psycopg2.extras import execute_values
import configparser
import logging

# Set up logger
logger = logging.getLogger(__name__)

# ============================================================================
# PATCH COORDINATES DATABASE CONFIGURATION
# ============================================================================
# Table names for storing patch coordinates extracted from pretraining dataset
# - Image metadata table: one row per image with extraction parameters and metadata
# - Patch coordinates table: one row per patch with normalized coordinates
PATCH_IMAGE_METADATA_TABLE_NAME = 'pretrain_nli_manuscript_patch_image_metadata'
PATCH_COORDINATES_TABLE_NAME = 'pretrain_nli_manuscript_patch_coordinates'

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from train.split_data import build_splits
from utilities.VisionModule.xml_patch_extraction import extract_patches_with_xml
from utilities.xml_loader import find_xml_path_pretrain
from system import (
    BASE_DIR,
    DB_CONFIG_PATH,
    TILE_SIZE,
    TILE_STRIDE,
    MAX_TILES_TRAIN
)


def _to_bool(value) -> bool:
    """Convert arbitrary metadata values to strict booleans."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 't', 'yes', 'y')
    return bool(value)


def get_db_connection(config_path=DB_CONFIG_PATH):
    """Get database connection."""
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
    # Ensure autocommit is OFF so we can control commits manually
    conn.autocommit = False
    return conn


def create_tables(conn, image_metadata_table: str, patch_coords_table: str):
    """Create the image metadata and patch coordinates tables if they don't exist."""
    with conn.cursor() as cur:
        # Image metadata table: one row per image
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {image_metadata_table} (
                id SERIAL PRIMARY KEY,
                image_path TEXT NOT NULL UNIQUE,
                manuscript_id TEXT NOT NULL,
                patch_size INTEGER NOT NULL,
                stride INTEGER NOT NULL,
                num_patches INTEGER NOT NULL,
                xml_path TEXT,
                text_regions_count INTEGER,
                is_two_page BOOLEAN,
                page_split_x REAL,
                rotation_angle INTEGER,
                rotation_corrected BOOLEAN,
                reading_order_applied BOOLEAN,
                image_width INTEGER,
                image_height INTEGER,
                original_image_width INTEGER,
                original_image_height INTEGER,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_{image_metadata_table}_image_path ON {image_metadata_table}(image_path);
            CREATE INDEX IF NOT EXISTS idx_{image_metadata_table}_manuscript_id ON {image_metadata_table}(manuscript_id);
        """)
        
        # Patch coordinates table: one row per patch
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {patch_coords_table} (
                id SERIAL PRIMARY KEY,
                image_path TEXT NOT NULL,
                patch_index INTEGER NOT NULL,
                center_x_normalized REAL NOT NULL,
                center_y_normalized REAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                FOREIGN KEY (image_path) REFERENCES {image_metadata_table}(image_path) ON DELETE CASCADE,
                UNIQUE(image_path, patch_index)
            );
            
            CREATE INDEX IF NOT EXISTS idx_{patch_coords_table}_image_path ON {patch_coords_table}(image_path);
            CREATE INDEX IF NOT EXISTS idx_{patch_coords_table}_patch_index ON {patch_coords_table}(image_path, patch_index);
        """)
        conn.commit()


def extract_manuscript_id_from_path(image_path: str) -> str:
    """Extract manuscript_id from image path."""
    # Path format: /base_dir/manuscript_id/parent_directory/picture_id
    parts = image_path.rstrip('/').split('/')
    if len(parts) >= 2:
        return parts[-3]  # manuscript_id is third from the end
    return 'unknown'


def process_image(
    image_path: str,
    patch_size: int,
    stride: int,
    max_patches: int,
) -> Tuple[Dict, List[Dict]]:
    """
    Process a single image and extract patch coordinates.
    
    Returns:
        Tuple of (image_metadata_dict, list_of_patch_coordinate_dicts)
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        w, h = image.size
        original_size = image.size
        
        # Determine explicit XML path for pretrain repository (if available)
        xml_override = find_xml_path_pretrain(image_path)

        # Extract patches using XML-based extraction
        patches, coords_tensor, metadata = extract_patches_with_xml(
            image=image,
            image_path=image_path,
            patch_size=patch_size,
            stride=stride,
            max_patches=max_patches,
            xml_path=xml_override,
        )
        
        # Convert coords tensor to list
        coords = coords_tensor.tolist()  # List of [cx, cy] pairs
        
        # Extract manuscript_id from path
        manuscript_id = extract_manuscript_id_from_path(image_path)
        
        # Normalize metadata values
        text_regions_count = int(metadata.get('text_regions_count', 0) or 0)
        is_two_page = _to_bool(metadata.get('is_two_page', False))
        rotation_corrected = _to_bool(metadata.get('rotation_corrected', False))
        reading_order_applied = _to_bool(metadata.get('reading_order_applied', False))
        page_split_x = metadata.get('page_split_x')
        if page_split_x is not None:
            try:
                page_split_x = float(page_split_x)
            except (TypeError, ValueError):
                page_split_x = None
        rotation_angle = metadata.get('rotation_angle', 0)
        try:
            rotation_angle = int(rotation_angle)
        except (TypeError, ValueError):
            rotation_angle = 0
        original_width, original_height = metadata.get('original_image_size', (w, h))
        try:
            original_width = int(original_width)
            original_height = int(original_height)
        except (TypeError, ValueError):
            original_width, original_height = w, h
        
        # Coordinates are already normalized to the rotated image (if rotation was applied)
        # We store them as-is - they're in rotated space, ready to use when loading from DB
        # The rotation_angle flag tells us to rotate the image when loading

        # Prepare image metadata record (one per image)
        image_metadata = {
            'image_path': image_path,
            'manuscript_id': manuscript_id,
            'patch_size': patch_size,
            'stride': stride,
            'num_patches': len(patches),
            'xml_path': metadata.get('xml_path'),
            'text_regions_count': text_regions_count,
            'is_two_page': is_two_page,
            'page_split_x': page_split_x,
            'rotation_angle': rotation_angle,
            'rotation_corrected': rotation_corrected,
            'reading_order_applied': reading_order_applied,
            'image_width': w,
            'image_height': h,
            'original_image_width': original_width,
            'original_image_height': original_height,
        }
        
        # Prepare patch coordinate records (one per patch)
        # Note: Coordinates are normalized to the rotated image (if rotation was applied)
        # When loading from DB, we'll rotate the image and use these coordinates directly
        patch_records = []
        for patch_idx, (cx, cy) in enumerate(coords):
            patch_record = {
                'image_path': image_path,
                'patch_index': patch_idx,
                'center_x_normalized': float(cx),
                'center_y_normalized': float(cy),
            }
            patch_records.append(patch_record)
        
        return image_metadata, patch_records
        
    except Exception as e:
        logger.exception(f"Error processing image {image_path}")
        print(f"Error processing image {image_path}: {e}")
        return None, []


def insert_image_metadata(conn, table_name: str, image_metadata: Dict, commit: bool = False):
    """Insert or update image metadata in the database."""
    if not image_metadata:
        return
    
    with conn.cursor() as cur:
        columns = [
            'image_path', 'manuscript_id', 'patch_size', 'stride', 'num_patches',
            'xml_path', 'text_regions_count', 'is_two_page', 'page_split_x',
            'rotation_angle', 'rotation_corrected', 'reading_order_applied',
            'image_width', 'image_height', 'original_image_width', 'original_image_height'
        ]
        
        values = (
            image_metadata['image_path'],
            image_metadata['manuscript_id'],
            image_metadata['patch_size'],
            image_metadata['stride'],
            image_metadata['num_patches'],
            image_metadata['xml_path'],
            image_metadata['text_regions_count'],
            image_metadata['is_two_page'],
            image_metadata['page_split_x'],
            image_metadata['rotation_angle'],
            image_metadata['rotation_corrected'],
            image_metadata['reading_order_applied'],
            image_metadata['image_width'],
            image_metadata['image_height'],
            image_metadata['original_image_width'],
            image_metadata['original_image_height'],
        )
        
        # Use ON CONFLICT to handle duplicates (update if exists)
        insert_query = f"""
            INSERT INTO {table_name} (
                {', '.join(columns)}
            ) VALUES ({', '.join(['%s'] * len(columns))})
            ON CONFLICT (image_path) 
            DO UPDATE SET
                manuscript_id = EXCLUDED.manuscript_id,
                patch_size = EXCLUDED.patch_size,
                stride = EXCLUDED.stride,
                num_patches = EXCLUDED.num_patches,
                xml_path = EXCLUDED.xml_path,
                text_regions_count = EXCLUDED.text_regions_count,
                is_two_page = EXCLUDED.is_two_page,
                page_split_x = EXCLUDED.page_split_x,
                rotation_angle = EXCLUDED.rotation_angle,
                rotation_corrected = EXCLUDED.rotation_corrected,
                reading_order_applied = EXCLUDED.reading_order_applied,
                image_width = EXCLUDED.image_width,
                image_height = EXCLUDED.image_height,
                original_image_width = EXCLUDED.original_image_width,
                original_image_height = EXCLUDED.original_image_height,
                updated_at = NOW()
        """
        
        cur.execute(insert_query, values)
        if commit:
            conn.commit()


def insert_patch_coordinates(conn, table_name: str, patch_records: List[Dict], commit: bool = False):
    """Insert patch coordinates into the database."""
    if not patch_records:
        return
    
    with conn.cursor() as cur:
        # Prepare data for bulk insert
        columns = [
            'image_path', 'patch_index',
            'center_x_normalized', 'center_y_normalized'
        ]
        
        values = []
        for record in patch_records:
            values.append((
                record['image_path'],
                record['patch_index'],
                record['center_x_normalized'],
                record['center_y_normalized'],
            ))
        
        # Use ON CONFLICT to handle duplicates (update if exists)
        insert_query = f"""
            INSERT INTO {table_name} (
                {', '.join(columns)}
            ) VALUES %s
            ON CONFLICT (image_path, patch_index) 
            DO UPDATE SET
                center_x_normalized = EXCLUDED.center_x_normalized,
                center_y_normalized = EXCLUDED.center_y_normalized
        """
        
        execute_values(cur, insert_query, values)
        if commit:
            conn.commit()


def get_existing_images(conn, image_metadata_table: str) -> set:
    """Get set of image paths that already have metadata in the database."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT image_path FROM {image_metadata_table}")
        return {row[0] for row in cur.fetchall()}


def main():
    """Main function to extract and store patch coordinates."""
    print("=" * 80)
    print("PATCH COORDINATES EXTRACTION FOR PRETRAINING DATASET")
    print("=" * 80)
    
    # Get database connection
    print(f"\nConnecting to database using config: {DB_CONFIG_PATH}")
    conn = get_db_connection()
    
    # Create tables
    print(f"\nCreating/verifying tables:")
    print(f"  - Image metadata: {PATCH_IMAGE_METADATA_TABLE_NAME}")
    print(f"  - Patch coordinates: {PATCH_COORDINATES_TABLE_NAME}")
    create_tables(conn, PATCH_IMAGE_METADATA_TABLE_NAME, PATCH_COORDINATES_TABLE_NAME)
    
    # Get pretraining dataset splits
    print(f"\nLoading pretraining dataset from: {BASE_DIR}")
    splits, split_stats = build_splits(BASE_DIR)
    
    # Collect all image paths from all splits (train, val, test)
    all_image_paths = []
    for split_name in ['train', 'val', 'test']:
        split_paths = []
        for manuscript_id, items in splits[split_name].items():
            split_paths.extend([p for p, x in items])
        all_image_paths.extend(split_paths)
        print(f"Found {len(split_paths)} images in {split_name} set")
    
    print(f"\nTotal images across all splits: {len(all_image_paths)}")
    
    # Check which images are already processed
    existing_images = get_existing_images(conn, PATCH_IMAGE_METADATA_TABLE_NAME)
    print(f"Found {len(existing_images)} images already in database")
    
    # Filter out already processed images
    images_to_process = [path for path in all_image_paths if path not in existing_images]
    print(f"Processing {len(images_to_process)} new images")
    
    if not images_to_process:
        print("\nAll images already processed!")
        conn.close()
        return
    
    # Process images
    print(f"\nExtracting patch coordinates...")
    print(f"Patch size: {TILE_SIZE}, Stride: {TILE_STRIDE}, Max patches: {MAX_TILES_TRAIN}")
    print(f"Committing every 100 manuscripts...")
    
    # Group images by manuscript for efficient processing
    images_by_manuscript = {}
    for image_path in images_to_process:
        manuscript_id = extract_manuscript_id_from_path(image_path)
        if manuscript_id not in images_by_manuscript:
            images_by_manuscript[manuscript_id] = []
        images_by_manuscript[manuscript_id].append(image_path)
    
    manuscript_ids = list(images_by_manuscript.keys())
    print(f"Processing {len(manuscript_ids)} manuscripts")
    
    total_patches = 0
    processed_count = 0
    error_count = 0
    manuscripts_processed = 0
    
    # Process manuscripts in groups of 100
    commit_interval = 100
    image_batch_size = 50  # Process images in smaller batches for memory efficiency
    
    for manuscript_idx, manuscript_id in enumerate(tqdm(manuscript_ids, desc="Processing manuscripts")):
        manuscript_images = images_by_manuscript[manuscript_id]
        
        # Process images in this manuscript in batches
        for i in range(0, len(manuscript_images), image_batch_size):
            image_batch = manuscript_images[i:i + image_batch_size]
            batch_image_metadata = []
            batch_patch_records = []
            
            for image_path in image_batch:
                image_metadata, patch_records = process_image(
                    image_path=image_path,
                    patch_size=TILE_SIZE,
                    stride=TILE_STRIDE,
                    max_patches=MAX_TILES_TRAIN,
                )
                
                if image_metadata and patch_records:
                    batch_image_metadata.append(image_metadata)
                    batch_patch_records.extend(patch_records)
                    processed_count += 1
                    total_patches += len(patch_records)
                else:
                    error_count += 1
            
            # Insert batch into database (but don't commit yet - we'll commit per manuscript)
            try:
                # Insert image metadata first
                for image_metadata in batch_image_metadata:
                    insert_image_metadata(conn, PATCH_IMAGE_METADATA_TABLE_NAME, image_metadata, commit=False)
                
                # Then insert patch coordinates
                if batch_patch_records:
                    insert_patch_coordinates(conn, PATCH_COORDINATES_TABLE_NAME, batch_patch_records, commit=False)
            except Exception as e:
                logger.exception(f"Error inserting images from manuscript {manuscript_id}")
                print(f"\nError inserting images from manuscript {manuscript_id}: {e}")
                conn.rollback()  # Rollback the failed transaction
                error_count += len(image_batch)
                # Continue with next batch instead of raising
        
        manuscripts_processed += 1
        
        # Commit every 100 manuscripts
        if manuscripts_processed % commit_interval == 0:
            try:
                conn.commit()
                print(f"\n✓ Committed {manuscripts_processed} manuscripts ({processed_count} images, {total_patches} patches)")
            except Exception as e:
                logger.exception("Error committing transaction")
                print(f"\n✗ Error committing transaction: {e}")
                conn.rollback()
                raise
    
    # Final commit for remaining manuscripts
    if manuscripts_processed % commit_interval != 0:
        try:
            conn.commit()
            print(f"\n✓ Final commit: {manuscripts_processed} manuscripts ({processed_count} images, {total_patches} patches)")
        except Exception as e:
            print(f"\n✗ Error committing final transaction: {e}")
            conn.rollback()
            raise
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Processed images: {processed_count}")
    print(f"Total patches extracted: {total_patches}")
    print(f"Errors: {error_count}")
    print(f"Average patches per image: {total_patches / processed_count if processed_count > 0 else 0:.2f}")
    
    # Get final database statistics
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {PATCH_IMAGE_METADATA_TABLE_NAME}")
        unique_images = cur.fetchone()[0]
        
        cur.execute(f"SELECT COUNT(*) FROM {PATCH_COORDINATES_TABLE_NAME}")
        total_patches_db = cur.fetchone()[0]
        
        print(f"\nDatabase statistics:")
        print(f"  Unique images in database: {unique_images}")
        print(f"  Total patch records: {total_patches_db}")
        print(f"  Average patches per image: {total_patches_db / unique_images if unique_images > 0 else 0:.2f}")
    
    conn.close()
    print("\n✅ Done!")


if __name__ == '__main__':
    main()

