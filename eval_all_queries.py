#!/usr/bin/env python3
"""
Script to evaluate a trained EigenPlaces model on multiple query sets.
This script will test on all query folders found in the test directory.
Can optionally generate CSV files in VPR analysis format for visualization.
"""

import sys
import torch
import logging
import multiprocessing
import os
import csv
import numpy as np
from datetime import datetime
from pathlib import Path

import test
from args_parser import parse_arguments
import commons
from datasets.test_dataset import TestDataset
from eigenplaces_model import eigenplaces_network

torch.backends.cudnn.benchmark = True  # Provides a speedup

def find_query_folders(test_dataset_folder):
    """Find all query folders in the test dataset directory."""
    test_path = Path(test_dataset_folder)
    query_folders = []
    
    # Look for folders that start with 'queries'
    for item in test_path.iterdir():
        if item.is_dir() and item.name.startswith('queries'):
            query_folders.append(item.name)
    
    return sorted(query_folders)

def extract_utm_coordinates(image_path):
    """Extract UTM coordinates from image filename."""
    filename = os.path.basename(image_path)
    parts = filename.split('@')
    if len(parts) >= 3:
        try:
            utm_east = float(parts[1])
            utm_north = float(parts[2])
            return utm_east, utm_north
        except (ValueError, IndexError):
            return None, None
    return None, None

def calculate_distance(utm1, utm2):
    """Calculate Euclidean distance between two UTM coordinates."""
    if utm1 is None or utm2 is None:
        return float('inf')
    return np.sqrt((utm1[0] - utm2[0])**2 + (utm1[1] - utm2[1])**2)

def generate_vpr_csv(test_ds, model, args, output_path, top_k=5):
    """Generate CSV file in VPR analysis format."""
    logging.info(f"Generating VPR CSV for {top_k} top matches...")
    
    model = model.eval()
    with torch.no_grad():
        # Extract database descriptors
        logging.debug("Extracting database descriptors for CSV generation")
        database_subset_ds = torch.utils.data.Subset(test_ds, list(range(test_ds.database_num)))
        database_dataloader = torch.utils.data.DataLoader(
            dataset=database_subset_ds, 
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size, 
            pin_memory=(args.device == "cuda")
        )
        
        all_descriptors = np.empty((len(test_ds), args.fc_output_dim), dtype="float32")
        for images, indices in database_dataloader:
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
        
        # Extract query descriptors
        logging.debug("Extracting query descriptors for CSV generation")
        queries_subset_ds = torch.utils.data.Subset(
            test_ds, 
            list(range(test_ds.database_num, test_ds.database_num + test_ds.queries_num))
        )
        queries_dataloader = torch.utils.data.DataLoader(
            dataset=queries_subset_ds, 
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size, 
            pin_memory=(args.device == "cuda")
        )
        
        for images, indices in queries_dataloader:
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
    
    # Get query and database descriptors
    queries_descriptors = all_descriptors[test_ds.database_num:]
    database_descriptors = all_descriptors[:test_ds.database_num]
    
    # Find top-k matches using FAISS
    import faiss
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)
    
    _, predictions = faiss_index.search(queries_descriptors, top_k)
    
    # Generate CSV
    csv_columns = ['query_image_path', 'utm_east', 'utm_north']
    for i in range(1, top_k + 1):
        csv_columns.extend([
            f'reference_{i}_path',
            f'reference_{i}_distance',
            f'reference_{i}_utm_east',
            f'reference_{i}_utm_north'
        ])
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_columns)
        
        for query_idx, preds in enumerate(predictions):
            query_path = test_ds.queries_paths[query_idx]
            query_utm_east, query_utm_north = extract_utm_coordinates(query_path)
            
            row = [query_path, query_utm_east, query_utm_north]
            
            for pred_idx in preds:
                ref_path = test_ds.database_paths[pred_idx]
                ref_utm_east, ref_utm_north = extract_utm_coordinates(ref_path)
                
                # Calculate distance
                distance = calculate_distance(
                    (query_utm_east, query_utm_north),
                    (ref_utm_east, ref_utm_north)
                )
                
                row.extend([ref_path, distance, ref_utm_east, ref_utm_north])
            
            writer.writerow(row)
    
    logging.info(f"CSV file saved to: {output_path}")

def main():
    # Add custom argument for CSV generation
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate EigenPlaces model on multiple query sets')
    parser.add_argument('--test_dataset_folder', type=str, required=True,
                        help='path of the folder with test images (split in database/queries)')
    parser.add_argument('--resume_model', type=str, required=True,
                        help='path to best_model.pth')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda, cpu, or cuda:N')
    parser.add_argument('--backbone', type=str, default='ResNet18',
                        choices=['VGG16', 'ResNet18', 'ResNet50', 'ResNet101', 'ResNet152'],
                        help='Backbone architecture')
    parser.add_argument('--fc_output_dim', type=int, default=512,
                        help='Output dimension of final fully connected layer')
    parser.add_argument('--infer_batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    parser.add_argument('--positive_dist_threshold', type=int, default=25,
                        help='distance in meters for a prediction to be considered a positive')
    parser.add_argument('--save_dir', type=str, default='default',
                        help='name of directory on which to save the logs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # CSV generation arguments
    parser.add_argument('--generate_csv', action='store_true',
                        help='Generate CSV files in VPR analysis format')
    parser.add_argument('--csv_output_dir', type=str, 
                        default='C:/Users/danba/Desktop/Git Projects/VPR_dataset_analysis',
                        help='Directory to save CSV files')
    parser.add_argument('--csv_top_k', type=int, default=5,
                        help='Number of top-k matches to include in CSV')
    
    
    args = parser.parse_args()
    
    # Validate required arguments
    if args.test_dataset_folder is None:
        print("Error: --test_dataset_folder is required")
        sys.exit(1)
    
    if args.resume_model is None:
        print("Error: --resume_model is required (path to best_model.pth)")
        sys.exit(1)
    
    start_time = datetime.now()
    output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.make_deterministic(args.seed)
    commons.setup_logging(output_folder, console="info")
    
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")
    logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

    # Find all query folders
    query_folders = find_query_folders(args.test_dataset_folder)
    if not query_folders:
        logging.error(f"No query folders found in {args.test_dataset_folder}")
        sys.exit(1)
    
    logging.info(f"Found query folders: {query_folders}")

    #### Model
    if args.resume_model == "torchhub":
        model = torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                               backbone=args.backbone, fc_output_dim=args.fc_output_dim)
    else:
        model = eigenplaces_network.GeoLocalizationNet_(args.backbone, args.fc_output_dim)
        
        if args.resume_model is not None:
            logging.info(f"Loading model from {args.resume_model}")
            model_state_dict = torch.load(args.resume_model)
            model.load_state_dict(model_state_dict)
        else:
            logging.info("WARNING: You didn't provide a path to resume the model. " +
                         "Evaluation will be computed using randomly initialized weights.")

    model = model.to(args.device)

    # Test on each query folder
    all_results = {}
    
    for query_folder in query_folders:
        logging.info(f"\n{'='*60}")
        logging.info(f"Testing on {query_folder}")
        logging.info(f"{'='*60}")
        
        try:
            test_ds = TestDataset(args.test_dataset_folder, 
                                queries_folder=query_folder,
                                positive_dist_threshold=args.positive_dist_threshold)
            
            recalls, recalls_str = test.test(args, test_ds, model)
            all_results[query_folder] = {
                'recalls': recalls,
                'recalls_str': recalls_str,
                'dataset_info': str(test_ds)
            }
            
            logging.info(f"{test_ds}: {recalls_str}")
            
            # Generate CSV if requested
            if args.generate_csv:
                csv_filename = f"{query_folder}_results.csv"
                csv_path = os.path.join(args.csv_output_dir, csv_filename)
                logging.info(f"Generating CSV for {query_folder}...")
                generate_vpr_csv(test_ds, model, args, csv_path, args.csv_top_k)
            
        except Exception as e:
            logging.error(f"Error testing on {query_folder}: {str(e)}")
            all_results[query_folder] = {'error': str(e)}
    
    # Print summary
    logging.info(f"\n{'='*60}")
    logging.info("SUMMARY OF ALL RESULTS")
    logging.info(f"{'='*60}")
    
    for query_folder, result in all_results.items():
        if 'error' in result:
            logging.info(f"{query_folder}: ERROR - {result['error']}")
        else:
            logging.info(f"{query_folder}: {result['recalls_str']}")
    
    # Save results to file
    results_file = f"{output_folder}/all_query_results.txt"
    with open(results_file, 'w') as f:
        f.write("EigenPlaces Multi-Query Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {args.resume_model}\n")
        f.write(f"Test Dataset: {args.test_dataset_folder}\n")
        f.write(f"Backbone: {args.backbone}\n")
        f.write(f"Output Dim: {args.fc_output_dim}\n")
        f.write(f"Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for query_folder, result in all_results.items():
            f.write(f"{query_folder}:\n")
            if 'error' in result:
                f.write(f"  ERROR: {result['error']}\n")
            else:
                f.write(f"  {result['dataset_info']}\n")
                f.write(f"  {result['recalls_str']}\n")
            f.write("\n")
    
    logging.info(f"\nDetailed results saved to: {results_file}")
    
    # Summary of CSV files if generated
    if args.generate_csv:
        logging.info(f"\n{'='*60}")
        logging.info("CSV FILES GENERATED")
        logging.info(f"{'='*60}")
        for query_folder in query_folders:
            if 'error' not in all_results[query_folder]:
                csv_filename = f"{query_folder}_results.csv"
                csv_path = os.path.join(args.csv_output_dir, csv_filename)
                logging.info(f"{query_folder}: {csv_path}")

if __name__ == '__main__':
    main()
