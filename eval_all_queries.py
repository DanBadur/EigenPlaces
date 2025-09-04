#!/usr/bin/env python3
"""
Script to evaluate a trained EigenPlaces model on multiple query sets.
This script will test on all query folders found in the test directory.
"""

import sys
import torch
import logging
import multiprocessing
import os
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

def main():
    args = parse_arguments()
    
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

if __name__ == '__main__':
    main()
