import argparse
import os
from utilities.data import BIDSManager

def main():
    parser = argparse.ArgumentParser(description="LIDAR: Longitudinal Imaging Deformation Analysis Repository")
    
    # Required Arguments
    parser.add_argument("bids_dir", help="Path to the root of the BIDS dataset")

    
    # Task Selection
    parser.add_argument("--task", choices=["level1", "level2"], default="level1",
                        help="Processing level. Level1: Within-subject SST and Jacobians. Level2: Group mapping.")
    parser.add_argument("--subject", help="Process a specific subject (e.g., 001). If omitted, runs all.")
    
    # Performance Tuning
    parser.add_argument("--n_parallel", type=int, default=2, 
                        help="Number of subjects to process in parallel")
    parser.add_argument("--itk_threads", type=int, default=4, 
                        help="Number of threads per ANTs process")
    
    args = parser.parse_args()

    if args.task == "level1":
        print(f"--- Starting LIDAR Level 1: Subject-Specific Templates ---")
        manager = BIDSManager(
            bids_root=args.bids_dir, 
            n_parallel_subjects=args.n_parallel
        )
        # Update engine threads based on CLI input
        manager.engine.itk_threads = args.itk_threads 
        manager.run_all(subject_id=args.subject)
        
    elif args.task == "level2":
        # Placeholder for the next stage
        print("Level 2 selected. Searching for SST derivatives...")
        sst_dir = os.path.join(args.output_dir, "dbm")
        if not os.path.exists(sst_dir):
            print(f"Error: Could not find Level 1 derivatives at {sst_dir}. Run Level 1 first.")
        else:
            print("Logic for Level 2 (Group Template & Normalization) to be implemented.")

if __name__ == "__main__":
    main()