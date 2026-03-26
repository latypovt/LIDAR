import argparse
from utilities.data import BIDSManager

def main():
    parser = argparse.ArgumentParser(description="LIDAR: Longitudinal Imaging Deformation Analysis Repository")
    parser.add_argument("bids_dir", help="Path to the BIDS dataset root")
    
    # Task selection updated to include population template building
    parser.add_argument("--task", choices=["level1", "level2", "all", "pop_template"], default="level1",
                        help="level1: SST/Jacobians. level2: Warp to Common Space. pop_template: Build Study-Specific Atlas. all: Run full chain.")
    
    # NEW: Toggle between local shape (relative) and total volume (absolute)
    parser.add_argument("--jac_type", choices=["relative", "absolute"], default="absolute",
                        help="Relative (local change) vs Absolute (total volume change, includes head size).")
    
    parser.add_argument("--mni_template", help="Standard space template (MNI or your Study-Specific Atlas).")
    parser.add_argument("--subject", help="Optional specific subject ID.")
    parser.add_argument("--n_parallel", type=int, default=2)
    parser.add_argument("--itk_threads", type=int, default=4)
    
    args = parser.parse_args()

    manager = BIDSManager(
        bids_root=args.bids_dir, 
        n_parallel_subjects=args.n_parallel,
        itk_threads=args.itk_threads
    )

    if args.task == "level1":
        # Pass the Jacobian type down to the LDBMEngine
        manager.run_level1(subject_id=args.subject, jac_type=args.jac_type)
        
    elif args.task == "pop_template":
        # Build the 'Level 2' representative average of your subjects
        output_atlas = os.path.join(args.bids_dir, "derivatives", "dbm", "study_atlas.nii.gz")
        manager.build_population_template(output_atlas)

    elif args.task == "level2":
        if not args.mni_template: raise ValueError("Level 2 requires --mni_template (MNI or study_atlas.nii.gz)")
        # Now uses the Composed Warp approach
        manager.run_level2_composed(args.mni_template, subject_id=args.subject)

    elif args.task == "all":
        if not args.mni_template: raise ValueError("Full pipeline requires --mni_template")
        manager.run_all_levels(args.mni_template, subject_id=args.subject, jac_type=args.jac_type)

if __name__ == "__main__":
    main()