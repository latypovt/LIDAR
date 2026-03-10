import argparse
from utilities.data import BIDSManager

def main():
    parser = argparse.ArgumentParser(description="LIDAR: Longitudinal Imaging Deformation Analysis Repository")
    parser.add_argument("bids_dir", help="Path to the BIDS dataset root")
    parser.add_argument("--task", choices=["level1", "level2", "all"], default="level1",
                        help="Level1: SST/Jacobians. Level2: MNI Warp. all: Run both.")
    parser.add_argument("--mni_template", help="Required for level2 or all tasks.")
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
        manager.run_level1(subject_id=args.subject)
    elif args.task == "level2":
        if not args.mni_template: raise ValueError("Level 2 requires --mni_template")
        manager.run_level2(args.mni_template, subject_id=args.subject)
    elif args.task == "all":
        if not args.mni_template: raise ValueError("Full pipeline requires --mni_template")
        manager.run_all_levels(args.mni_template, subject_id=args.subject)

if __name__ == "__main__":
    main()