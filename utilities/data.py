import os
from bids import BIDSLayout
from concurrent.futures import ThreadPoolExecutor
from utilities.ldbm import LDBMEngine # Ensure this matches your file structure

class BIDSManager:
    def __init__(self, bids_root, n_parallel_subjects=2, itk_threads=4):
        self.layout = BIDSLayout(bids_root)
        # Standardizing naming to match your 'lidar' request
        self.deriv_root = os.path.join(bids_root, "derivatives", "dbm") 
        self.engine = LDBMEngine(itk_threads=itk_threads) 
        self.n_parallel = n_parallel_subjects

    def get_subject_workload(self, sub_id):
        """Finds sessions and prepares output paths following BIDS derivatives."""
        sessions = self.layout.get_sessions(subject=sub_id)
        t1_paths = [self.layout.get(subject=sub_id, session=s, suffix='T1w', extension='nii.gz', return_type='file')[0] 
                    for s in sessions]
        
        sub_deriv_dir = os.path.join(self.deriv_root, f"sub-{sub_id}", "sst")
        os.makedirs(sub_deriv_dir, exist_ok=True)
        sst_path = os.path.join(sub_deriv_dir, f"sub-{sub_id}_desc-SST_T1w.nii.gz")
        
        return t1_paths, sst_path, sessions

    def process_subject(self, sub_id):
        """Workflow for a single subject."""
        try:
            t1_paths, sst_path, sessions = self.get_subject_workload(sub_id)
            
            if not os.path.exists(sst_path):
                self.engine.build_sst(t1_paths, sst_path)
            
            for i, ses in enumerate(sessions):
                out_dir = os.path.join(self.deriv_root, f"sub-{sub_id}", f"ses-{ses}", "anat")
                os.makedirs(out_dir, exist_ok=True)
                jac_out = os.path.join(out_dir, f"sub-{sub_id}_ses-{ses}_desc-logJacobian_stat.nii.gz")
                
                self.engine.generate_log_jacobian(t1_paths[i], sst_path, jac_out)
            
            return f"Successfully processed sub-{sub_id}"
        except Exception as e:
            return f"Error in sub-{sub_id}: {str(e)}"

    def run_all(self, subject_id=None):
        """Runs processing. If subject_id is provided, runs only that one."""
        if subject_id:
            subjects = [subject_id]
            print(f"Targeted Run: Processing sub-{subject_id} only.")
        else:
            subjects = self.layout.get_subjects()
            print(f"Batch Run: Processing {len(subjects)} subjects...")
        
        # Parallel execution
        with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
            results = list(executor.map(self.process_subject, subjects))
        
        for res in results:
            print(res)

if __name__ == "__main__":
    # Example usage
    manager = BIDSManager(bids_root="/path/to/data", deriv_root="/path/to/derivatives")
    manager.run_all()