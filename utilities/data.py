import os
from bids import BIDSLayout
from concurrent.futures import ThreadPoolExecutor
from utilities.ldbm import LDBMEngine

class BIDSManager:
    def __init__(self, bids_root, n_parallel_subjects=2, itk_threads=4):
        self.layout = BIDSLayout(bids_root)
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

    def process_level1_subject(self, sub_id):
        """Workflow for Level 1: SST and Jacobians in SST space."""
        try:
            t1_paths, sst_path, sessions = self.get_subject_workload(sub_id)
            
            # Defensive Check: Skip SST if it exists
            if not os.path.exists(sst_path):
                self.engine.build_sst(t1_paths, sst_path)
            
            for i, ses in enumerate(sessions):
                out_dir = os.path.join(self.deriv_root, f"sub-{sub_id}", f"ses-{ses}", "anat")
                os.makedirs(out_dir, exist_ok=True)
                jac_out = os.path.join(out_dir, f"sub-{sub_id}_ses-{ses}_desc-logJacobian_stat.nii.gz")
                
                # Defensive Check: Skip Jacobian if it exists
                if not os.path.exists(jac_out):
                    self.engine.generate_log_jacobian(t1_paths[i], sst_path, jac_out)
            
            return f"DONE Level 1: sub-{sub_id}"
        except Exception as e:
            return f"FAIL Level 1: sub-{sub_id} -> {str(e)}"

    def process_level2_subject(self, sub_id, mni_path):
        """Workflow for Level 2: Warp Jacobians to MNI space."""
        try:
            sessions = self.layout.get_sessions(subject=sub_id)
            sst_path = os.path.join(self.deriv_root, f"sub-{sub_id}", "sst", f"sub-{sub_id}_desc-SST_T1w.nii.gz")
            
            if not os.path.exists(sst_path):
                return f"SKIP Level 2: sub-{sub_id} (No SST)"

            for ses in sessions:
                jac_in = os.path.join(self.deriv_root, f"sub-{sub_id}", f"ses-{ses}", "anat", 
                                      f"sub-{sub_id}_ses-{ses}_desc-logJacobian_stat.nii.gz")
                jac_out = os.path.join(self.deriv_root, f"sub-{sub_id}", f"ses-{ses}", "anat", 
                                       f"sub-{sub_id}_ses-{ses}_space-MNI_desc-logJacobian_stat.nii.gz")
                
                # Defensive Check: Skip warping if already done
                if os.path.exists(jac_in) and not os.path.exists(jac_out):
                    self.engine.warp_sst_to_mni(sst_path, mni_path, jac_in, jac_out)
            
            return f"DONE Level 2: sub-{sub_id}"
        except Exception as e:
            return f"FAIL Level 2: sub-{sub_id} -> {str(e)}"

    def run_level1(self, subject_id=None):
        """Runs SST and Subject-Space Jacobian generation."""
        subjects = [subject_id] if subject_id else self.layout.get_subjects()
        print(f"--- Running Level 1 for {len(subjects)} subjects ---")
        
        with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
            results = list(executor.map(self.process_level1_subject, subjects))
        for r in results: print(r)

    def run_level2(self, mni_path, subject_id=None):
        """Parallel execution of MNI warping."""
        subjects = [subject_id] if subject_id else self.layout.get_subjects()
        print(f"--- Running Level 2 (MNI) for {len(subjects)} subjects ---")
        
        with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
            # We use a lambda to pass the mni_path to each thread
            results = list(executor.map(lambda s: self.process_level2_subject(s, mni_path), subjects))
        for r in results: print(r)

    def run_all_levels(self, mni_path, subject_id=None):
        """Full pipeline: Level 1 then Level 2."""
        self.run_level1(subject_id=subject_id)
        self.run_level2(mni_path, subject_id=subject_id)