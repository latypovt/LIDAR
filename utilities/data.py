import os
from bids import BIDSLayout
from concurrent.futures import ThreadPoolExecutor
from utilities.ldbm import LDBMEngine
import glob
import ants

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

    def process_level1_subject(self, sub_id, jac_type='relative'):
        """Workflow for Level 1: SST and Jacobians in SST space."""
        try:
            t1_paths, sst_path, sessions = self.get_subject_workload(sub_id)
            
            # Defensive Check: Skip SST if it exists
            if not os.path.exists(sst_path):
                self.engine.build_sst(t1_paths, sst_path)
            
            for i, ses in enumerate(sessions):
                out_dir = os.path.join(self.deriv_root, f"sub-{sub_id}", f"ses-{ses}", "anat")
                os.makedirs(out_dir, exist_ok=True)
                
                # Update naming to track jacobian type
                jac_out = os.path.join(out_dir, f"sub-{sub_id}_ses-{ses}_desc-{jac_type}Jacobian_stat.nii.gz")
                
                # Defensive Check: Skip Jacobian if it exists
                if not os.path.exists(jac_out):
                    self.engine.generate_log_jacobian(t1_paths[i], sst_path, jac_out, jac_type)
            
            return f"DONE Level 1: sub-{sub_id}"
        except Exception as e:
            return f"FAIL Level 1: sub-{sub_id} -> {str(e)}"

    def process_level2_subject(self, sub_id, template_path, jac_type='relative'):
        """Workflow for Level 2: Composed Warp to Atlas space."""
        try:
            # We need t1_paths to generate the fresh composed warp from native space
            t1_paths, sst_path, sessions = self.get_subject_workload(sub_id)
            
            if not os.path.exists(sst_path):
                return f"SKIP Level 2: sub-{sub_id} (No SST)"

            for i, ses in enumerate(sessions):
                out_dir = os.path.join(self.deriv_root, f"sub-{sub_id}", f"ses-{ses}", "anat")
                
                # Update naming to reflect common template space and jacobian type
                jac_out = os.path.join(out_dir, f"sub-{sub_id}_ses-{ses}_space-Template_desc-{jac_type}Jacobian.nii.gz")
                
                # Defensive Check: Skip warping if already done
                if not os.path.exists(jac_out):
                    self.engine.warp_composed_jacobian(t1_paths[i], sst_path, template_path, jac_out, jac_type)
            
            return f"DONE Level 2: sub-{sub_id}"
        except Exception as e:
            return f"FAIL Level 2: sub-{sub_id} -> {str(e)}"

    def run_level1(self, subject_id=None, jac_type='relative'):
        """Runs SST and Subject-Space Jacobian generation."""
        subjects = [subject_id] if subject_id else self.layout.get_subjects()
        print(f"--- Running Level 1 ({jac_type.upper()}) for {len(subjects)} subjects ---")
        
        with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
            # Lambda passes the jac_type down to the worker thread
            results = list(executor.map(lambda s: self.process_level1_subject(s, jac_type), subjects))
        for r in results: print(r)

    def run_level2_composed(self, template_path, subject_id=None, jac_type='relative'):
        """Parallel execution of MNI/Atlas composed warping."""
        subjects = [subject_id] if subject_id else self.layout.get_subjects()
        print(f"--- Running Level 2 Composed ({jac_type.upper()}) for {len(subjects)} subjects ---")
        
        with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
            # Lambda passes both the template path and jac_type down to the worker thread
            results = list(executor.map(lambda s: self.process_level2_subject(s, template_path, jac_type), subjects))
        for r in results: print(r)

    def run_all_levels(self, mni_path, subject_id=None, jac_type='relative'):
            """Full pipeline: Level 1 then Level 2."""
            self.run_level1(subject_id=subject_id, jac_type=jac_type)
            self.run_level2_composed(mni_path, subject_id=subject_id, jac_type=jac_type)

    def build_population_template(self, output_path, iterations=3):
        # 1. Find the paths
        all_sst_paths = glob.glob(os.path.join(self.deriv_root, "sub-*", "sst", "*_desc-SST_T1w.nii.gz"))
        
        # 2. LOAD AND PAD THE SPATIAL GRID
        print(f"--- Loading and Padding {len(all_sst_paths)} SSTs ---")
        all_sst_images = []
        for p in all_sst_paths:
            img = ants.image_read(p)
            # Pad by 30 voxels on all 6 sides (X, Y, Z) to expand the FOV
            padded_img = ants.pad_image(img, pad_width=[(30, 30), (30, 30), (30, 30)], value=0.0)
            all_sst_images.append(padded_img)
            
        print(f"--- Building Level 2 Population Template from 30 SSTs ---")
        
        # 3. Pass the padded images to the engine
        pop_template = ants.build_template(
            image_list=all_sst_images,
            iterations=iterations,
            type_of_transform='SyN',
            syn_metric='cc',
            syn_niters=[50, 50, 10],
            gradient_step=0.2 
        )
        
        ants.image_write(pop_template, output_path)
        return output_path