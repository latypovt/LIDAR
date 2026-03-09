import os
import ants
import antspynet  # Deep learning for robust skull stripping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class LDBMEngine:
    def __init__(self, itk_threads=1):
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(itk_threads)

    def preprocess(self, image_path):
        """High-precision preprocessing for pediatric T1w."""
        img = ants.image_read(image_path)
        img = ants.n4_bias_field_correction(img)
        
        # DL-based brain extraction (modality='t1') is significantly 
        # more robust than standard masking for pediatric scans
        mask = antspynet.brain_extraction(img, modality='t1')
        brain = ants.mask_image(img, mask)
        
        # Resample to 1mm isotropic for Jacobian parity
        brain = ants.resample_image(brain, (1, 1, 1), use_voxels=False, interp_type=0)
        return brain

    def build_sst(self, session_paths, output_path):
        """Constructs SST with Rigid Pre-alignment to fix orientation."""
        # 1. Preprocess all sessions
        preprocessed_brains = [self.preprocess(p) for p in session_paths]
        
        # 2. MANDATORY Rigid Alignment to fix the 'double vision'
        # Use the first session as a temporary reference for orientation
        ref = preprocessed_brains[0]
        aligned_brains = []
        for brain in preprocessed_brains:
            reg = ants.registration(fixed=ref, moving=brain, type_of_transform='Rigid')
            aligned_brains.append(reg['warpedmovout'])
            
        # 3. Iterative Unbiased Template Build
        # Now that orientations are fixed and skull is gone, SyN can focus on neuroanatomy
        sst = ants.build_template(
            image_list=aligned_brains,
            iterations=4,             # Clinical standard
            type_of_transform='SyN',  # Non-linear refinement
            gradient_step=0.2
        )
        ants.image_write(sst, output_path)
        return sst
    
    def generate_log_jacobian(self, moving_path, sst_path, output_path):
        """Calculates Log-Jacobian from Session to SST."""
        fixed = ants.image_read(sst_path)
        moving = self.preprocess(moving_path) # Ensure moving is prepped same as SST
        
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN')
        
        # Extract Jacobian from the forward warp field
        jacobian = ants.create_jacobian_determinant_image(
            fixed, reg['fwdtransforms'][0], do_log=True
        )
        ants.image_write(jacobian, output_path)
        return output_path