import os
import ants
import antspynet  # Deep learning for robust skull stripping
from dipy.denoise.gibbs import gibbs_removal  # For Gibbs ringing correction


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class LDBMEngine:
    def __init__(self, itk_threads=1):
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(itk_threads)

    def preprocess(self, image_path, resample_spacing=(1, 1, 1)):
            """Standardizes orientation and performs De-Gibbs + DL-based stripping."""
            img = ants.image_read(image_path)
            
            # 1. Fix Orientation
            img = ants.reorient_image2(img, "RPI") 
            
            # 2. Gibbs Ringing Removal (Kellner et al. method)
            # We convert to numpy for dipy, then back to ants
            data = img.numpy()
            corrected_data = gibbs_removal(data, slice_axis=2) # Assuming axial acquisition
            img = img.new_image_like(corrected_data)
            
            # 3. N4 Bias Correction
            img = ants.n4_bias_field_correction(img)
            
            # 4. Robust DL Brain Extraction
            prob_mask = antspynet.brain_extraction(img, modality='t1')
            mask = ants.threshold_image(prob_mask, 0.5, 1.0)
            brain = ants.mask_image(img, mask)
            
            # 5. Resample with B-Spline (better for preventing new ringing)
            # Switch interp_type to 4 (B-Spline) instead of 0 (Linear)
            brain = ants.resample_image(brain, resample_spacing, use_voxels=False, interp_type=4)
            
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
        # Update this in utilities/ldbm.py
        sst = ants.build_template(
            image_list=aligned_brains,
            iterations=3,             # Increase to 5 for clinical grade
            gradient_step=0.2,       # Slower step size to prevent 'overshooting' (ringing)
            type_of_transform='SyN',
            # Added explicit multi-resolution niters for internal SyN
            syn_niters=[50, 50, 10], 
            syn_metric='mi'          # Cross-correlation is more robust to ringing than MI
            
        )
        ants.image_write(sst, output_path)
        return sst
    
    def warp_sst_to_mni(self, sst_path, mni_template_path, jacobian_path, output_path):
        """
        Registers SST to MNI and warps the Level 1 Jacobian.
        """
        fixed_mni = ants.image_read(mni_template_path)
        moving_sst = ants.image_read(sst_path)
        
        # 1. Register SST to MNI (SyN is best for cross-subject mapping)
        # Using 'SyN' here ensures the subject's anatomy fits the standard grid
        reg = ants.registration(
            fixed=fixed_mni, 
            moving=moving_sst, 
            type_of_transform='SyN'
        )
        
        # 2. Warp the Jacobian to MNI Space
        # Use 'linear' interpolation for the statistical map to preserve values
        warped_jac = ants.apply_transforms(
            fixed=fixed_mni,
            moving=ants.image_read(jacobian_path),
            transformlist=reg['fwdtransforms'],
            interpolator='linear'
        )
        
        ants.image_write(warped_jac, output_path)
        return output_path
    
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