import os
import ants
import numpy as np

class LDBMEngine:
    def __init__(self, itk_threads=1):
        # Control internal ANTs parallelism to avoid CPU over-subscription
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(itk_threads)

    def preprocess(self, image_path, resample_spacing=(1, 1, 1)):
        """N4 Bias Correction + Brain Extraction + Resampling."""
        img = ants.image_read(image_path)
        # 1. N4 Correction
        img = ants.n4_bias_field_correction(img)
        # 2. Resample to ensure Jacobian consistency
        img = ants.resample_image(img, resample_spacing, use_voxels=False, interp_type=0)
        # 3. Brain Extraction
        mask = ants.get_mask(img)
        return img * mask

    def build_sst(self, session_paths, output_path):
        """Constructs the Unbiased Subject-Specific Template."""
        images = [self.preprocess(p) for p in session_paths]
        
        # High-level SyN template building (The 'Aperture' method)
        sst = ants.build_template(
            image_list=images,
            iterations=3,
            gradient_step=0.2,
            type_of_transform='SyN'
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