import os
import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import label

def main():
    if len(sys.argv) != 4:
        print("Usage: python apply_fwe_threshold.py <feat_dir> <p_thresh> <k_thresh>")
        sys.exit(1)

    feat_dir = sys.argv[1]
    p_thresh = float(sys.argv[2])
    k_thresh = float(sys.argv[3])

    p_img_path = os.path.join(feat_dir, "map_punc.nii.gz")
    t_img_path = os.path.join(feat_dir, "map_tstat.nii.gz")

    if not os.path.exists(p_img_path) or not os.path.exists(t_img_path):
        print(f"  [!] Skipping: {feat_dir} lacks required maps.")
        sys.exit(0)

    p_img = nib.load(p_img_path)
    p_data = p_img.get_fdata()
    t_img = nib.load(t_img_path)
    t_data = t_img.get_fdata()

    # 1. Voxel-wise threshold (ignore background NaNs/zeros)
    sig_mask = (p_data < p_thresh) & (p_data > 0)
    sig_mask = sig_mask.astype(int)

    # 2. Cluster-size threshold (NN1 face-to-face connectivity to match AFNI)
    labeled_array, num_features = label(sig_mask)
    final_mask = np.zeros_like(sig_mask)

    for i in range(1, num_features + 1):
        cluster_size = np.sum(labeled_array == i)
        if cluster_size >= k_thresh:
            final_mask[labeled_array == i] = 1

    # 3. Export
    p_str = str(p_thresh).replace('.', '')
    
    # Save the binary mask for data extraction
    mask_out = os.path.join(feat_dir, f"mask_FWE_p{p_str}.nii.gz")
    nib.save(nib.Nifti1Image(final_mask, p_img.affine, p_img.header), mask_out)

    # Save the thresholded T-stat for visualization in FSLeyes
    masked_t = t_data * final_mask
    t_out = os.path.join(feat_dir, f"map_tstat_FWE_p{p_str}.nii.gz")
    nib.save(nib.Nifti1Image(masked_t, p_img.affine, p_img.header), t_out)

    print(f"  -> Thresholded p < {p_thresh} (k >= {k_thresh}). Saved to {feat_dir}")

if __name__ == "__main__":
    main()