import os
import argparse
import ants
import numpy as np
import scipy.ndimage as ndi

def prepare_lnm_clusters(tstat_path, group_atlas_path, mni_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Loading Images ---")
    img_tstat = ants.image_read(tstat_path)
    img_group = ants.image_read(group_atlas_path)
    img_mni = ants.image_read(mni_path)
    
    print("--- Step 1: Registering Group Template to MNI Space ---")
    # SyN registration to calculate the warp from your study atlas to MNI
    reg = ants.registration(
        fixed=img_mni, 
        moving=img_group, 
        type_of_transform='SyN',
        reg_iterations=(40, 20, 0) # Standard precision is sufficient for template-to-template
    )
    
    print("--- Step 2: Warping TFCE T-Stat Map to MNI Space ---")
    # nearestNeighbor is mandatory here to prevent edge-smearing of thresholded stats
    warped_tstat = ants.apply_transforms(
        fixed=img_mni,
        moving=img_tstat,
        transformlist=reg['fwdtransforms'],
        interpolator='nearestNeighbor'
    )
    
    tstat_data = warped_tstat.numpy()
    tstat_data = np.nan_to_num(tstat_data) # Sanitize any NaN values from AFNI/FSL
    
    # Define 3D connectivity for cluster isolation (26-neighbor connectivity)
    structure = np.ones((3, 3, 3), dtype=int)
    
    print("--- Step 3 & 4: Isolating Positive and Negative Clusters ---")
    
    # Process Positive Clusters (Expansion/Growth signatures)
    pos_mask = (tstat_data > 0).astype(int)
    pos_labels, pos_num_features = ndi.label(pos_mask, structure=structure)
    print(f"Found {pos_num_features} isolated positive clusters.")
    
    for i in range(1, pos_num_features + 1):
        cluster_data = (pos_labels == i).astype(np.float32)
        # Create a new ANTs image inheriting the warped MNI spatial geometry
        cluster_img = warped_tstat.new_image_like(cluster_data)
        out_file = os.path.join(output_dir, f"seed_MNI_positive_cluster_{i:02d}.nii.gz")
        ants.image_write(cluster_img, out_file)
        
    # Process Negative Clusters (Pruning/Atrophy signatures)
    neg_mask = (tstat_data < 0).astype(int)
    neg_labels, neg_num_features = ndi.label(neg_mask, structure=structure)
    print(f"Found {neg_num_features} isolated negative clusters.")
    
    for i in range(1, neg_num_features + 1):
        cluster_data = (neg_labels == i).astype(np.float32)
        cluster_img = warped_tstat.new_image_like(cluster_data)
        out_file = os.path.join(output_dir, f"seed_MNI_negative_cluster_{i:02d}.nii.gz")
        ants.image_write(cluster_img, out_file)

    print(f"--- Complete. All binary seed masks saved to: {output_dir} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolate and warp TFCE clusters for Lesion Network Mapping.")
    parser.add_argument("tstat_path", help="Path to the thresholded TFCE t-stat map (in group space)")
    parser.add_argument("group_atlas_path", help="Path to your study_atlas.nii.gz")
    parser.add_argument("mni_path", help="Path to the standard MNI152 template")
    parser.add_argument("output_dir", help="Directory to save the isolated MNI-space seed NIfTIs")
    
    args = parser.parse_args()
    prepare_lnm_clusters(args.tstat_path, args.group_atlas_path, args.mni_path, args.output_dir)