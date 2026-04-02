import os
import argparse
import pandas as pd
import numpy as np
import nibabel as nib
import statsmodels
import nilearn
import joblib 
import patsy
from nilearn.maskers import NiftiMasker
import statsmodels.formula.api as smf
from scipy.stats import false_discovery_control 
from joblib import Parallel, delayed
from tqdm import tqdm
import glob
import warnings

def run_voxel_mixedlm(voxel_slice, df, formula):
    temp_df = df.copy()
    temp_df['JD'] = voxel_slice
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.mixedlm(formula, temp_df, groups=temp_df['subject_id'])
            result = model.fit(reml=True)
            return {
                'tvalues': result.tvalues.to_dict(),
                'params': result.params.to_dict(),
                'pvalues': result.pvalues.to_dict(),
                'resid': result.resid.values  # EXTRACTING THE ERROR (AFNI DEPENDENCY)
            }
        except Exception:
            # Fallback for failed convergence to maintain array structure
            return {'failed': True, 'resid': np.zeros(len(df))}

def main():
    parser = argparse.ArgumentParser(description="LIDAR: Voxel-wise MixedLM + AFNI 3dClustSim + PALM TFCE Export")
    parser.add_argument("bids_dir")
    parser.add_argument("metadata")
    parser.add_argument("analysis_name")
    parser.add_argument("--formula", default="JD ~ age + sex + session * outcome")
    parser.add_argument("--mask", required=True)
    parser.add_argument("--n_jobs", type=int, default=10)
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()

    root_out_dir = os.path.join(args.bids_dir, "derivatives", "dbm", "stats", args.analysis_name)
    os.makedirs(root_out_dir, exist_ok=True)

    # 1. Metadata, Formula Detection, and Mean Centering
    df = pd.read_csv(args.metadata, sep=None, engine='python')
    
    # Dynamically find which variables from the formula are in the dataframe
    formula_vars = args.formula.replace('~', '+').replace('*', '+').split('+')
    active_vars = [v.strip() for v in formula_vars if v.strip() in df.columns]

    for var in active_vars:
            # Mean center continuous biological covariates for model stability
            if 'age' in var.lower() or 'icv' in var.lower():
                mean_val = df[var].mean()
                df[var] = df[var] - mean_val
                print(f"  [STAT] Mean centering '{var}': Subtracted {mean_val:.4f}")

    # 2. File Matching
    # Updated to catch 'space-Template' and any jacobian type
    search_pattern = os.path.join(args.bids_dir, "derivatives", "dbm", "sub-*", "ses-*", "anat", "*_space-Template_desc-*Jacobian.nii.gz")
    jac_files = sorted(glob.glob(search_pattern))
    df['sub_match'] = df['subject_id'].apply(lambda x: f"sub-{x}" if "sub-" not in str(x) else str(x))
    df['ses_match'] = df['session'].apply(lambda x: f"ses-{x}" if "ses-" not in str(x) else str(x))

    matched_files, matched_metadata = [], []
    for f_path in jac_files:
        fname = os.path.basename(f_path)
        sub_bids, ses_bids = fname.split('_')[0], fname.split('_')[1]
        row = df[(df['sub_match'] == sub_bids) & (df['ses_match'] == ses_bids)]
        if not row.empty:
            matched_files.append(f_path); matched_metadata.append(row.iloc[0])

    temp_df = pd.DataFrame(matched_metadata)
    temp_df['file_list_idx'] = range(len(matched_files))
    
    formula_vars = args.formula.replace('~', '+').replace('*', '+').split('+')
    check_vars = [v.strip() for v in formula_vars if v.strip() in temp_df.columns]
    final_df = temp_df.dropna(subset=check_vars)
    final_files = [matched_files[i] for i in final_df['file_list_idx']]
    final_df = final_df.reset_index(drop=True)

    # 3. MNI Coordinate Tracking
    mask_img = nib.load(args.mask)
    mask_data = mask_img.get_fdata()
    affine = mask_img.affine
    header = mask_img.header
    mask_indices = np.array(np.where(mask_data > 0)).T 

    masker = NiftiMasker(mask_img=args.mask).fit()
    voxel_data = masker.transform(final_files)
    n_total = voxel_data.shape[1]

    # =========================================================================
    # 4. TFCE / FSL PALM EXPORT ENGINE
    # =========================================================================
    tfce_dir = os.path.join(root_out_dir, "tfce_palm")
    os.makedirs(tfce_dir, exist_ok=True)
    print(f"Exporting PALM dependencies to {tfce_dir}...")

    # A. Create 4D NIfTI Stack
    stacked_img = masker.inverse_transform(voxel_data)
    nib.save(stacked_img, os.path.join(tfce_dir, "4D_logJacobian.nii.gz"))
    nib.save(mask_img, os.path.join(tfce_dir, "mask.nii.gz"))

   # B. Generate Design Matrix (Fixed Effects Only)
    fe_formula = args.formula.split('~')[1].strip()
    design_df = patsy.dmatrix(fe_formula, final_df, return_type='dataframe')
    
    # Do NOT add Subject Dummies. 
    # Your baseline variables perfectly predict subject IDs, causing the rank deficiency.
    # Your '-eb eb.csv' file handles the repeated measures permutations in PALM.
    
    # Save raw CSV for PALM and Labels for you
    design_df.to_csv(os.path.join(tfce_dir, "design.csv"), index=False, header=False)
    with open(os.path.join(tfce_dir, "design_labels.txt"), "w") as f:
        f.write("\n".join(design_df.columns))

    # C. Generate Contrasts
    n_cols = design_df.shape[1]
    contrasts, contrast_names = [], []
    for i, col in enumerate(design_df.columns):
        if col == 'Intercept': continue 
        con_pos = np.zeros(n_cols); con_pos[i] = 1
        con_neg = np.zeros(n_cols); con_neg[i] = -1
        contrasts.extend([con_pos, con_neg])
        contrast_names.extend([f"{col}_pos", f"{col}_neg"])

    pd.DataFrame(contrasts).to_csv(os.path.join(tfce_dir, "contrasts.csv"), index=False, header=False)
    with open(os.path.join(tfce_dir, "contrast_labels.txt"), "w") as f:
        f.write("\n".join(contrast_names))
        
    # D. Exchangeability Blocks
    final_df['EB'] = final_df['subject_id'].astype('category').cat.codes + 1
    final_df[['EB']].to_csv(os.path.join(tfce_dir, "eb.csv"), index=False, header=False)
    # =========================================================================

    indices_to_process = range(50000, 60000) if args.test_run else range(n_total)
    
    # 5. Processing (MixedLM execution)
    raw_results = Parallel(n_jobs=args.n_jobs)(
        delayed(run_voxel_mixedlm)(voxel_data[:, i], final_df, args.formula) 
        for i in tqdm(indices_to_process, desc="Processing", ascii="○◔◑◕●", colour="#FEFD9b")
    )
    joblib.dump(raw_results, os.path.join(root_out_dir, "raw_results.pkl"))

    # =========================================================================
    # 6. AFNI 3dClustSim DEPENDENCY: RECONSTRUCT 4D RESIDUALS
    # =========================================================================
    print("Reconstructing 4D Residual map for AFNI Spatial Autocorrelation...")
    resid_matrix = np.zeros(voxel_data.shape) 
    
    for i, res in enumerate(raw_results):
        if res and not res.get('failed', False):
            resid_matrix[:, i] = res['resid']
            
    resid_img = masker.inverse_transform(resid_matrix)
    nib.save(resid_img, os.path.join(root_out_dir, "4D_residuals.nii.gz"))
    print(f"Residuals saved to {root_out_dir}/4D_residuals.nii.gz")
    # =========================================================================

    # 7. Full Feature Discovery & Scipy Export
    sample_res = next(r for r in raw_results if r and not r.get('failed', False))
    all_features = [k for k in sample_res['params'].keys() if k != 'Group Var']

    for feature in all_features:
        clean_name = feature.replace(':', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        feat_dir = os.path.join(root_out_dir, clean_name)
        os.makedirs(feat_dir, exist_ok=True)
        
        vol_t, vol_b, vol_p, vol_pfdr = np.zeros(mask_data.shape), np.zeros(mask_data.shape), np.ones(mask_data.shape), np.ones(mask_data.shape)
        valid_p, valid_local_idx = [], []
        
        for i, res in enumerate(raw_results):
            if res and not res.get('failed', False) and feature in res['pvalues']:
                p_val = res['pvalues'][feature]
                if not np.isnan(p_val):
                    valid_p.append(p_val); valid_local_idx.append(i)

        if valid_p:
            p_fdr = false_discovery_control(valid_p)
            rows = []
            
            for idx, (l_idx, p_unc, p_adj) in enumerate(zip(valid_local_idx, valid_p, p_fdr)):
                global_idx = indices_to_process[l_idx]
                x, y, z = mask_indices[global_idx]
                res = raw_results[l_idx]
                
                t, b = res['tvalues'][feature], res['params'][feature]
                vol_t[x, y, z], vol_b[x, y, z], vol_p[x, y, z], vol_pfdr[x, y, z] = t, b, p_unc, p_adj
                
                mni = nib.affines.apply_affine(affine, [x, y, z])
                rows.append({'X': mni[0], 'Y': mni[1], 'Z': mni[2], 't': t, 'beta': b, 'p_unc': p_unc, 'p_fdr': p_adj})

            pd.DataFrame(rows).to_csv(os.path.join(feat_dir, f"stats_{clean_name}.csv"), index=False)
            for d, n in zip([vol_t, vol_b, vol_p, vol_pfdr], ['tstat', 'beta', 'punc', 'pfdr']):
                nib.save(nib.Nifti1Image(d.astype(np.float32), affine, header), 
                         os.path.join(feat_dir, f"map_{n}.nii.gz"))

    print(f"Success. Isolated results for {len(all_features)} features saved to: {root_out_dir}")

if __name__ == "__main__":
    main()