import os
import json
import argparse
import pandas as pd
import numpy as np
import nibabel as nib
import statsmodels
import nilearn
from nilearn.maskers import NiftiMasker
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from tqdm import tqdm
import glob
import warnings

def run_voxel_mixedlm(voxel_slice, df, formula):
    """
    Worker function for parallel voxel-wise MixedLM.
    Groups by subject_id to account for longitudinal random intercepts.
    """
    temp_df = df.copy()
    temp_df['JD'] = voxel_slice
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.mixedlm(formula, temp_df, groups=temp_df['subject_id'])
            # Correcting parameter to 'reml' for restricted maximum likelihood
            result = model.fit(reml=True)
            return {
                'tvalues': result.tvalues.to_dict(),
                'params': result.params.to_dict(),
                'pvalues': result.pvalues.to_dict()
            }
        except Exception:
            return None

def main():
    parser = argparse.ArgumentParser(description="LIDAR Voxel-wise MixedLM Pipeline")
    parser.add_argument("bids_dir", help="BIDS root directory")
    parser.add_argument("metadata", help="CSV with clinical variables")
    parser.add_argument("analysis_name", help="Output subfolder name")
    parser.add_argument("--formula", default="JD ~ age + sex + session * outcome")
    parser.add_argument("--mask", required=True, help="MNI mask path")
    parser.add_argument("--n_jobs", type=int, default=10)
    args = parser.parse_args()

    # 1. Output Setup
    out_dir = os.path.join(args.bids_dir, "derivatives", "dbm", "stats", args.analysis_name)
    os.makedirs(out_dir, exist_ok=True)

    # 2. Metadata, Centering, and Provenance
    df = pd.read_csv(args.metadata, sep=None, engine='python')
    provenance = {
        "software": {"statsmodels": statsmodels.__version__, "nilearn": nilearn.__version__},
        "formula": args.formula,
        "variable_stats": {}
    }

    # Mean-centering 'age' to make the intercept biologically meaningful
    for var in ['age']:
        if var in df.columns:
            m, s = df[var].mean(), df[var].std()
            provenance["variable_stats"][var] = {"mean": m, "std": s, "centered": True}
            print(f"Centering {var}: Mean={m:.2f}, Std={s:.2f}")
            df[var] = df[var] - m

    # 3. File Discovery & Matching (Robust Glob bypasses BIDS indexing)
    search_pattern = os.path.join(args.bids_dir, "derivatives", "dbm", "sub-*", "ses-*", "anat", "*_space-MNI_desc-logJacobian_stat.nii.gz")
    jac_files = sorted(glob.glob(search_pattern))
    
    # Handle BIDS prefixes for session and subject IDs
    df['sub_match'] = df['subject_id'].apply(lambda x: f"sub-{x}" if "sub-" not in str(x) else str(x))
    df['ses_match'] = df['session'].apply(lambda x: f"ses-{x}" if "ses-" not in str(x) else str(x))

    matched_files, matched_metadata = [], []
    for f_path in jac_files:
        fname = os.path.basename(f_path)
        sub_bids, ses_bids = fname.split('_')[0], fname.split('_')[1]
        row = df[(df['sub_match'] == sub_bids) & (df['ses_match'] == ses_bids)]
        if not row.empty:
            matched_files.append(f_path)
            matched_metadata.append(row.iloc[0])

    temp_df = pd.DataFrame(matched_metadata)
    temp_df['file_list_idx'] = range(len(matched_files))

    # 4. Data Scrubbing & Synchronization
    formula_vars = args.formula.replace('~', '+').replace('*', '+').split('+')
    check_vars = [v.strip() for v in formula_vars if v.strip() in temp_df.columns]
    final_df = temp_df.dropna(subset=check_vars)
    
    # Sync brain images with clinical rows remaining after dropna
    final_files = [matched_files[i] for i in final_df['file_list_idx']]
    final_df = final_df.drop(columns=['file_list_idx']).reset_index(drop=True)
    provenance["n_final"] = len(final_df)

    # 5. Extraction & Parallel Computation
    masker = NiftiMasker(mask_img=args.mask).fit()
    voxel_data = masker.transform(final_files)
    n_voxels = voxel_data.shape[1]

    print(f"Processing {n_voxels} voxels | N={len(final_df)} | Jobs={args.n_jobs}")
    raw_results = Parallel(n_jobs=args.n_jobs)(
        delayed(run_voxel_mixedlm)(voxel_data[:, i], final_df, args.formula) 
        for i in tqdm(range(n_voxels), desc="Processing", ascii="○◔◑◕●", colour="#FEFD9b")
    )

    # 6. Post-Processing: Maps and CSV Cluster Report
    # Extract the interaction term (e.g., 'session:outcome') or last main effect
    target = args.formula.split('+')[-1].strip().replace('*', ':')
    t_map, b_map, p_map = [], [], []
    coords = masker.masker_analysed_coords_
    cluster_rows = []

    for i, res in enumerate(raw_results):
        if res and target in res['tvalues']:
            t, b, p = res['tvalues'][target], res['params'][target], res['pvalues'][target]
            t_map.append(t); b_map.append(b); p_map.append(p)
            cluster_rows.append({'X': coords[i][0], 'Y': coords[i][1], 'Z': coords[i][2], 't': t, 'beta': b, 'p': p})
        else:
            t_map.append(0); b_map.append(0); p_map.append(1)

    # False Discovery Rate (FDR) Correction
    _, p_fdr, _, _ = multipletests(p_map, method='fdr_bh')
    
    # Generate Cluster CSV
    full_stats = pd.DataFrame(cluster_rows)
    full_stats['p_fdr'] = p_fdr
    full_stats[full_stats['p_fdr'] < 0.05].to_csv(os.path.join(out_dir, "significant_voxels.csv"), index=False)

    # 7. Final Nifti Exports
    maps_to_save = {
        't_stat': t_map, 
        'beta': b_map, 
        'p_uncorrected': p_map, 
        'p_fdr': p_fdr
    }
    
    for name, data in maps_to_save.items():
        img = masker.inverse_transform(np.array(data))
        nib.save(img, os.path.join(out_dir, f"map_{name}.nii.gz"))

    # Save Provenance for Reproducibility
    with open(os.path.join(out_dir, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=4)

    print(f"Success. All maps, cluster CSV, and provenance saved to: {out_dir}")

if __name__ == "__main__":
    main()