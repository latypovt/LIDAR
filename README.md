# LIDAR: Longitudinal Imaging Deformation Analysis Repository

LIDAR is a BIDS-compatible pipeline designed for high-sensitivity longitudinal Deformation-Based Morphometry (DBM). It implements a two-level registration framework to isolate within-subject anatomical changes (e.g., VNS treatment effects) before mapping them to a common standard space.

## Core Features
* **Unbiased SST:** Creates a Subject-Specific Template (SST) as a midpoint to avoid longitudinal reference bias.
* **Defensive Execution:** Automatically skips subjects or sessions that have already been processed.
* **Parallelized Processing:** Multithreaded execution for both within-subject and group-level warping.
* **Robust Preprocessing:** Includes deep-learning brain extraction (ANTsPyNet) and Gibbs ringing correction (Dipy).

## Requirements
* Python 3.10+
* `antspyx`, `antspynet`, `dipy`, `pybids`

## Usage

### Run Full Pipeline
Executes Level 1 followed by Level 2 for a specific subject or the entire cohort.

```bash
python lidar_cli.py /path/to/bids --task all --subject 001 --mni_template /path/to/mni_template.nii.gz
```

### Level 1: Subject-Specific Processing
Generates the SST and calculates the log-Jacobian maps in the subject's native template space.
```bash
python lidar_cli.py /path/to/bids --task level1 --n_parallel 4 --itk_threads 4 
```

### Level 2: MNI Normalization
Warps the subject-level Jacobian maps into MNI space for group-level statistics.

```bash
python lidar_cli.py /path/to/bids --task level2 --mni_template /path/to/mni_template.nii.gz
```

### Modelling
Performs statistical modelling on the subject-level Jacobian maps.

```bash
python lidar_stats.py /path/to/bids source_csv_file analysis_name --formula "JD~variable1+variable2" --mask /path/to/mask.nii.gz --n_jobs 4
```
