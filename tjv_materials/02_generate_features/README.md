

# PHASE 3: Generate Features (still Andre code)


**Recap**
* Phase 1 was done for us, all we need to do is copy over their model weights for identification of nodules in the lungs
* Phase 2 was to process raw dicom files into processed, segmented, numpy ndarrays representing the cropped lungs of the patient
* Phase 3 is to use those segmented ndarrays from phase 2 to build features at the **patient level (this is what needs to be refactored to the nodule level)** 



** output files written to disk from phase 2: **
* `{PROJECT ROOT}/input/stage1_2x2x2/{patient id}.npz`
* `{PROJECT ROOT}/input/stage1_segmented_2x2x2/{patient_id}.npz`
* `{PROJECT ROOT}/input/stage1_segmented_2x2x2_crop/{patient id}.npz`



## Project directory


The items in **bold** are new relative to the previous checkpoint of deep-diving the project (001)


* **{PROJECT ROOT}**
	- py_script (name this whatever)
		* 001_read_segment_kaggle_data_scans.py - functions were taken from `unet_d8g_222f.py` in original repo
		* **002_generate_features.py** - functions to be taken from `lungs_var3_d8g_222f.py` in original repo
	- input
		* stage1/   (manually create this dir and place the `.dcm` dicom files here)
			- {patient one dir name}/
				* {image id}.dcm
				* {image id}.dcm
				* (there are ~200 of these images per patient id)
			- {patient two dir name}/
				* {image id}.dcm
				* ...
		* stage_2x2x2/  (manually create this dir and leave it empty)
			- these are produced by the 001 python script in py_script directory
			- {patient one id}.npz
			- {patient two id}.npz
		* stage_segmented_2x2x2/  (manually create this dir and leave it empty)
			- these are produced by the 001 python script in py_script directory
			- {patient one id}.npz
			- {patient two id}.npz
		* stage_segmented_2x2x2_crop/   (manually create this dir and leave it empty)
			- {patient one id}.npz
			- {patient two id}.npz
	- **luna/**
		* **models/**
			- **d8_2x2x2_best_weights.h5** (pre-trained model weights used to identify/isolate nodules in lungs

			
