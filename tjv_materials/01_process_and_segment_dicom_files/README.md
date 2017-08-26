

# PHASE 2: process and segment dicom files

Phase 1 was training an algorithm to identify nodules using the LUNA 16 dataset. The original authors of this project give the reader three different choices for how they'd like to implement this same functionality:

1) simply copy the pre-trained model weights provided in their repository (stored in a .h5 file) 
	- **this is the option I'm choosing** - it is the simplest and I don't think I could train the algorithm better than these people did
2) retrain the nodule identifier model from the pieces they provide
3) build the model architecture, preprocessing techniques, and model parameters from scratch using only the LUNA data


Because I chose the first option above, I can safely assume that my starting point in this project will be from the dicom files onward.


This sub directory shows the process of taking in the dicom files (Kaggle, in this instance), normalizing and preprocessing the images, and segmenting/isolating the lungs. This data is then stored for each patient in various versions of .npz files. 

Dicom files are read in from the `<PROJECT ROOT>/input/stage1/<patient id dir>/< ~200 image id>.dcm` location. One they are processed as numpy ndarrays, they are written out to disk in these directories:


* `<PROJECT ROOT>/input/stage1_2x2x2/<patient id>.npz`
* `<PROJECT ROOT>/input/stage1_segmented_2x2x2/<patient_id>.npz`
* `<PROJECT ROOT>/input/stage1_segmented_2x2x2_crop/<patient id>.npz`


I haven't really explored the differences between these three types of .npz outputs listed in different directories above, but I think that will become much more clear once I look at the next stage of the project as a whole. 


* **<PROJECT ROOT>**
	- py_script (name this whatever)
		* **001_read_segment_kaggle_data_scans.py** - functions were taken from `unet_d8g_222f.py` in original repo
	- input
		* stage1/   (manually create this dir and place the `.dcm` dicom files here)
			- <patient one dir name>/
				* <image id>.dcm
				* <image id>.dcm
				* (there are ~200 of these images per patient id)
			- <patient two dir name>/
				* <image id>.dcm
				* ...
		* stage_2x2x2/  (manually create this dir and leave it empty)
			- **these are produced by the 001 python script in py_script directory**
			- <patient one id>.npz
			- <patient two id>.npz
		* stage_segmented_2x2x2/  (manually create this dir and leave it empty)
			- **these are produced by the 001 python script in py_script directory**
			- <patient one id>.npz
			- <patient two id>.npz
		* stage_segmented_2x2x2_crop/   (manually create this dir and leave it empty)
			- <patient one id>.npz
			- <patient two id>.npz

			
