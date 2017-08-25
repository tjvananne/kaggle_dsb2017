
Very next step (as of 8/24/2017):
	1) download dicom files and start with part 2


-- Part 1 ----------------------------------------------------------------

The goal of part 1 is to take LUNA 16 data, and train an algorithm to identify where nodules are in the lungs. Once we have that algorithm, there is no more need for the LUNA 16 data. This team has provided us with the weights of that trained algorithm, so we don't even need to download the LUNA 16 data at all (unless the end user's patient's images are in the .mhd format, in which case we'll need a function to convert it to a dicom.

LUNA 16 data was used to create the model that identifies where the nodules are. LUNA 16 data comes in this weird MHD file format. Once the weights are identified for the nodule identifier, we no longer need to be able to process a mdh file unless that is what the files will look like the we will be doing predictions on at the very end.

There are too many files to mention that need to be run if you choose to do part 1 from scratch. For me, I'll just go ahead and use their pre-trained weights.


**** FINISHED WITH LUNA DATA ****

Output of part 1 is just this: cp -p ./nodule_identifiers/d8_2x2x2_best_weights.h5 luna/models
It would be unreasonable to assume I could train a better model than what they trained to identify nodules. It would be interesting to look into what they did and how, but that is not the goal at hand right now.




-- Part 2 ----------------------------------------------------------------

THIS IS WHERE WE'D START PROCESSING NEW DATA FOR REAL LIFE PREDICTIONS

The goal of part two is to segment the lungs in the Kaggle dicom (stage 1 / stage 2) data. 

Kaggle stage 1 and stage 2 data should be in dicom format. These will not require any pre-processing at the MHD level. 


Then we'll need to run unet_d8g_222f.py (part 2 only) which is the "segment_all()" function on the KAGGLE data. I guess the output here will be the same image files except for the images will be segmented?
 


-- Part 3 -----------------------------------------------------------------

Identify nodule masks using the luna trained model from part 1. Then generate all of the features that will be pushed into the final prediction stage (ensemble learner).

Here we'll run (preferably in spyder) lungs_var3_d8g_222f.py 












