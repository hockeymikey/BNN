## BNN - Vegetation classifcation with ConvNets

Set of all the work I did at UND OpenOrbiter REU program.
I was assigned to work on detection and classification of different types of vegetation in mosaics.
ConvNets were used, one being based on Stanford's cs231n class (stored in cs231n), the other being a Keras (tensorflow wrapper).
The results were dismal, only achiving at best ~40% accuracy and 60-70% in binary classification.

#### Layout of project:

1. cache - pickels and other files used for operations
2. cs231n - source of Stanford based ConvNet, called by VegSolver.py.  Has cpu and gpu versions.
3. data_prep - Scripts to extract and prepare data for ConvNets
	1. tif_utils.py - Tiff related functions
	2. moz_utils.py - Mosaic related functions
	3. img_extract.py - Extraction of the images from tiffs
	4. data_utils.py - Manages the images after extraction such as sorting or pickeling data
	5. lbl_utils.py - Extraction of label data from Excel files and related functions
	6. Practice - Testing new things
		* big_draw.py - Draws on each mosaic the extracted image as a red box, points as purple. Good for visual debugging image extraction.
4. logs - cs231n checkpoint logs from each training session
5. mnist - testing against mnist dataset
6. output - Extracted and converted files such as a csv of the points
7. Practice - Testing some random general scripts
8. tflow - Early tensorflow training against Veg data
9. tflow-testing - Later tensorflow, training against many different data sets including Veg data.
10. visual - Mosaics with visual drawing of extracted images from big_draw.py. .txt files are point locations/pixels
11. keras_binary.py - Tensorflow backend, trains binary against each class of dataset and stores results in txt file.
12. keras_training.py - Tensorflow backend, general classification of veg dataset.
13. prep_data.py - CMD UI for quickly extracting,sorting and pickeling veg data. Very basic atm, needs to be more advance.
14. Veg_solver.py - Main for cs231n ConvNet, training it against Veg dataset.