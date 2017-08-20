
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

from unet_d8g_222f import *


"""
# the things at the top of the file we're investigating
RESIZE_SPACING = [2,2,2]  # z, y, x  (x & y MUST be the same)
RESOLUTION_STR = "2x2x2"

img_rows = 448 
img_cols = 448 # global values

DO_NOT_USE_SEGMENTED = True

#STAGE = "stage1"

STAGE_DIR_BASE = "../input/%s/"  # on one cluster we had input_shared

LUNA_MASKS_DIR = "../luna/data/original_lung_masks/"
luna_subset = 0       # initial 
LUNA_BASE_DIR = "../luna/data/original_lungs/subset%s/"  # added on AWS; data as well 
LUNA_DIR = LUNA_BASE_DIR % luna_subset

CSVFILES = "../luna/data/original_lungs/CSVFILES/%s"
LUNA_ANNOTATIONS = CSVFILES % "annotations.csv"
LUNA_CANDIDATES =  CSVFILES % "candidates.csv"
"""



part = 0
processors = 1          # you may run several of these jobs; define processors to say 4, and start 4 separate jobs with part = 0, 1, 2, 3 respectively
showSummaryPlot = True
stage = "stage1" 

graphviz = GraphvizOutput(output_file='filter_none.png')

with PyCallGraph(output=graphviz):
    # for stage in ["stage1", "stage2"]:     
    # start_time = time.time()
    # print ("Starting segmentation, stage, part of a set/processors: ", stage, part, processors)
    part, processors, count = segment_all(stage, part, processors, showSummaryPlot)
    # print ("Completed, part, processors,total count tried, total time: ", stage, part, processors, count, time.time()-start_time)
	
