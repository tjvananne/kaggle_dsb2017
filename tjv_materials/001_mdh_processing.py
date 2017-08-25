import numpy as np
import pandas as pd
import SimpleITK as sitk
import os



def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    
    # Convert the image
    ct_scan = sitk.GetArrayFromImage(itkimage)
    
    # Read the origin of the ct_scan
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    
    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    
    return ct_scan, origin, spacing


# for every item ("f") in this dir, return it if it ends with ".mhd"
mhds = [f for f in os.listdir() if f.endswith(".mhd")]


this_file = mhds[0]
print(this_file)


this_scan, this_origin, this_spacing = load_itk(this_file)

