# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 15:17:11 2017

@author: tvananne

pulling in functions/logic from lungs_var3_d8g_222f.py in original repo

"""


print("Load modules...")

import numpy as np
import dicom
import glob
from matplotlib import pyplot as plt
import os
import time

import pandas as pd

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage # added for scaling\

from keras.models import load_model,Model
from keras.layers import Input, merge, UpSampling3D

from keras.layers import Convolution3D, MaxPooling3D # added for 3D

from keras.optimizers import Adam
from keras import backend as K

from scipy import stats

from scipy.stats import gmean
import gc

print("Done loading modules...")





# script constants -------------------------------------------------------------------------------------


RESIZE_SPACING = [2,2,2]
RESOLUTION_STR = "2x2x2"

ALT_WORKSTATION = ""  # "_shared"  # could be _shared on one of our clusters (empty on AWS)

STAGE_DIR_BASE = ''.join(["../input", ALT_WORKSTATION, "/%s/" ])                        # to be used with % stage
LABELS_BASE = ''.join(["../input", ALT_WORKSTATION, "/%s_labels.csv"])                  # to be used with % stage
#SAMPLE_SUBM_BASE = ''.join(["../input", ALT_WORKSTATION, "/%s_sample_submission.csv"]) # to be used with % stage

smooth = 1.






# function defs ----------------------------------------------------------------------------------------



def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)




def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)





def unet_model_xd3_2_6l_grid(nb_filter=48, dim=5, clen=3 , img_rows=None, img_cols=None ):
    
    #aiming for architecture as in http://cs231n.stanford.edu/reports2016/317_Report.pdf
    #The model is eight layers deep, consisting  of  a  series  of  three  CONV-RELU-POOL  lay- ers (with 32, 32, and 64 3x3 filters), a CONV-RELU layer (with 128 3x3 filters), three UPSCALE-CONV-RELU lay- ers (with 64, 32, and 32 3x3 filters), and a final 1x1 CONV- SIGMOID layer to output pixel-level predictions. Its struc- ture resembles Figure 2, though with the number of pixels, filters, and levels as described here

    ## 3D CNN version of undet_model_xd_6j 
    zconv = clen
    
    inputs = Input((1, dim, img_rows, img_cols))
    conv1 = Convolution3D(nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution3D(nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Convolution3D(2*nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution3D(2*nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)


    conv4 = Convolution3D(4*nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(pool2)
    conv4 = Convolution3D(4*nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(conv4)

    up6 = merge([UpSampling3D(size=(2, 2, 2))(conv4), conv2], mode='concat', concat_axis=1)
    conv6 = Convolution3D(2*nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(up6)
    conv6 = Convolution3D(2*nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(conv6)

        
    up7 = merge([UpSampling3D(size=(2, 2, 2))(conv6), conv1], mode='concat', concat_axis=1)  # original - only works for even dim 
    conv7 = Convolution3D(nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(up7)
    conv7 = Convolution3D(nb_filter, zconv, clen, clen, activation='relu', border_mode='same')(conv7)

    pool11 = MaxPooling3D(pool_size=(2, 1, 1))(conv7)

    conv12 = Convolution3D(2*nb_filter, zconv, 1, 1, activation='relu', border_mode='same')(pool11)
    conv12 = Convolution3D(2*nb_filter, zconv, 1, 1, activation='relu', border_mode='same')(conv12)
    pool12 = MaxPooling3D(pool_size=(2, 1, 1))(conv12)

    conv13 = Convolution3D(2*nb_filter, zconv, 1, 1, activation='relu', border_mode='same')(pool12)
    conv13 = Convolution3D(2*nb_filter, zconv, 1, 1, activation='relu', border_mode='same')(conv13)
    pool13 = MaxPooling3D(pool_size=(2, 1, 1))(conv13)

    if (dim < 16):
        conv8 = Convolution3D(1, 1, 1, 1, activation='sigmoid')(pool13)
    else:   # need one extra layer to get to 1D x 2D mask ...
            conv14 = Convolution3D(2*nb_filter, zconv, 1, 1, activation='relu', border_mode='same')(pool13)
            conv14 = Convolution3D(2*nb_filter, zconv, 1, 1, activation='relu', border_mode='same')(conv14)
            pool14 = MaxPooling3D(pool_size=(2, 1, 1))(conv14)
            conv8 = Convolution3D(1, 1, 1, 1, activation='sigmoid')(pool14)        

    model = Model(input=inputs, output=conv8)

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])
    #model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),  loss=dice_coef_loss, metrics=[dice_coef])

    return model







def calc_features_keras_3dx(stage, dim, run, processors, model_weights_name):
    
    STAGE_DIR = STAGE_DIR_BASE % stage
    LABELS = LABELS_BASE % stage
    
    start_from_model_weights = True
    if start_from_model_weights:
        model = unet_model_xd3_2_6l_grid(nb_filter=20, dim=dim, clen=3, img_rows=None , img_cols=None )   
       
        model.load_weights(model_weights_name)
        print(model.summary())
        print("Using Weights: ", model_weights_name)
   
    else:
       model_name = "SPECIFY_MODE_NAME_IF_WHEN_NEEDED"   #Warning, hard-wired, modify if/as needed
       model = load_model(model_name, 
                custom_objects={'dice_coef_loss': dice_coef_loss,
                                           'dice_coef': dice_coef
                                           }
                       )
       print(model.summary())
       print("Loaded model: ", model_name)
       
    ## obtain cancer labels for quick validation
    labels = pd.read_csv(LABELS) ### was data/
    #print(labels.head())


    source_data_name = ''.join([stage, "_", RESOLUTION_STR ])
    source_data_name_seg = ''.join([stage, "_segmented_", RESOLUTION_STR ])
    
    stage_dir_segmented = STAGE_DIR.replace(stage, source_data_name, 1) 
    
    files = glob.glob(os.path.join(stage_dir_segmented,'*.npz'))
    
    add_missing_entries_only = False
    if add_missing_entries_only:

        colnames = ["fn"]
        files_in = pd.read_csv("./files_in_222.csv", names=colnames)
        files = files_in.fn.tolist()[1:]
        len(files)
        path = files[2]  # 4 is a cancer, 2 is not but may have a lots of false positives 
        start_file = 1300 #0 #2        # was 508 for a restart
        last_file = len(files)  #
    else:
        start_file = 0 #2        # was 508 for a restart
        last_file = len(files)  #

    path = files[2]  # 4 is a cancer, 2 is not but may have a lots of false positives 
    #path = "../input_shared/stage1_2x2x2/21cfe28c4a891795399bb77d63e939d7.npz"
    count = start_file
    cnt = 0
    frames = []
    
    use_single_pass = True
    if not use_single_pass:
        grid=60  # orig 60
        crop=44  # orig 44
        expand=10  # orig 10
    else:
        grid=360/RESIZE_SPACING[1]  
        crop=44/RESIZE_SPACING[1]  # orig 44
        #expand=0  # orig 10  ---> irrelevant for this option    
    
    for path in files[start_file:last_file]:

        uid =  path[path.rindex('/')+1:] 
        uid = uid[:-4]

        if (uid in labels["id"].tolist()):
            cancer = int(labels["cancer"][labels["id"] == uid])
        else:
            cancer = -777
        
        count += 1
    
        
        if count % processors == run:    # do this part in this process, otherwise skip
        #for folder in mask_wrong:
        
  
            
            start_time = time.time()
            
            path_seg = path.replace(source_data_name, source_data_name_seg, 1)
            if RESIZE_SPACING[1] < 2:   # compatibility setting  for the 2 versions supported *8 and 16 layers)
                by_min, by_max, bx_min, bx_max, mask = find_roi_bb(path_seg)
            else:
                mask = np.load(''.join((path_seg, "")))['arr_0']
                by_min = mask.shape[1] % 4 
                bx_min = mask.shape[2] % 4
                bx_max = mask.shape[2]
                by_max = mask.shape[1]
  
            area_reduction = (by_max-by_min)*(bx_max-bx_min)/(mask.shape[1]*mask.shape[2])
            by_len = by_max - by_min
            bx_len = bx_max - bx_min
            print ("count, cancer, uid, by_min, by_max, bx_min, bx_max, height, width, area_reduction: ", count, cancer, uid, by_min, by_max, bx_min, bx_max, by_max-by_min, bx_max-bx_min, area_reduction)
            print ("geom new size y x: ", by_len, bx_len )
            mask = mask[:, by_min:by_max, bx_min:bx_max]  # unused as yet here - just for testing

            
            divisor = 4
            if (by_len % divisor > 0) or (bx_len % divisor > 0):
                print ("WARNING: for uid, by_len or bx_len not multiple of: ", uid, by_len, bx_len, divisor)
                    
           
            testPlot = False
            if testPlot:
                   # Show some slice in the middle
                plt.imshow(np.mean(mask, axis=0), cmap=plt.cm.gray)
                plt.show()
            #start_time = time.time() 
            images3 = get_segmented_preprocess_bb_xd_file(path, by_min, by_max, bx_min, bx_max, dim).astype(np.float32)  # added 20160307
            images3 = images3[:,np.newaxis]  # add a dimension for the 3 D model
       
            images3_seg = get_segmented_preprocess_bb_xd_file(path_seg, by_min, by_max, bx_min, bx_max, dim).astype(np.float32)  # added 20160307
            images3_seg = images3_seg[:,np.newaxis]  # add a dimension for the 3 D model

            
            if not use_single_pass:
                scans, gridwidth, gridheight = grid_data(images3, grid=grid, crop=crop, expand=expand)
            else:
                # juts crop the data
                gridwidth = gridheight = 1
                #scans = images3[:,:,:, crop:-crop,crop:-crop]
                scans = images3

                     
            pmasks =  model.predict(scans, batch_size =1, verbose=0)  # batch_size = 1 seems to be 10% faster than any of the 2, 3, 4, 8 (e.g. 66s vs 73-74 sec)
            #pmasks =  model.predict(scans, verbose=1)
               
               
            if not use_single_pass:
                pmasks3 = data_from_grid (pmasks, gridwidth, gridheight, grid=grid)
            else:
                pmasks3 = pmasks
            
            if use_single_pass:
            # now crop the images3 to the size of pmasks3 for simplicity ...
                nothing_to_do = 1  #  # already cropped through the bb    
            else:
                images3 = images3[:,:,:, crop:-crop,crop:-crop]
           
            path_mpred = path.replace(source_data_name, (source_data_name + "_mpred3_%s") % str(dim), 1)
            pmasks3[pmasks3 < 0.5] =0        # zero out eveything below 0.5 -- this should reduce the file size 
            np.savez_compressed (path_mpred, pmasks3)
            
            bb = pd.DataFrame(
                    {"by_min":  by_min,
                     "by_max":  by_max,
                     "bx_min":  bx_min,
                     "bx_max":  bx_max
                     },
                     index=[uid])
            bb.to_csv( path_mpred[:-4] + ".csv", index=True)
            
            
            read_bb_back = False
            if read_bb_back:
                print ("by_min, by_max, bx_min, bx_max: ", by_min, by_max, bx_min, bx_max)
                bb = pd.read_csv(path_mpred[:-4] + ".csv", index_col = 0) #to check
                ### use uid to ensure errot checking and also consolidation of the bb's if need to
                by_min = bb.loc[uid].by_min
                by_max = bb.loc[uid].by_max
                bx_min = bb.loc[uid].bx_min
                bx_max = bb.loc[uid].bx_max
                                
            
            dim_out = pmasks3.shape[2]
            
            #        reduce the images and pmasks to 2 dimension sets
    
            testPlot = False
            j_pred = dim // 2
            if testPlot:
                i=31  # for tests
                for i in range(images3.shape[0]):
                    j = 0
                    for j in range(j_pred, j_pred+1):  # we use the dim cut for results
                        img = images3[i, 0, j]
                        #pmsk = pmasks3[i,0,j]  # for multi-plane output
                        pmsk = pmasks3[i,0,0]
                        pmsk_max = np.max(pmsk)
                        #print ('scan '+str(i))
                        print ('scan & max_value: ', i, pmsk_max)
                        if pmsk_max > 0.99:
                            f, ax = plt.subplots(1, 2, figsize=(10,5))
                            ax[0].imshow(img,cmap=plt.cm.gray)
                            ax[1].imshow(pmsk,cmap=plt.cm.gray)
                            ####ax[2].imshow(masks1[i,:,:],cmap=plt.cm.gray)
                            plt.show()
                            #print ("max =", np.max(pmsk))
            
            #### list solution for the several rolls that also reduce the memory requirements as we only store the relevant cut ...
            
            if dim_out > 1:
                pmaskl = [np.roll(pmasks3, shift, axis =0)[dim:,0,shift] for shift in range(dim)]  # pmask list (skipping the first dim elements)
                pmav = np.min(pmaskl, axis = 0)
                ### also adjust the images3 -- skipp the first dim layers 
                images3 = images3[dim:0]
            else:
                pmav = pmasks3[:,0,0]
   
            
            if testPlot:
                for i in range(pmasks3.shape[0]):
                    j = 0
                    for j in range(j_pred, j_pred+1):
                        #img = images3[i, 0, j]
                        
                        print ('scan '+str(i))
                        f, ax = plt.subplots(1, 2, figsize=(10,5))
                        ax[0].imshow(images3[i, 0, j],cmap=plt.cm.gray)
                        ax[1].imshow(pmav[i],cmap=plt.cm.gray)

                dist = pmav.flatten()
                thresh = 0.005
                dist = dist[(dist > thresh) & (dist < 1-thresh)]
                plt.hist(dist, bins=80, color='c')
                plt.xlabel("Nodules prob")
                plt.ylabel("Frequency")
                plt.show()
                print(len(dist))  #2144 for pmav dist , 3748 for pm , 1038 for pm[:,0,0], 627 for pm[:,0,1], 770 for pm[:,0, 2], 1313 for for pm[:,0, 3]; 1129 for pmav2 - peak around 0.5 where it was not there before
                
            part=0
            frames0 = []
            frames1 = []
            zsel = dim // 2  # use the mid channel/cut (could use any other for the stats and trhey are repeated)
            segment = 2
            for segment in range(3):
                # 0 = top part
                # 1 bottom half
                # 2 all
                if segment == 0:
                    sstart = 0
                    send = images3.shape[0] // 2
                elif segment == 1:
                    sstart = images3.shape[0] // 2
                    send = images3.shape[0]
                else:                   ### the last one must be the entire set
                    sstart = 0
                    send = images3.shape[0]
        

                ims = images3[sstart:send,0,zsel]      # selecting the zsel cut for nodules calc ...
                ims_seg = images3_seg[sstart:send,0,zsel] 
                #pms = pmasks3[sstart:send,0,0]
                pms = pmav[sstart:send]
                
                # threshold the precited nasks ...
                
                #for thresh in [0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999]:
                for thresh in [0.75, 0.9999999, 0.99999]:
 
                    
                    idx = pms > thresh
                    nodls = np.zeros(pms.shape).astype(np.int16)
                    nodls[idx] = 1
                    
                    nx =  nodls[idx]
                    # volume = np.sum(nodls)  # counted as a count within hu_describe ...
                    nodules_pixels = ims[idx]   # flat
                    nodules_hu = pix_to_hu(nodules_pixels)
                    part_name = ''.join([str(segment), '_', str(thresh)])
                    #df = hu_describe(nodules_hu, uid, 1)
                    df = hu_describe(nodules_hu, uid=uid, part=part_name)
                    
                    ### add any additional params to df ...
                    #sxm = ims * nodls
                    add_projections = False
                    axis = 1
                    nodules_projections = []
                    for axis in range(3):
                         #sxm_projection = np.max(sxm, axis = axis)
                         nodls_projection = np.max(nodls, axis=axis)
                         naxis_name = ''.join(["naxis_", str(axis),"_", part_name])
                         if add_projections:   
                             df[naxis_name] = np.sum(nodls_projection)
                         nodules_projections.append(nodls_projection)
                    
                    ## find the individual nodules ... as per the specified probabilities 
                    labs, labs_num = measure.label(idx, return_num = True, neighbors = 8 , background = 0)  # label the nodules in 3d, allow for diagonal connectivity
                    if labs_num > 0:                  
                        regprop = measure.regionprops(labs, intensity_image=ims)
                        areas = [rp.area for rp in regprop]
                        #ls = [rp.label for rp in regprop]
                        max_val = np.max(areas)
                        max_index = areas.index(max_val)
                        max_label = regprop[max_index].label
                        #max_ls = ls[max_index]
                        zcenters = [(r.bbox[0]+r.bbox[3])/2  for r in regprop]
                        zweighted = sum(areas[i] * zcenters[i] for i in range(len(areas))) / sum(areas)
                        zrel = zweighted / ims.shape[0]
                        
                        idl = labs ==  regprop[max_index].label   
                        nodules_pixels = ims[idl]
                        nodules_hu = pix_to_hu(nodules_pixels)
                    else:
                        nodules_hu = []
                        zrel = -777
                    part_name = ''.join([str(segment), '_', str(thresh),'_n1'])
                    df2 = hu_describe(nodules_hu, uid=uid, part=part_name) 
                    zrel_name = ''.join(["zrel_", part_name])
                    df2[zrel_name] = zrel
                    count_name = ''.join(["ncount_", str(segment), '_', str(thresh)])
                    df2[count_name] = labs_num
                       
                    df3 = pd.concat( [df, df2], axis =1)
                    
                    frames0.append(df3)
                    part += 1
                
                dfseg = pd.concat(frames0, axis=1, join_axes=[frames0[0].index])  # combine rows
                
                 ##### add any section features independent of nodules ....
                # estimate lung volume by counting all relevant pixels using the segmented lungs data
                # for d16g do it for the 3 different areas, including the emphysema calcs
                
                
                HU_LUNGS_MIN0 = -990  # includes emphysema 
                HP_LUNGS_EMPHYSEMA_THRESH = -950
                HP_LUNGS_EMPHYSEMA_THRESH2 = -970
                HU_LUNGS_MAX = -400
    
                pix_lungs_min = hu_to_pix(HU_LUNGS_MIN0)
                pix_lungs_max = hu_to_pix(HU_LUNGS_MAX)

                idv = (ims_seg >pix_lungs_min) & (ims_seg < pix_lungs_max)
                idv_emphysema  =  (ims_seg >hu_to_pix(HP_LUNGS_EMPHYSEMA_THRESH))  & (ims_seg < pix_lungs_max)
                idv_emphysema2 =  (ims_seg >hu_to_pix(HP_LUNGS_EMPHYSEMA_THRESH2)) & (ims_seg < pix_lungs_max)
                
                test_emphysema = False
                if test_emphysema:
                    
                    f, ax = plt.subplots(1, 4, figsize=(20,5))
                    ax[0].imshow(idv[:,idv.shape[1]//2,:],cmap=plt.cm.gray)
                    ax[1].imshow(idv[:,idv.shape[1]//2,:],cmap=plt.cm.gray)
                    #ax[1].imshow(sxm_projection_1,cmap=plt.cm.gray)
                    ax[2].imshow(ims[:,ims.shape[1]//2, :],cmap=plt.cm.bone)
                    ax[3].imshow(ims_seg[:,ims_seg.shape[1]//2, :],cmap=plt.cm.bone)
                    
                    #ax[2].imshow(sxm_projection_2,cmap=plt.cm.gray)
                    plt.show()
      
                    
                    idv_all = []
                    hu_lungs_min_all=[]
                    for HU_LUNGS_MIN in range(-1000, -800, 10):
                        pix_lungs_min = hu_to_pix(HU_LUNGS_MIN)
                        pix_lungs_max = hu_to_pix(HU_LUNGS_MAX)

                        idv = (ims_seg >pix_lungs_min) & (ims_seg < pix_lungs_max)
                        idv = (ims >pix_lungs_min) & (ims < pix_lungs_max)
                        idv_all.append(np.sum(idv))
                        hu_lungs_min_all.append(HU_LUNGS_MIN)
                    e_results = pd.DataFrame(
                        {"HU min": hu_lungs_min_all,
                         "volume approx": np.round(idv_all, 4)
                         })
                    plt.figure()
                    e_results.plot()
                    e_results.plot(kind='barh', x='HU min', y='volume approx', legend=False, figsize=(6, 10))
                        
                        
                lvol = np.sum(idv)
                       
                dfseg[''.join(['lvol', '_', str(segment)])] =  lvol   # 'lvol_2' etc.
                dfseg[''.join(['emphy', '_', str(segment)])] = (lvol - np.sum(idv_emphysema))/(lvol+1)
                dfseg[''.join(['emphy2', '_', str(segment)])] = (lvol - np.sum(idv_emphysema2))/(lvol+1)
        
                frames1.append(dfseg)
                
        
            df = pd.concat(frames1, axis=1, join_axes=[frames1[0].index])  # combine rows
            
            ## use the most recent threshold as a test ...
           
            testPlot2 = True
            #if testPlot2 and ims.shape[0] > ims.shape[1]:        
            if testPlot2:        
                print ("Summary, Count, Cancer, labs_num (nodules count 0), uid, process time: ", count, cancer, labs_num, uid, time.time()-start_time)
 
                if ims.shape[0] > ims.shape[1]:
                    print ("Suspicious hight, shape: ", images3.shape, uid)
                f, ax = plt.subplots(1, 4, figsize=(20,5))
                ax[0].imshow(nodules_projections[0],cmap=plt.cm.gray)
                ax[1].imshow(nodules_projections[1],cmap=plt.cm.gray)
                #ax[1].imshow(sxm_projection_1,cmap=plt.cm.gray)
                ax[2].imshow(ims[:,ims.shape[1]//2, :],cmap=plt.cm.bone)
                ax[3].imshow(ims_seg[:,ims_seg.shape[1]//2, :],cmap=plt.cm.bone)
                
                #ax[2].imshow(sxm_projection_2,cmap=plt.cm.gray)
                plt.show()
  
                
            ### calculate the volume in the central sections (say 60% - 5 sectors)  .-- may be capturing circa 90% of total volume for some (trying to reduce impact of outside/incorect segmetations??)..
            sections = 5
            for sections in range(3,7):       
                zstart = ims_seg.shape[0] // sections
                zstop = (sections -1) * ims_seg.shape[0] // sections
                ims_part = ims_seg[zstart:zstop]
                
                #idv = (ims_part > 0.9 * np.min(ims_part)) & (ims_part < 0.9 * np.max(ims_part))
                idv = (ims_seg >pix_lungs_min) & (ims_seg < pix_lungs_max)
          
                #df["lvol_sct"] =  np.sum(idv) 
                df["lvol_sct%s" % sections] =  np.sum(idv) 
                
                #df["lvol"]
                #df["lvol_sct"] /  df["lvol"]
            
            
            df["cancer"] = cancer
    
            
            testPrint = False
            if testPrint:
                dfeasy = np.round(df, 2)
                if cnt == 0:
                    print (dfeasy)
                    #print(dfeasy.to_string(header=True))
                else:
                    print(dfeasy.to_string(header=False))
         
            cnt += 1
            frames.append(df)
            del(images3)
            del(images3_seg)
            del(mask)
            del(nodls)
            del(pmasks3)
            if (cnt % 10 == 0):
                print ("Scans processed: ", cnt)
                gc.collect()  # looks that memory goes up /leakage somwehere
           
          
    result = pd.concat(frames)
    
    return result














# main() program right here: --------------------------------------------------------------------------

stage = "stage1"
feats = []
feats1 = []
feats2 = []


dim = 8
run = 0
processors = 1


batch_size = 1 # was 36
#count, cnt = calc_keras_3d_preds(dim, run, processors, batch_size)

date_version = "0411x"       # set as per the final submission 

make_predictions = True  # you may wish to set it False once the nodules have beein identified, as this step is time consuming
if make_predictions:
  for stage in ["stage1", "stage2"]:
    start_time = time.time()
    
    ### REDEFINE the nodule identifier to be used if/as needed, as per the ReadMe.txt description  (by commenting out only the relevant option)
    model_weights_name = "../luna/models/d8_2x2x2_best_weights.h5"  #  Option 1

    #model_weights_name = "../luna/models/d8g_bre_weights_50.h5"    #  Option 2
    #model_weights_name = "../luna/models/d8g4a_weights_74.h5"      #  Option 3
    
    feats = calc_features_keras_3dx(stage, dim, run, processors, model_weights_name)
    
    fname = 'feats_base_8_%s_%s_%s.csv'% (stage, len(feats), date_version)
    print ("OVERALL Process time, predictions & base features: ", stage, fname, time.time()-start_time)
    feats.to_csv(fname, index=True)
    

# Create 3 features files, starting from the most recent one, and 2 compatible with the March calculations
stage = "stage2"
for stage in ["stage1", "stage2"]:
    start_time = time.time()
    
    feats3 = recalc_features_keras_3dx(stage, dim, run, processors, withinsegonly= False, valonly=False)
    
    fname3 = 'feats_keras8_0313_%s_%s_%s.csv'% (stage, len(feats1), date_version)
    feats3.to_csv(fname3, index=True)
    print ("OVERALL feature file and process: ", stage, fname3, time.time()-start_time)
    print ("Validation: Any NaNs in features? ", feats3.isnull().values.any())
    

    
    start_time = time.time()
    
    feats1 = recalc_features_keras_3dx_0313(stage, dim, run, processors, withinsegonly= False, valonly=False)
    
    fname1 = 'feats_keras8_0313_%s_%s_%s.csv'% (stage, len(feats1), date_version)
    feats1.to_csv(fname1, index=True)
    print ("OVERALL feature file and process: ", stage, fname1, time.time()-start_time)
    print ("Validation: Any NaNs in features? ", feats1.isnull().values.any())

    feats_not_in_0311_version = {'xcenter_0_0.5_n0_0', 'xcenter_0_0.5_n1_0', 'xcenter_0_0.5_n2_0', 'xcenter_0_0.5_n3_0', 'xcenter_0_0.5_n4_0', 'xcenter_0_0.75_n0_0', 'xcenter_0_0.75_n1_0', 'xcenter_0_0.75_n2_0',
         'xcenter_0_0.75_n3_0', 'xcenter_0_0.75_n4_0', 'xcenter_0_0.95_n0_0', 'xcenter_0_0.95_n1_0', 'xcenter_0_0.95_n2_0', 'xcenter_0_0.95_n3_0', 'xcenter_0_0.95_n4_0', 'xcenter_0_0.98_n0_0', 'xcenter_0_0.98_n1_0',
         'xcenter_0_0.98_n2_0', 'xcenter_0_0.98_n3_0', 'xcenter_0_0.98_n4_0', 'xcenter_0_0.9999999_n0_0', 'xcenter_0_0.9999999_n1_0', 'xcenter_0_0.9999999_n2_0', 'xcenter_0_0.9999999_n3_0', 'xcenter_0_0.9999999_n4_0',
         'xcenter_0_0.999999_n0_0', 'xcenter_0_0.999999_n1_0', 'xcenter_0_0.999999_n2_0', 'xcenter_0_0.999999_n3_0', 'xcenter_0_0.999999_n4_0', 'xcenter_0_0.99999_n0_0', 'xcenter_0_0.99999_n1_0', 'xcenter_0_0.99999_n2_0',
         'xcenter_0_0.99999_n3_0', 'xcenter_0_0.99999_n4_0', 'xcenter_0_0.9999_n0_0', 'xcenter_0_0.9999_n1_0', 'xcenter_0_0.9999_n2_0', 'xcenter_0_0.9999_n3_0', 'xcenter_0_0.9999_n4_0', 'xcenter_0_0.999_n0_0',
         'xcenter_0_0.999_n1_0', 'xcenter_0_0.999_n2_0', 'xcenter_0_0.999_n3_0', 'xcenter_0_0.999_n4_0', 'xcenter_0_0.99_n0_0', 'xcenter_0_0.99_n1_0', 'xcenter_0_0.99_n2_0', 'xcenter_0_0.99_n3_0', 'xcenter_0_0.99_n4_0',
         'xcenter_0_0.9_n0_0', 'xcenter_0_0.9_n1_0', 'xcenter_0_0.9_n2_0', 'xcenter_0_0.9_n3_0', 'xcenter_0_0.9_n4_0', 'xcenter_1_0.5_n0_0', 'xcenter_1_0.5_n1_0', 'xcenter_1_0.5_n2_0', 'xcenter_1_0.5_n3_0',
         'xcenter_1_0.5_n4_0', 'xcenter_1_0.75_n0_0', 'xcenter_1_0.75_n1_0', 'xcenter_1_0.75_n2_0', 'xcenter_1_0.75_n3_0', 'xcenter_1_0.75_n4_0', 'xcenter_1_0.95_n0_0', 'xcenter_1_0.95_n1_0', 'xcenter_1_0.95_n2_0',
         'xcenter_1_0.95_n3_0', 'xcenter_1_0.95_n4_0', 'xcenter_1_0.98_n0_0', 'xcenter_1_0.98_n1_0', 'xcenter_1_0.98_n2_0', 'xcenter_1_0.98_n3_0', 'xcenter_1_0.98_n4_0', 'xcenter_1_0.9999999_n0_0', 'xcenter_1_0.9999999_n1_0',
         'xcenter_1_0.9999999_n2_0', 'xcenter_1_0.9999999_n3_0', 'xcenter_1_0.9999999_n4_0', 'xcenter_1_0.999999_n0_0', 'xcenter_1_0.999999_n1_0', 'xcenter_1_0.999999_n2_0', 'xcenter_1_0.999999_n3_0', 'xcenter_1_0.999999_n4_0',
         'xcenter_1_0.99999_n0_0', 'xcenter_1_0.99999_n1_0', 'xcenter_1_0.99999_n2_0', 'xcenter_1_0.99999_n3_0', 'xcenter_1_0.99999_n4_0', 'xcenter_1_0.9999_n0_0', 'xcenter_1_0.9999_n1_0', 'xcenter_1_0.9999_n2_0',
         'xcenter_1_0.9999_n3_0', 'xcenter_1_0.9999_n4_0', 'xcenter_1_0.999_n0_0', 'xcenter_1_0.999_n1_0', 'xcenter_1_0.999_n2_0', 'xcenter_1_0.999_n3_0', 'xcenter_1_0.999_n4_0', 'xcenter_1_0.99_n0_0', 'xcenter_1_0.99_n1_0',
         'xcenter_1_0.99_n2_0', 'xcenter_1_0.99_n3_0', 'xcenter_1_0.99_n4_0', 'xcenter_1_0.9_n0_0', 'xcenter_1_0.9_n1_0', 'xcenter_1_0.9_n2_0', 'xcenter_1_0.9_n3_0', 'xcenter_1_0.9_n4_0', 'xcenter_2_0.5_n0_0',
         'xcenter_2_0.5_n1_0', 'xcenter_2_0.5_n2_0', 'xcenter_2_0.5_n3_0', 'xcenter_2_0.5_n4_0', 'xcenter_2_0.75_n0_0', 'xcenter_2_0.75_n1_0', 'xcenter_2_0.75_n2_0', 'xcenter_2_0.75_n3_0', 'xcenter_2_0.75_n4_0',
         'xcenter_2_0.95_n0_0', 'xcenter_2_0.95_n1_0', 'xcenter_2_0.95_n2_0', 'xcenter_2_0.95_n3_0', 'xcenter_2_0.95_n4_0', 'xcenter_2_0.98_n0_0', 'xcenter_2_0.98_n1_0', 'xcenter_2_0.98_n2_0', 'xcenter_2_0.98_n3_0',
         'xcenter_2_0.98_n4_0', 'xcenter_2_0.9999999_n0_0', 'xcenter_2_0.9999999_n1_0', 'xcenter_2_0.9999999_n2_0', 'xcenter_2_0.9999999_n3_0', 'xcenter_2_0.9999999_n4_0', 'xcenter_2_0.999999_n0_0', 'xcenter_2_0.999999_n1_0',
         'xcenter_2_0.999999_n2_0', 'xcenter_2_0.999999_n3_0', 'xcenter_2_0.999999_n4_0', 'xcenter_2_0.99999_n0_0', 'xcenter_2_0.99999_n1_0', 'xcenter_2_0.99999_n2_0', 'xcenter_2_0.99999_n3_0', 'xcenter_2_0.99999_n4_0',
         'xcenter_2_0.9999_n0_0', 'xcenter_2_0.9999_n1_0', 'xcenter_2_0.9999_n2_0', 'xcenter_2_0.9999_n3_0', 'xcenter_2_0.9999_n4_0', 'xcenter_2_0.999_n0_0', 'xcenter_2_0.999_n1_0', 'xcenter_2_0.999_n2_0',
         'xcenter_2_0.999_n3_0', 'xcenter_2_0.999_n4_0', 'xcenter_2_0.99_n0_0', 'xcenter_2_0.99_n1_0', 'xcenter_2_0.99_n2_0', 'xcenter_2_0.99_n3_0', 'xcenter_2_0.99_n4_0', 'xcenter_2_0.9_n0_0', 'xcenter_2_0.9_n1_0',
         'xcenter_2_0.9_n2_0', 'xcenter_2_0.9_n3_0', 'xcenter_2_0.9_n4_0', 'xshape', 'yshape', 'zshape'}

    feats2 = feats1.drop(feats_not_in_0311_version, 1)
    fname2 ='feats_keras_0311_%s_%s_%s.csv'% (stage, len(feats2), date_version) 
    feats2.to_csv(fname2, index=True)
    print ("OVERALL feature file and process: ", stage, fname2, time.time()-start_time)
    


