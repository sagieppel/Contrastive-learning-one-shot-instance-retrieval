import cv2
import os
import numpy as np
import random
import augment_image
#=============================================================================================================
class reader():
    def __init__(self,main_dir,augmentation_level):
        self.data=[]
        self.augmentation_level = augmentation_level
        for sdr in os.listdir(main_dir):
            path1=main_dir+"/"+sdr
            if not os.path.isdir(path1): continue
            file_list=[]
            for fl in os.listdir(path1):
                if not ".jpg" in fl: continue
                file_list.append(path1+"//"+fl)
            if len(file_list)>=2:
                    self.data.append({"name":sdr,"files":file_list})
           # self.spmaker = Create_Scene_Shape_Class.shapemaker(shape_dir,texture_dir, rotate_shape=True,keep_shape_size=True,rotate_texture=True,uniform_shape_texture=False,uniform_background=True,black_and_white=False)
#--------------------------------------------------------------------------------------------------
#########################################################################################################################
    def readbatch(self,ncluster=4,ninst=2,augment=False):
         imgs = []
         lbls = []
         names = []
         files_data =[]
         label = -1
         for fcl in range(ncluster):
             label+=1
             while(True):
               dt = random.choice(self.data)
               if dt["name"] not in names:break
             names.append(dt["name"])
             insts_names=[]
             for  finst in range(ninst):
                #-------------Choose and read random image and mask-------------------------------------
                    while (True):
                         ins_file = random.choice(dt["files"])
                         if ins_file not in insts_names: break
                         if len(dt["files"])<ninst: exit("ERRROR NOT ENOUGH instances in subdir",dt["names"],len(dt["files"]))
                    names.append(dt["name"])
                    insts_names.append(ins_file)
                    im = cv2.imread(ins_file)
                    ins_mask_file = ins_file.replace(".jpg", "_MASK.png")
                    if os.path.exists(ins_mask_file):
                         mask = cv2.imread(ins_mask_file, 0) > 120
                    else:
                         print("Cant load mask")
                         mask = None
                 # -------------------------Augment---------------------------------------------------------------------
                    if augment and  self.augmentation_level != "none":
                        im, mask = augment_image.augment_image(im, mask=mask,
                                                               apply_gaussian=np.random.rand() < 0.3,
                                                               apply_decolor=np.random.rand() < 0.05,
                                                               apply_noise=np.random.rand() < 0.25,
                                                               apply_intensity=np.random.rand() < 0.2,
                                                               min_size=256,high_augmentation=(self.augmentation_level=='high'))
                    imgs.append(im)
                    lbls.append(label)
                    files_data.append({"dir":dt["name"],"path":ins_file})
         return np.array(imgs), np.array(lbls), files_data
##################################################################################################################################################

if __name__ == "__main__":
          main_dir = r"/home/sagiep/Desktop/ShapeAndTexture/Test_shapes/shape_and_texture_all_Different//"
          rd = reader(main_dir = main_dir)
          imgs,lbls, files = rd.readbatch(ncluster=4,ninst=2,augment=False)
          y=0
          for i in range(imgs.shape[0]):
              cv2.imshow( str(i)+"  Label:"+str(lbls[i]),imgs[i]) # files[i]["di"]+
          cv2.waitKey()






