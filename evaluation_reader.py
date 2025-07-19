# reader for the evaluation script
import cv2
import os
import random
import numpy as np



############################colldect data images/and instance aon all instances from dir#############################################################################################
# Note all images belonging to the same instance must be in the same dir and have .jpg file, other then this the dir structure doesnt matter
def collect_folder_data(dr,dr_name="", dic_imgs = {},list_indx = [],max_instance_per_class=100000,depth=0):
    instance_per_class = 0
    for ifl in os.listdir(dr):
        path = dr + "/" + ifl

        if ".jpg" in ifl:  # if the folder structure is  mdir/instance_dir/img.jpg
            if dr_name not in dic_imgs: dic_imgs[dr_name] = []  # all the images belong to same folder are belong to the same instance
            dic_imgs[dr_name].append(path)
            list_indx.append({"inst": dr_name, "ins_num": len(dic_imgs[dr_name]), "file": path})
        else:
            if os.path.isdir(path):
                dic_imgs, list_indx=collect_folder_data(path, dr_name=dr_name+"_"+ifl, dic_imgs=dic_imgs, list_indx=list_indx,max_instance_per_class=max_instance_per_class,depth=depth+1)
                instance_per_class+=1
                if instance_per_class>=max_instance_per_class and depth>0: break  # super folders that contain subfolders of instances are refer to as classes
    return  dic_imgs, list_indx




####################################################################################################################################
class evaluation_reader():
    def __init__(self,main_dir = r"",max_img_per_instance=10,max_img_total=10000, max_instance_per_class=100000):
        self.max_img_per_mat=max_img_per_instance # Total number of image to test per instance
        self.max_img2test=max_img_total # total number of image to test
        self.main_dir=main_dir # main folder with image divided into main_dir/class_dir/object_dir/instance_image.png
              #=========================Create dictionary of all images===============================
        self.dic_imgs={} # structure that will contain all images arrange by instance and index
        self.list_indx = []  # list of all images

        self.dic_imgs, self.list_indx = collect_folder_data(main_dir, dic_imgs = {},list_indx = [],depth=0,max_instance_per_class=max_instance_per_class)

        self.indx=0
        self.finish = False
        self.all_correct={}
        self.all_incorrect={}
##########################################################################################################################
        ##########################################################################################################################
    def write_correct(self, outpath):
            txt = "path\tcorrect\tfalse\n"
            for ky in self.all_correct:
                txt += ky + "\t" + str(self.all_correct[ky]) + "\t" + str(self.all_incorrect[ky]) + "\n"
            fl = open(outpath, "w")
            fl.write(txt)
            fl.close()
##################################################################################################################################################
#==========================Go over all images and make one question per image
    def get_next_question(self):
        if self.indx>=len(self.list_indx): return False,None
        if self.indx == len(self.list_indx)-1: self.finish=True
        img_data = self.list_indx[self.indx]

        mat = img_data["inst"]  # anchor image material
        anc_path = img_data["file"] # anchor imager path
        self.img_file = img_data["file"]
        nn =  img_data["ins_num"] # anchor instance number
        self.indx += 1
        if len(self.dic_imgs[mat])<2: return self.get_next_question() # there need to be at least two instance of same object for question to generated


        #--------------------Select anchor image and positive image-------------------
        anch_im = cv2.imread(anc_path)
        self.anc_path=anc_path

        while(True):
                 pos_path = random.choice(self.dic_imgs[mat])
                 if pos_path!= anc_path:
                     pos_im = cv2.imread(pos_path)
                     self.pos_path = pos_path
                     break
        #---------------Select two negative images----------------------------------
        neg_im=[]
        for kk in range(2):
            while (True):
                neg_mat = random.choice(list(self.dic_imgs.keys()))
                if neg_mat == mat or len(self.dic_imgs[neg_mat]) == 0: continue
                im_path = random.choice(list(self.dic_imgs[neg_mat]))
                neg_im.append(cv2.imread(im_path))
                break

        batch=np.array([pos_im,anch_im,neg_im[0],neg_im[1]])



        return True,batch
##########################################################################################################################################################################
    def add_correct(self):
        if self.anc_path not in self.all_correct:
            self.all_correct[self.anc_path] = 0
            self.all_incorrect[self.anc_path] = 0
        self.all_correct[self.anc_path] += 1

        if self.pos_path not in self.all_correct:
            self.all_correct[self.pos_path] = 0
            self.all_incorrect[self.pos_path] = 0
        self.all_correct[self.pos_path] += 1

##########################################################################################################################################################################
    def add_incorrect(self):
            if self.anc_path not in self.all_correct:
                self.all_correct[self.anc_path] = 0
                self.all_incorrect[self.anc_path] = 0
            self.all_incorrect[self.anc_path] += 1

            if self.pos_path not in self.all_correct:
                self.all_correct[self.pos_path] = 0
                self.all_incorrect[self.pos_path] = 0
            self.all_incorrect[self.pos_path] += 1


