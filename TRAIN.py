import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import numpy as np
import reader_flat

import MODEL_CONVNEXT as Model
import cv2
import reader_class

#---------------------------Input parameters-----------------------------------------------------



train_dir=r"3D_Shape_Recognition_And_Retrieval_Max_variations"

train_folder_structure= "class_base"# "flat"/"class_base". File folder strcuture "flat" mean flat directory: train_dir/instance_dir/img.jpg.  "class" mean train_dir/class_dir/instance_dir/img.jpg
# "class" should be used for the LAS&T 3D objects synthetic set and "flat" for everything


test_dir=""
test_folder_structure= "flat"# "flat"/"class_base". File folder strcuture "flat" mean flat directory: train_dir/instance_dir/img.jpg.  "class" mean train_dir/class_dir/instance_dir/img.jpg
# "class" should be used for the LAS&T 3D objects synthetic set and "flat" for everything



saved_model=""
start_from=0 # start iteration from
log_dir="logs" # where trained net will be saved
augmentation_level="high" # "high"/"medium"/"low"
load_model=True
pre_trained_model_path=log_dir+"/defult.torch"
#---load net ---------------------------------------------------------------------------------------------


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=Model.Net()
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)


########################################################################################################################################################
# Train function
###########################################################################################################################
def train(reader_train, readers_test, start_from=0,log_dir="logs/", num_epochs=100000, eval_every=100):
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    eval_file = log_dir + '/evaluation_results.txt'
    fl=open(eval_file,"w"); fl.close()
    mean_loss = 0
    model.train()
    mean_accuracy= np.zeros([20],dtype=np.float32)
    print("start training")
    for step in range(start_from,num_epochs): # main training loop

                imgs, lbls, files = reader_train.readbatch(ncluster=5, ninst=2,augment=True) # read data
       #*****************************************************************************************
                # for i in range(imgs.shape[0]):
                #     cv2.imshow(str(i) + "Train  Label:" + str(lbls[i]), imgs[i])  # files[i]["di"]+
                # cv2.waitKey()
                # cv2.destroyAllWindows()
       #*********************Maintraining steps*************************************************************************
                optimizer.zero_grad()
                embeddings = model(imgs)
                loss,accuracy =model.LossCosineSimilarity(embeddings, lbls, temp=0.2)
                loss.backward()
                optimizer.step()
     #*************Statitics evaluation ****************************************88
                mean_loss = mean_loss*0.99+loss.detach().cpu()*0.01
                mean_accuracy[step%20]=accuracy
                if step%10==0:
                          print(step,"mean loss",mean_loss,"loss",loss, " accuracy",accuracy," mean accuracy",mean_accuracy.mean())
    #----------Run evaluation on test set---------------------------------
                if step % eval_every == 0 and readers_test is not None:
                    res=evaluate(readers_test)
                    print(str(res))
                    txt = "\n" + str(step) + ") \t" +"\t"+str(res) +"\t"
                    print(txt)
                    fl = open(eval_file, "a+")
                    fl.write(txt)
                    fl.close()
    #------------Save model----------------------------------------------------------------------------
                if step % 5000 == 0: # save model
                          torch.save(model.state_dict(), log_dir+"//"+str(step)+".torch")
                if step % 1000 == 0: # save model temporary
                    torch.save(model.state_dict(), log_dir + "//defult.torch")

################################################################################################################################################
# evaluation function
#####################################################################################################################
def evaluate(reader_test):

   # print("\n*************************Testing**************************************************************************\n")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for ii in range(20):
                imgs, lbls, files = reader_test.readbatch(ncluster=3, ninst=2,augment=False)

         #       *****************************************************************************************
         #        for i in range(imgs.shape[0]):
         #            cv2.imshow(str(i) + "Eval Label:" + str(lbls[i]), imgs[i])  # files[i]["di"]+
         #        cv2.waitKey()
         #        cv2.destroyAllWindows()
          #      **********************************************************************************************
                embeddings = model(imgs)
                similarities = torch.mm(embeddings, embeddings.T)
                similarities -= torch.eye(similarities.shape[0]).cuda() * 10
                predictions = torch.argmax(similarities, dim=1)
                predictions=predictions.detach().cpu().numpy()
                correct += (lbls == lbls[predictions]).sum()
                total += len(lbls)

    print(f"Test Accuracy: {correct / total:.4f}")

    model.train()
    return correct / total
#####################################################################################################################################################



#-------Create reader----------------------------------------------------------------------
if train_folder_structure=="class_base":
        train_reader = reader_class.reader(main_dir=train_dir,augmentation_level=augmentation_level)
else:
        train_reader = reader_flat.reader(main_dir=train_dir, augmentation_level=augmentation_level)
if len(test_dir)>0:
    if test_folder_structure=="class_base":
            test_reader = reader_class.reader(main_dir=test_dir,augmentation_level=augmentation_level)
    else:
            test_reader = reader_flat.reader(main_dir=test_dir, augmentation_level=augmentation_level)
else:
    test_reader=None
#------start from previously trained ned  ----------------------------------------------
if load_model==True and os.path.exists(pre_trained_model_path):
    model.load_state_dict(torch.load(pre_trained_model_path, weights_only=True))
    print("Loading model weights from ", pre_trained_model_path)
#---------------------------------------------------------------------------------------------

train(reader_train=train_reader,start_from=start_from,num_epochs=10002001, readers_test=test_reader,log_dir=log_dir,eval_every=1000)