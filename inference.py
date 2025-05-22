# Match retrieve images of same instance using pretrained model
# find which of images sample_image1,sample_image2 is more similar to anchor_image
import torch
import cv2
import torch
import MODEL_CONVNEXT as Model
import cropping_resizing
import numpy as np
#_-------------input parameters---------------------------------------------------------------
anchor_image="samples/benchmark/27/20250510_195351.jpg"
sample_image1="samples/benchmark/27/20250511_224530.jpg"
sample_image2="samples/benchmark/28/20250510_195351_B.jpg"



saved_model=("logs/defult.torch") #trained model weights

#----------------------Build net----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=Model.Net()
model.load_state_dict(torch.load(saved_model,weights_only=True))
########################################################################################3333
# read images
im1=cropping_resizing.resize_and_center_crop(cv2.imread(anchor_image))
im2=cropping_resizing.resize_and_center_crop(cv2.imread(sample_image1))
im3=cropping_resizing.resize_and_center_crop(cv2.imread(sample_image2))
batch=np.array([im1,im2,im3])
################Make predictions and find similarity
embeddings = model(batch)
similarities = torch.mm(embeddings, embeddings.T)
similarities -= torch.eye(similarities.shape[0]).cuda() * 10
predictions = torch.argmax(similarities, dim=1)
predictions=predictions.detach().cpu().numpy()
if predictions[0]==1:
    print("The contenct of image "+anchor_image+" is more similar to"+sample_image1)
else:
    print("The contenct of image " + anchor_image + "is more similar to" + sample_image2)


