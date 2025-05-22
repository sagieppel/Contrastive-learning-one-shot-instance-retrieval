# Evaluate pretrained model on benchmark/test set

import torch
import evaluation_reader as reader_eval
import MODEL_CONVNEXT as Model


#----------Input parameters-----------------------------------------------------------------------------
test_dir=r"samples/benchmark/" # path folder with benchmark/test set
saved_model=("logs/defult.torch") #trained model weights

#----------------------Build net----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=Model.Net()
model.load_state_dict(torch.load(saved_model,weights_only=True))
# load reader and pretrained model
reader=reader_eval.evaluation_reader(main_dir=test_dir, max_img_per_instance=100, max_img_total=10000,max_instance_per_class=4000)


################################################################################################################################################

def evaluate(reader_test):

   # print("\n*************************Testing**************************************************************************\n")
    model.eval()
    correct, total = 0, 0


    reader_test.indx=0
    with torch.no_grad():
        ii = 0
        for i in range(1): # Run few times to get better accuracy
            reader_test.indx = 0

            while(True):
                    ii+=1

                    success, imgs = reader_test.get_next_question()
                    if not success: break

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
                    total+=1
                    if predictions[0]==1:
                                 correct += 1
                                 reader_test.add_correct()
                    else:
                                 reader_test.add_incorrect()
                    print(ii,") accuracy",correct/total)
                    reader_test.write_correct("STATITICS.xls")
                    # *****************************************************************************************

    return correct / total
#####################################################################################################################################################

evaluate(reader_test=reader) # Run evaluation