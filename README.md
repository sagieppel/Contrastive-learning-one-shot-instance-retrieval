# Simple Contrastive Learning Net For Recognition of Shapes And Textures

Simple net that can be used using the [Large Shape and Texture Dataset](https://sites.google.com/view/lastdataset/home) for one-shot recognition of shapes and textures

# Training

Download and extract train data for :
[3D shape recognition one-shot training set](https://zenodo.org/records/15453634/files/3D_Shape_Recognition_Synthethic_GENERAL_LARGE_SET_76k.zip?download=1)

[2D shape recognition one-shot training set](https://zenodo.org/records/15453634/files/2D_Shapes_Recognition_Textured_Synthetic_Resize2_GENERAL_LARGE_SET_61k.zip?download=1)

[3D materials recognition one-shot training set](https://zenodo.org/records/15453634/files/3D_Materials_PBR_Synthetic_GENERAL_LARGE_SET_80K.zip?download=1)

[2D texture recognition one-shot training set](https://zenodo.org/records/15453634/files/2D_Textures_Recogition__GENERAL_LARGE_SET_Synthetic_53K.zip?download=1)

**In Train.py**  
Set the **train_dir** to the folder with the training data.

If you use 3D shapes set **train_folder_structure= "class_base"**
In any other case set  **train_folder_structure= "flat"**

Run Train.py script, and the train models will appear in the log folder (note about 10k training steps should be enough for good results)
This was run with RTX 3090

# Evaluating
Train the model or download trained model weights for 3D shape recognition from [here](https://icedrive.net/s/WzjBZCCzRWTgQafZPiRxS1y8DwAN).
Download the benchmark for 3D shape recognition and retrieval from a single image from [here](https://zenodo.org/records/15453634/files/Real_Images_3D_shape_matching_Benchmarks.zip?download=1).   
In **evaluate.py** 
Set the path to the trained model weight in **saved_model**. 
Set path to benchmark folder in **test_dir**.
Run Evaluating.py script

# Inference 
Train the model or download trained model weights for 3D shape recognition from [here](https://icedrive.net/s/WzjBZCCzRWTgQafZPiRxS1y8DwAN).
In **inference.py**
Set the path to the trained model weights in **saved_model**. 
Set paths to 3 images in
**anchor_image**,  
**sample_image1**, 
**sample_image2** 
The script will run the model with the 3 images and will find if **anchor_image**  is more similar to **sample_image1** or **sample_image2**.
