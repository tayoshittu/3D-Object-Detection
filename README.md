# 3D-object-detection
1. I have used preprocessed data from https://drive.google.com/file/d/1P_uFQcvVFf10TLxjIaFMfjto8ZHON-N2/view?usp=sharing
which contains preprocessed training and validation data. 

2. This is put to training using train.py

3. A trained model is saved and after evaluation, eval_sunrgbd folder contains results on validation data.
 You have to use 3 files from eval_sunrgbd folder from drive
•	**_pc.ply (It contains point clouds of input )
•	*_pred_confident_nms_bbox.ply(It contains predicted bounding boxes)
•	*_pred_map_cls.txt (It contains information of class the output belongs to)
Suppose in *_pred_map_cls.txt, it is written.
3 
0--------0.9844583
3 
0.35467-------277
1
3.55---0.45
here, 3,3 and 1 are class names from ['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']

which means 3 is chair and 1 is table etc

4. Output is always a ply file and we can visualize it in Meshlab software.

5. test_run.py is set to take input stream of ply files from a folder and save corresponding results in a directory.

7. Google Drive containing all my project files used in google colab
https://drive.google.com/drive/folders/1ScYig5Jx61cnWL7RnpE5L3qQyNgA_OiR?usp=sharing

training and validation datasets can be downloaded from the google drive located in
/sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_train and /sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_val
and place them in your sunrgbd folder to run a your host computer.


steps on google colab available in 3D_Object_Detection.ipynb
1. download and install cuda 10

2. go to pointnet2 directory

3. CUDA 10 Layers were compiled for the backbone network PointNet++
!python setup.py install

3. then go to /3d-object-detection folder and train model using this command
!python train.py --log_dir log

4. test and evaluate with the checkpoint_sunrgbd.tar
!python eval.py

you can also run demo by 
!python demo.py

