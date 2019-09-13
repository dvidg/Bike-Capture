### Video Masking example.

For simplicity, only two input photos have been included - these are contained in the /images folder.

This model is heavily influenced by the excellent implementation of a Mask R-CNN network presented by Adrian Rosebrock in: https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/ 

The file has been edited to remove superfluous features and training performed on my own synthetic dataset - included with this model. Additionally, an 'Ignore' class was implemented to remove the features that are not the human - these are irrelevant for segmentation for aerodynamic analysis


#### RUNNING THE FILE

python mask_rcnn_video.py --mask-rcnn mask-rcnn-coco --image "image address"

eg:
python mask_rcnn.py --mask-rcnn mask-rcnn-coco --image images/ComplexCyclistRGB.jpg
