#!/bin/sh

# questions
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip

# answers
wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip

# balanced pairs
wget http://visualqa.org/data/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip http://visualqa.org/data/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip

# bottom up features (https://github.com/peteanderson80/bottom-up-attention)
wget https://storage.googleapis.com/bottom-up-attention/trainval.zip https://storage.googleapis.com/bottom-up-attention/test2015.zip
## alternative bottom-up features: 36 fixed proposals per image instead of 10--100 adaptive proposals per image.
#wget https://storage.googleapis.com/bottom-up-attention/trainval_36.zip https://storage.googleapis.com/bottom-up-attention/test2015_36.zip

unzip "*.zip"
