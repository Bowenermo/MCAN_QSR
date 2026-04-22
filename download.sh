wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip && \
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip && \
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip && \
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip && \
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip && \

unzip -o v2_Questions_Train_mscoco.zip && \
unzip -o v2_Questions_Val_mscoco.zip && \
unzip -o v2_Questions_Test_mscoco.zip && \
unzip -o v2_Annotations_Train_mscoco.zip && \
unzip -o v2_Annotations_Val_mscoco.zip 
