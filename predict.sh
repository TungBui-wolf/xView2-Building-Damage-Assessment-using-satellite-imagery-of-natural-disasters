## TASK1: lOCALIZATION

echo "predicting unet (base on paper 2015) localization model"
python predict_unet_base.py --mode loc

echo "predicting resnet34(pretrained) backbone for unet localization model"
python predict_resnet34_unet.py --mode loc

echo "predicting resnet50(pretrained) backbone for unet localization model"
python predict_resnet50_unet.py --mode loc

### TASK2: CLASSIFICATION

echo "predicting unet (base on paper 2015) classification model"
python predict_unet_base.py --mode cls

echo "predicting resnet34(pretrained) backbone for unet classification model"
python predict_resnet34_unet.py --mode cls

echo "predicting resnet50(pretrained) backbone for unet classification model"
python predict_resnet50_unet.py --mode cls

### CREATE SUBMISSION

echo "create submission"
python create_submission.py

### SCORING

python ./src/xview2_scoring.py


echo "ALL DONE!"
