## PREPROCESS

echo "Creating masks..."
python image_process/create_masks.py
echo "Masks created"

echo "Split image and masks to create new database ..."
python image_process/split_images_masks.py
echo "Split Done - Now, you can start with split_dataset"

## TASK1: lOCALIZATION

echo "training unet (base on paper 2015) localization model"
python train_unet_base.py --mode loc

echo "training resnet34(pretrained) backbone for unet localization model"
python train_resnet34_unet.py --mode loc

echo "training resnet50(pretrained) backbone for unet localization model"
python train_resnet50_unet.py --mode loc

### TASK2: CLASSIFICATION

echo "training unet (base on paper 2015) classification model"
python train_unet_base.py --mode cls

echo "training resnet34(pretrained) backbone for unet classification model"
python train_resnet34_unet.py --mode cls

echo "training resnet50(pretrained) backbone for unet classification model"
python train_resnet50_unet.py --mode cls

### DONE

echo "All models trained!"