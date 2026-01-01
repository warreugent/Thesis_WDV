## Data Structure
```
Data/
└── data_folder/
    ├── images/                             # Images split in train and test set
    |   ├── train/
    |   └── test/
    ├── annotations/                        # JSON train and test files in COCO format
    |   ├── instances_train.json
    |   └── instances_test.json
    └── yolosplits/                         # a folder containing nested train splits at different instance budgets
        ├── inst 250_r0/                    # we repeat the nested split for each budget 3 times (r0, r1, r2)
        |   ├── images/
        |   ├── labels/
        |   ├── dateset.yaml                # required file telling yolo models where the data sits
        |   ├── manifest_train_images.txt   # contains the train images file names
        |   ├── manifest_val_images.txt     # contains the val images file names
        |   └── stats.json                  # statistics on the split
        ├── inst 250_r1/
        ├── inst 250_r2/
        ├── inst 500_r0/
        ├── inst 500_r1/
        ├── inst 500_r2/
        ├── inst 1000_r0/
        ├── inst 1000_r1/
        └── inst 1000_r2/
```