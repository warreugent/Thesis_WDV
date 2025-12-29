## Data Structure
```
Data/
└── data_folder/
    ├── images/                 # Images split in train and test set
    |   ├── train/
    |   └── test/
    ├── annotations/            # JSON train and test files in COCO format
    |   ├── instances_train.json
    |   └── instances_test.json
    └── yolosplits/             # a folder containing nested train splits at different instance budgets
        ├── inst 250/
        ├── inst 500/
        └── inst 1000/
```