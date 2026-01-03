This is a write up with a top level overview of this repository

```
Data/
├── README.md                               # Structure overview of the Data directory (more details)
└── ...                                     # All datasets and splits

environments/                               # Yaml and bash files for the required envs
├── main-env-locked.yml
├── rexomni_env.yml
└── semidetr_env.bash

Evaluations/                                # Contains json log files per experiment with eval statistics
├── Experiment_1/
|   ├── apples_test_gd_b_evaluation.json    # format: {dataset}_{set}_{model and repeat e.g. r0}_evaluation.json
|   └── ...                               
├── Experiment_2/
└── evaluation.ipynb                        # A notebook to calculate eval statistics, Results dir -> Evaluations dir
                                            
External/                                   # Directory to store external repositories (empty)

Models/                                     # Contains all model families' subdirectories and code
├── MLLM/ 
|   └── rexomni.ipynb                              
├── SEMI-DETR/                              # WIP (environment is ready)
|   └── semidetr.ipynb
├── VLM/
|   └── vlms.ipynb
└── YOLO/  
    ├── runs/                               # YOLO training runs and models
    ├── yolo_outputs/                       # YOlO style outputs (transformed to COCO format before part of Results dir)
    ├── yolo.ipynb
    └── yolo11m.pt

Notebooks                                   # Additional notebooks
├── dataset_split.ipynb                     # train-test split
├── dataset_statistics.ipynb
└── display_predictions.ipynb

Results                                      # COCO json files with the output predictions (each subdir has same format)
├── Experiment_1/
|   ├── apples_test_gd_b_predictions.json    # format: {dataset}_{set}_{model and repeat e.g. r0}_predictions.json
|   └── ...                               
├── Experiment_2.1/
└── 

src                                             
└── annotation_methods/                      # Importables  
    ├── __init__.py
    ├── budget_splits.py
    ├── coco_utils.py
    ├── io_utils.py                         
    ├── vlm_helpers.py                         
    └── yolo_helpers.py    

Timing Experiments                           # Timing experiment data and calculations
├── Timing Experiments Data/
|   ├── apples/
|   ├── tomatoes/
└── timing_experiments.ipynb
```