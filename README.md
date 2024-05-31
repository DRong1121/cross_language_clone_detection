## Cross-Language Code Clone Detection (XLCCD)  


### Introduction  

This project aims to develop a module for semantic (Type-4) clone detection towards Java and Python code snippets.  
The module is developed in Python 3.10, PyTorch 1.13.0 and can be trained on CUDA 11.4, Ubuntu 20.04.2.   
The pipeline of the module is shown in [streamline.png](https://github.com/DRong1121/cross_language_clone_detection/tree/main/streamline.png).  


### Requirements

System Requirements:  
- Ubuntu 20.04.2  

Runtime and Develop Requirements:  
- Python 3.10.13  
- pip 23.3  
- PyTorch 1.13.0  
- CUDA 11.4  
- other dependencies: see requirements.txt

Network Requirements:
- at least 10MB/s bandwidth
- access to HuggingFace

Storage Requirements:
- at least 5GB for downloaded and saved models


### Datasets

The Pre-training Dataset: see Jupyter Notebook--[Pre-training_Dataset_Exploration.ipynb](https://github.com/DRong1121/cross_language_clone_detection/tree/main/notebooks/Pre-training_Dataset_Exploration.ipynb)   
The Fine-tuning Dataset: see Jupyter Notebook--[Fine-tuning_Dataset_Exploration.ipynb](https://github.com/DRong1121/cross_language_clone_detection/tree/main/notebooks/Fine-tuning_Dataset_Exploration.ipynb)  
Data Augmentation Technique: see Jupyter Notebook--[Data_Augmentation.ipynb](https://github.com/DRong1121/cross_language_clone_detection/tree/main/notebooks/Data_Augmentation.ipynb)  
Code Tokenization Methodology: see Jupyter Notebook--[Code_Tokenization.ipynb](https://github.com/DRong1121/cross_language_clone_detection/tree/main/notebooks/Code_Tokenization.ipynb)  

### Scripts

For [the Baseline Experiment](https://github.com/DRong1121/cross_language_clone_detection/tree/main/core/fine_tuning_procedure.py), run the following commands:  
```
cd /xlccd/core/scripts  
bash run_fine_tune_train.sh  
bash run_fine_tune_test.sh  
```

For [the C4 Experiment](https://github.com/DRong1121/cross_language_clone_detection/tree/main/core/c4_distributed.py), run the following commands:  
```
cd /xlccd/core/scripts  
bash run_c4_train.sh  
bash run_c4_test.sh  
```

For the XLCCD Experiment([pre-training](https://github.com/DRong1121/cross_language_clone_detection/tree/main/core/pre_training_procedure.py), [fine-tuning](https://github.com/DRong1121/cross_language_clone_detection/tree/main/core/fine_tuning_procedure.py)), run the following commands:
```
cd /xlccd/core/scripts  
bash run_pre_train.sh  
bash run_fine_tune_train.sh  
bash run_fine_tune_test.sh  
```

For [Data Augmentation](https://github.com/DRong1121/cross_language_clone_detection/tree/main/core/transcoder/pipeline.py), run the following commands (need to install dependencies for Transcoder): 
```
cd /xlccd/core/scripts  
bash run_augmentation.sh  
```
Relevant links:  
Transcoder Model: https://github.com/facebookresearch/CodeGen  
fastBPE: https://github.com/glample/fastBPE  


### Other Infos
见 [跨语言克隆代码检测--成果说明文档.docx](https://github.com/DRong1121/cross_language_clone_detection/tree/main/docs/跨语言克隆代码检测--成果说明文档.docx)  
