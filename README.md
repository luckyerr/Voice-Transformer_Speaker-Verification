# VOT--Voice Transformer
The repository contains a framework called VOT, which is used to train the speaker verification model. The model consists of multiple transformers in parallel, and the outputs of these transformers are adaptively combined. Deep fusion semantic memory network (DFSMN) is integrated into the attention part of these transformers to capture long-distance information and enhance local dependence. There is also a new loss function called Additional Angular Margin Focusing Loss (AAMF) to solve the problem of hard sample mining.  
we  provide Voice-Transformer architecture：
![model-sym](https://anonymous.4open.science/r/VOT-C994/picture/3.jpg)  
where the DFSMN model is shown below：
![model-sym](https://anonymous.4open.science/r/VOT-C994/picture/4.jpg)  
### Dependencies
```
pip install -r requirements.txt
```
### Dataset
We have done experiments on Voxceleb1 and CNCeleb2 datasets.
The following script can be used to download and prepare the VoxCeleb1 and VoxCeleb2 datasets for training and testing.

```
python ./dataprep.py --save_path data --download --user USERNAME --password PASSWORD 
python ./dataprep.py --save_path data --extract
python ./dataprep.py --save_path data --convert
```
In order to use data augmentation, also run:

```
python ./dataprep.py --save_path data --augment
```
If you still want to test on the Cn-Celeb2 dataset, click on the download link below
The dataset for Cn-Celeb2 can be download from [here](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2022/Track3_validation_data.zip). 
The test list for Cn-Celeb2 can be downloaded from [here](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2022/Track3_validation_trials.txt). 


In addition to the Python dependencies, `wget` and `ffmpeg` must be installed on the system.

### Train
We trained on normal serial transformer encoder， serial voice encoder，parallel voice encoder：
![model-sym](https://anonymous.4open.science/r/VOT-C994/picture/2.jpg)
With respect to the series and parallel structures, we took four variants of the experiment respectively：
![model-sym](https://anonymous.4open.science/r/VOT-C994/picture/5.jpg)  
train the VOT with AM
```
python ./trainSpeakerNet.py --config ./configs/VOT_AM.yaml
```
train the VOT with AAM
```
python ./trainSpeakerNet.py --config ./configs/VOT_AAM.yaml
```
train the serial-VOT-withoutmemory with AAMF
```
python ./trainSpeakerNet.py --config ./configs/VOT_serial_focal.yaml
```
train the serial-VOT-memory with AAMF
```
python ./trainSpeakerNet.py --config ./configs/VOT_serial_focal_memory.yaml
```
train the parallel-VOT-memory with AAMF
```
python ./trainSpeakerNet.py --config ./configs/VOT_focal.yaml
```
### Test
```
python ./trainSpeakerNet.py --eval --config ./configs/VOT_focal.yaml --initial_model exps/VOT_focal/model/model000000007.model
```
We demonstrate the efficiency of DFSMN on speaker verification tasks：
![model-sym](https://anonymous.4open.science/r/VOT-C994/picture/6.jpg)
We show that the parallel structure requires fewer parameters and is experimentally better
![model-sym](https://anonymous.4open.science/r/VOT-C994/picture/7.jpg)
We trained on Voxceleb1 training set and test on the Voxceleb1 test set (there are no common speakers)：
![model-sym](https://anonymous.4open.science/r/VOT-C994/picture/8.jpg)
We trained on Voxceleb1 training set and test on the  Cn-Celeb2 test set (there are no common speakers)：
![model-sym](https://anonymous.4open.science/r/VOT-C994/picture/9.jpg)
### generate your own trainlist
Simply modify the path in the file and run it to produce a training file that matches the format.
```
python generate_traintxt.py
```

### Feature Visualization
Need to change the path inside the file
```
python Feature_Visualization.py
```
we  provide the results of feature visualization of FBANK and MFCC feature
![model-sym](https://anonymous.4open.science/r/VOT-C994/picture/1.jpg)

* code reference from https://github.com/clovaai/voxceleb_trainer
