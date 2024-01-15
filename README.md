# VOT
The repository contains a framework called VOT, which is used to train the speaker verification model. The model consists of multiple transformers in parallel, and the outputs of these transformers are adaptively combined. Deep fusion semantic memory network (DFSMN) is integrated into the attention part of these transformers to capture long-distance information and enhance local dependence. There is also a new loss function called Additional Angular Margin Focusing Loss (AAMF) to solve the problem of hard sample mining.

* code reference from https://github.com/clovaai/voxceleb_trainer
