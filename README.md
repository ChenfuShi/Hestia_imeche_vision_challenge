# Vision challenge repository for 2021 Hestia imeche UAS challenge

directories: 
- data: contains all the extracted frames from videos collected from plane
- NN_recognition_train: directory that contained the stuff to train the models
- recording_buttons: the two scripts that were used to collect the videos from the plane
- square_recognition: the pipeline that runs on the plane itself for recognition
- tests_chenfu: these are the jupyter notebooks that I used for development

Unfortunately we did not manage to get the final pipeline that runs on the plane itself to work although the neural networks recognize the targets alright and can be easily quantized and run on the raspberry pi with the pycoral edge TPU.
