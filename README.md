# Classification Task with Neural Network

This is the programming assignment of Data Mining course(course code:CSIT5210/MSBD5002) in CSE, HKUST 2017 Fall.

## Prerequisites

- Linux(tested on Redhat)

- Python 3.5.4

- Numpy

- Tensorflow

- Shell tools: awk&**shulf**&sed are properly installed

## Data preparation
please go to http://www.cse.ust.hk/~leichen/courses/mscbd-5002/assignments/dataset.rar or https://drive.google.com/open?id=1bldf8x1isvLjlRAXjadTcXJkDayQeQNJ to download the dataset.
**Extract it into folder `dataset`**


## How to run?

**Before you run, make sure your tensorflow environment is properly activated!!!**

- How to train?

  - If you want to start it all over again, open a terminal and type

    `cd  $root/train`

    `sh run.sh`

    but it would be time consuming (it will take approximately 15 mins ) due to the feature extraction procedure, OR, to save your time,

  - You can use the extracted feature data to do training(recommended):

    `cd  $root/train`

    `sh training.sh`

- How to predict?

  - Same as training, if you want to start it all over again:

    `cd  $root/predict`

    `sh run.sh`

    OR

  - You can use the extracted feature data to do training

    `cd  $root/predict`

    `sh predict.sh`

  If you successfully run `predict.sh`, an output file `output.csv` will be generated.

If you have any problem regarding this program, feel free to reach me out:  yfanal@connect.ust.hk



## Problem description

Clickstream is important for mining user's latent behavior. As for online learning platform, thelectuerer can monitor students' learning pattern by clickstream pattern analysis. Video-clickstream records students' click actions when watching lecture videos. A general video-clickstream log file contains the following events: load_video, play_video, pause_video, seek_video,stop_video, and speed_change_video.In this part, you need to predict students' final exam performance, passed or failed, based ontheir clickstream event log from 63 lecture videos in the same online learning semester.

The dataset including:	

- TrainFeatures.csv: 5050 students' video clickstream log info covering 63 videos. You needto use it for training.

- TestFeatures.csv: 1293 students' video clickstream log info covering 63 videos. You need touse it in the testing stage. The students in TrainFeatures.csv and TestFeatures.csvbelong to the same learning semester and share the same grading strategy.

- TrainLabel.csv: The label for 5050 students (1 for pass, 0 for fail).

- TestData.csv: The students you need to give prediction for. Their learning log info can be

  found in TestFeatures.csv.

- VideoInfo.csv: Video duration info for 63 lecture videos.

- Sample_submission.csv: The sample submission file you may refer.

- Description.pdf: Some description for the log events and attributes.
