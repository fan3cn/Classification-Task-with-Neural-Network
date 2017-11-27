### Feature Engineering

> Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. Feature engineering is fundamental to the application of machine learning, and is both difficult and expensive.The features in your data are important to the predictive models you use and will influence the results you are going to achieve. The quality and quantity of the features will have great influence on whether the model is good or not. [Feature engineering-From Wikipedia](https://en.wikipedia.org/wiki/Feature_engineering)

In the click stream cases, the original data provides several operations conducted by a student, some of them are useful while others not that much. **We believe that a student's final performance is strongly relevant to the videos that he/she has watched and the operation he/she did on that video.** As a result, we abandoned some features like `click_time` ,`video_duration`, `session`, and only consider the quadratic combination of `video_id` and `event_type`  which closely simulates a student's operation on a video. That makes sense in real senario. And our final result with training accuracy of 99% and **cross validation accuracy of 85%**  proves that too. 

Concretely, we first expand `event_type`  by 

- splitting `seek_video` into:
  - `seek_video_forward`  
  - `seek_video_backward` 
- splitting `seek_video` into:
  - `speed_change_video_fast`
  - `speed_change_video_slow`

Then we have the following 8 types of  `event_type` which are list in `../dataset/event_type.txt` as well:

- load_video
- play_video
- pause_video
- seek_video_forward
- seek_video_backward
- stop_video 
- speed_change_video_fast
- speed_change_video_slow

Next, we jointly combine `video` and the expanded  `event_type`  together, as a result, **we expand our feature space to 63 * 8 = 504 !!!**

For example, for the following features:

- `b6cf1ed99599b07e359fb2612c2462da_seek_video_backward`

  Explains how many minutes in total a student seeked video `b6cf1ed99599b07e359fb2612c2462da` backward.

- `685254f9d4324869ba11d564db57921_speed_change_video_slow`

  Explains how many times in total a student slowed the speed of watching video `685254f9d4324869ba11d564db57921`

- `8234e7a65ea3cdf57508e232cb4f58d1_pause_video`

  Explains how many times in total a student paused the play of video `8234e7a65ea3cdf57508e232cb4f58d1`

### Model

We do the classification by using a 3-layer nerual network architecture with 25 * 12 * 1 **neurons** from the first hidden layer to the last output layer. We use `sigmoid_cross_entropy` as the loss function and Adam(Adaptive Moment Estimation) method for gradient descent.
