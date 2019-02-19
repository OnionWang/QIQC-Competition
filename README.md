Quara Insincere Questions Classification Competition Summary
===================================

  
This is my first competition in kaggle, here are some thoughts and tricks.

My kernel is : https://www.kaggle.com/wangcong95/qicq-competition

My kaggle home is : https://www.kaggle.com/wangcong95

Describe
———————————

This is a binaray classification problem, give you one short question, you should identify whether it is sincere. An insincere question is defined as a question intended to make a statement rather than look for helpful answers. Some characteristics that can signify that a question is insincere:

 - Has a non-neutral tone
 - Is disparaging or inflammatory
 - Isn't grounded in reality
 - Uses sexual content (incest, bestiality, pedophilia) for shock value, and not to seek genuine answers

This competition is a kernel only competition, which can not use external data; By the way, you should finish this task with in 2 hours if you use GPU, otherwise 6 hours is required. That means you cannot use transfer model like BERT, and you cannot ensemble hundred of models without thinking.

Data Cleaning And Word Embedding
———————————

At first, I think stronger data cleaning will get better f1-score. However, with the more clean method I use, my f1-score didn't get better and sometimeseven get worse. I think hold some noise in original data may help models get more robustness features. When I see 3rd place solution, I found they didn't use complex data cleaning skills, but a simple way to add text coverage in pre-trained word embeddings. Most of us just average 2 or 3 embeddings as embedding layer weight. Average is a simple way to get better features, and indeed get better f1-score; however, it may lose some information, if we add it in different weight, or concat in different combination and sequence, we may get more information and more varierence models.

Model Structure
———————————

Here are two kernels which got the most high f1-score in public kernels:

 - https://www.kaggle.com/jannen/reaching-0-7-fork-from-bilstm-attention-kfold
 - https://www.kaggle.com/artgor/text-modelling-in-pytorch

These two kernel are both use pytorch and fix seed to reproduce the result. I used NN model but a little different from it. In fact, it is more simple. Structure is just like this:
![Image text](https://github.com/OnionWang/QIQC-Competition/blob/master/model_structure.png)

I didn't use spartialdropout, it is a very strong regulation skill, and can get a pretty good f1-score by single model, but need more epoches to let model overfit. At first I used it and every fold cv is about 0.6900, average cv is 0.6887, it looks pretty good. But when ensemble, I can only get 0.6960 in public LB, however those cv is 0.6700 can get 0.7000 in public LB. Then I found it helpful to overfit model. That's what I want to say in next section. By the way, I used CycleLearnRate to adapt my learn rate, which promote my model score.

Ensemble Skill
———————————

Most of us may find that someone has low cv score, but get pretty high ensemble score. At first I feel confused, then I found my cv model has strong correlation, cause they get low value loss thus becoming more robustness. As a result, they have little varierace, we know add model varierance will promote ensemble preformance a lot. Overfit is a pretty good way to add model varierance. So I used simple model, which just need 130s per epoch, that means I can train 9 models, every model has 5 epoches. I used 3 fold, and repeat 3 times, every time used different seed to split data, every fold used different seed to initialize weight. It took about 6300s to train my model, after ensemble, I get 0.701 in public LB.

**Local Evaluate**
———————————

It is important to evaluate your model, a suitable evaluation can avoid overfit about public LB, and help you to choose better kernel at final submission. When I begin this competition, I just used average cv score to evaluate my result. Then I found when I got higher cv score, my public LB score became lower. I felt frustrated and didn't known how to evaluate my model. Thanks for @Benjamin Minixhofer and his kernel:

 - https://www.kaggle.com/bminixhofer/a-validation-framework-impact-of-the-random-seed

Yes, ensemble will promote performance, if your models have low correlation, you may get a high score even your cv score is low. So I just split part of whole train data as local test data(about 4 times of public test data). My model will evaluate in this local test dataset. My local test score is strongly correlation with my public LB. After private LB rerun, my best kernel is the kernel which got highest local test score.

Useless Try
———————————

 - add static features
 - use different loss function
 - use the layer before output layer as text features, use xgboost to predict final target
 - change model structure to add varierace
 - stronger data cleaning
 - use transformer as encoder

Useful Skills
———————————

Here are what I have learned from this competition:

 - a little overfitting can get better performance for ensemble model
 - padding at batch level can really speed up train process
 - fine-tune at last epoch can raise your model performance
 - checkpoint ensemble seems at a pretty timesaving way, another interesting method is Stochastic Weight Averaging (SWA:https://arxiv.org/abs/1802.10026)

Then I use these tricks to rewrite my kernel:
padding at batch level reduce train time, thus I use 3 fold repeat 4 times(total 12 models) to ensemble, and fine-tune embedding layer at every last epoch. My local test score from 0.7009 become 0.7021 without changing model structure and data-preprocess.

 I also tried checkpoint ensemble, it indeed add my single fole score, but also add similarity, which lower my final score. SWA looks like attractive, but I may do it wrong way.

Thanks.
