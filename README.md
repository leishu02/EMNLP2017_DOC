# EMNLP2017_DOC
code for our EMNLP 2018 paper "DOC: Deep Open Classification of Text Documents"

DOC's experiment setting is huge. I trimmed them into one file containing every function from pre-processing till evaluating. In paper, I use google-new pretrained embedding. This code does not use pretrained embedding. If you want to fully re-produce the result, you may need to randomly sample 10 times seen-unseen classes split and load the pretrained embedding. 

20NewsGroup: please download 20news-18828.tar.gz from http://qwone.com/~jason/20Newsgroups/ (preprocess: use every word inside) 

50EleReviews.json contains 50-product original reviews.( 25% of 50 classes: 12 seen classes, 75% of 50 classes: 37 seen classes). please download from https://drive.google.com/file/d/1Kgtqbp0B67S-f4W7ULfG-_YP2kOdZ-Do/view?usp=sharing

DOC_emnlp17.py or .ipynb (ipython notebook, it has running results) contains code. 

We have one continual project which solves UNSEEN CLASS DISCOVERY IN OPEN-WORLD CLASSIFICATION https://arxiv.org/pdf/1801.05609.pdf. It shows that DOC also works well on image. 

We have one meta-learning based continuing work recently accepted at the web conference (WWW) 2019:Open-world Learning and Application to Product Classification (code and data is available, see link in paper) 
https://www.cs.uic.edu/~liub/publications/WWW-2019-camera-ready.pdf

library: 
python 2.7 
keras 2.1.2 
scipy 
json 
numpy 
sklearn 
jupyter (if you want to use .ipynb file)

