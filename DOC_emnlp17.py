
# coding: utf-8

# In[1]:

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[2]:

import json

import numpy as np
np.random.seed(1)

import keras
print keras.__version__ #version 2.1.2
from keras import preprocessing


# In[3]:

fn = '50EleReviews.json' #origial review documents, there are 50 classes
with open(fn, 'r') as infile:
        docs = json.load(infile)
X = docs['X']
y = np.asarray(docs['y'])
num_classes = len(docs['target_names'])

'''
50EleReviews.json
y : size =  50000
X : size =  50000
target_names : size =  50
'''


# In[4]:

#count each word's occurance
def count_word(X):
    word_count = dict()
    for d in X:
        for w in d.lower().split(' '): #lower
            if w in word_count:
                word_count[w] += 1
            else:
                word_count[w] = 1            
    return word_count

word_count = count_word(X)
print 'total words: ', len(word_count)


# In[5]:

#get frequent words
freq_words = [w  for w, c in word_count.iteritems() if c > 10]
print 'frequent word size = ', len(freq_words)


# In[6]:

#word index
word_to_idx = {w: i+2  for i, w in enumerate(freq_words)} # index 0 for padding, index 1 for unknown/rare words
idx_to_word = {i:w for w, i in word_to_idx.iteritems()}


# In[7]:

def index_word(X):
    seqs = []
    max_length = 0
    for d in X:
        seq = []
        for w in d.lower().split():
            if w in word_to_idx:
                seq.append(word_to_idx[w])
            else:
                seq.append(1) #rare word index = 1
        seqs.append(seq)
    return seqs


# In[8]:

#index documents and pad each review to length = 3000
indexed_X = index_word(X)
padded_X = preprocessing.sequence.pad_sequences(indexed_X, maxlen=3000, dtype='int32', padding='post', truncating='post', value = 0.)


# In[9]:

#split review into training and testing set
def splitTrainTest(X, y, ratio = 0.7): # 70% for training, 30% for testing
    shuffle_idx = np.random.permutation(len(y))
    split_idx = int(0.7*len(y))
    shuffled_X = X[shuffle_idx]
    shuffled_y = y[shuffle_idx]
    
    return shuffled_X[:split_idx], shuffled_y[:split_idx], shuffled_X[split_idx:], shuffled_y[split_idx:]   

train_X, train_y, test_X, test_y = splitTrainTest(padded_X, y)

print train_X.shape, train_y.shape, test_X.shape, test_y.shape


# In[10]:

#split reviews into seen classes and unseen classes
def splitSeenUnseen(X, y, seen, unseen):
    seen_mask = np.in1d(y, seen)# find examples whose label is in seen classes
    unseen_mask = np.in1d(y, unseen)# find examples whose label is in unseen classes
    
    print np.array_equal(np.logical_and(seen_mask, unseen_mask), np.zeros((y.shape), dtype= bool))#expect to see 'True', check two masks are exclusive
    
    # map elements in y to [0, ..., len(seen)] based on seen, map y to unseen_label when it belongs to unseen classes
    to_seen = {l:i for i, l in enumerate(seen)}
    unseen_label = len(seen)
    to_unseen = {l:unseen_label for l in unseen}
        
    return X[seen_mask], np.vectorize(to_seen.get)(y[seen_mask]), X[unseen_mask], np.vectorize(to_unseen.get)(y[unseen_mask])

seen = range(25)#seen classes
unseen = range(25,50)#unseen classes

seen_train_X, seen_train_y, _, _ = splitSeenUnseen(train_X, train_y, seen, unseen)
seen_test_X, seen_test_y, unseen_test_X, unseen_test_y = splitSeenUnseen(test_X, test_y, seen, unseen)

from keras.utils.np_utils import to_categorical
cate_seen_train_y = to_categorical(seen_train_y, len(seen))#make train y to categorial/one hot vectors


# In[11]:

#Network, in the paper, I use pretrained google news embedding, here I do not use it and set the embedding layer trainable
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding, Input, Concatenate
from keras.layers import Conv1D, GlobalMaxPooling1D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import backend as K

def Network(MAX_SEQUENCE_LENGTH = 3000, EMBEDDING_DIM = 300, nb_word = len(word_to_idx)+2, filter_lengths = [3, 4, 5],
    nb_filter = 150, hidden_dims =250):
    
    graph_in = Input(shape=(MAX_SEQUENCE_LENGTH,  EMBEDDING_DIM))
    convs = []
    for fsz in filter_lengths:
        conv = Conv1D(filters=nb_filter,
                                 kernel_size=fsz,
                                 padding='valid',
                                 activation='relu')(graph_in)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)

    if len(filter_lengths)>1:
        out = Concatenate(axis=-1)(convs)
    else:
        out = convs[0]

    graph = Model(inputs=graph_in, outputs=out) #convolution layers
    
    emb_layer = [Embedding(nb_word,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True),
                 Dropout(0.2)
        ]
    conv_layer = [
            graph,
        ]
    feature_layers1 = [
            Dense(hidden_dims),
            Dropout(0.2),
            Activation('relu')
    ]
    feature_layers2 = [
            Dense(len(seen) ),
            Dropout(0.2),
    ]
    output_layer = [
            Activation('sigmoid')
    ]

    model = Sequential(emb_layer+conv_layer+feature_layers1+feature_layers2+output_layer)
    model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model


# In[12]:

model = Network()    
print model.summary()


# In[13]:

bestmodel_path = 'bestmodel.h5'

checkpointer = ModelCheckpoint(filepath=bestmodel_path, verbose=1, save_best_only=True)
early_stopping=EarlyStopping(monitor='val_loss', patience=5)

model.fit(seen_train_X, cate_seen_train_y,
              epochs=100, batch_size=128, callbacks=[checkpointer, early_stopping], validation_split=0.2)

model.load_weights(bestmodel_path)


# In[14]:

#predict on training examples for cauculate standard deviation
seen_train_X_pred = model.predict(seen_train_X)
print seen_train_X_pred.shape


# In[16]:

#fit a gaussian model
from scipy.stats import norm as dist_model
def fit(prob_pos_X):
    prob_pos = [p for p in prob_pos_X]+[2-p for p in prob_pos_X]
    pos_mu, pos_std = dist_model.fit(prob_pos)
    return pos_mu, pos_std


# In[18]:

#calculate mu, std of each seen class
mu_stds = []
for i in range(len(seen)):
    pos_mu, pos_std = fit(seen_train_X_pred[seen_train_y==i, i])
    mu_stds.append([pos_mu, pos_std])

print mu_stds


# In[20]:

#predict on test examples
test_X_pred = model.predict(np.concatenate([seen_test_X,unseen_test_X], axis = 0))
test_y_gt = np.concatenate([seen_test_y,unseen_test_y], axis = 0)
print test_X_pred.shape, test_y_gt.shape


# In[23]:

#get prediction based on threshold
test_y_pred = []
scale = 1.
for p in test_X_pred:# loop every test prediction
    max_class = np.argmax(p)# predicted class
    max_value = np.max(p)# predicted probability
    threshold = max(0.5, 1. - scale * mu_stds[max_class][1])#find threshold for the predicted class
    if max_value > threshold:
        test_y_pred.append(max_class)#predicted probability is greater than threshold, accept
    else:
        test_y_pred.append(len(seen))#otherwise, reject
  


# In[24]:

#evaluate
from sklearn.metrics import precision_recall_fscore_support


# In[27]:

precision, recall, fscore, _ = precision_recall_fscore_support(test_y_gt, test_y_pred)
print 'macro fscore: ', np.mean(fscore)


# In[ ]:



