import os
import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as KTF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显�? 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)
# coding: utf-8

# In[1]:



import gc
# In[1]:
#import theano
import numpy as np
import scipy as sp
from scipy import sparse as ssp
from scipy.stats import spearmanr
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
import os
import tensorflow as tf
#import keras.backend.tensorflow_backend as KTF
import getface
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
import del_click0 as dc
from keras.models import *
# 指定第一块GPU可用 

#from config import path
path = './'

# In[2]:
print('read')
columns = ['user_id','photo_id','click','like','follow','time','playing_time','duration_time']
train_interaction = pd.read_table(path+'fstrain_interaction.txt',header=None)
#train_interaction = pd.read_csv(path+'small_train')
train_interaction.columns = columns
train_photo_time_start = train_interaction[['photo_id', 'time']].groupby(['photo_id']).min().reset_index()
train_photo_time_start.columns = ['photo_id', 'start_time']
train_interaction = pd.merge(train_interaction, train_photo_time_start, how='left', on='photo_id')
train_interaction['relative_time'] = ((train_interaction['time'] - train_interaction['start_time'])/(1e8/1000)).astype(int)

test_columns = ['user_id','photo_id','time','duration_time']
del(train_interaction['like'])
del(train_interaction['follow']) 
del(train_interaction['playing_time'])
gc.collect()

test_interaction = pd.read_table(path+'fstest_interaction.txt',header=None)
test_interaction.columns = test_columns

test_photo_time_start = test_interaction[['photo_id', 'time']].groupby(['photo_id']).min().reset_index()
test_photo_time_start.columns = ['photo_id', 'start_time']
test_interaction = pd.merge(test_interaction, test_photo_time_start, how='left', on='photo_id')
test_interaction['relative_time'] = ((test_interaction['time'] - test_interaction['start_time'])/(1e8/1000)).astype(int)
face_train = getface.get_photo_face_feature(mode = 0, has_flag =  False)
face_test = getface.get_photo_face_feature(mode = 1, has_flag =  False)
train_interaction = pd.merge(train_interaction,face_train,on=['photo_id'],how='left')
test_interaction = pd.merge(test_interaction,face_test,on=['photo_id'],how='left')
train_interaction = train_interaction.fillna(0)
test_interaction = test_interaction.fillna(0)
# In[2]:

data = pd.concat([train_interaction,test_interaction],axis = 0)
text = pd.read_csv('photo_text_df_full_10.csv')
data = pd.merge(data, text, how='left', on='photo_id')
del(text)
gc.collect()

fm = pd.read_csv("/home/wulikang/kuaishou/noprenn/fengmian/lxf_read_data/visual_32_mean_5.csv")
data = pd.merge(data, fm, how='left', on='photo_id')
fm_mean = pd.read_csv('./user_feature')
data = pd.merge(data, fm_mean, how='left', on='user_id')
data = data.fillna(0)
del(fm)
gc.collect()

# In[3]:
print('labelencoder')

from sklearn.preprocessing import LabelEncoder
le_rtime = LabelEncoder()
data['relative_time'] = le_rtime.fit_transform(data['relative_time'])

le_dtime = LabelEncoder()
data['duration_time'] = le_dtime.fit_transform(data['duration_time'])
data['time'] = data['time'] - data['time'].min()
max_time_value = data['time'].max()
data['time'] = (data['time']/(max_time_value/1000)).astype(int)
time_label = LabelEncoder()
data['time'] = time_label.fit_transform(data['time'])
wo_label = LabelEncoder()
data['woman_num'] = wo_label.fit_transform(data['woman_num'])
rate_label = LabelEncoder()
data['face_rate_mean'] = rate_label.fit_transform(data['face_rate_mean'])
age_label = LabelEncoder()
data['age_max'] = age_label.fit_transform(data['age_max'])
man_label = LabelEncoder()
data['man_num'] = man_label.fit_transform(data['man_num'])

visual_label = LabelEncoder()

data['visual_mean'] = visual_label.fit_transform(data['user_inter_all_num'])
print('leabel_end')

print('generatedoc')
def generate_doc(df,name,concat_name):
    res = df.astype(str).groupby(name)[concat_name].apply((lambda x :' '.join(x))).reset_index()
    res.columns = [name,'%s_doc'%concat_name]
    return res


user_doc = pd.read_csv(path+'fsnluser_doc.csv') 


print('doctocsv')
 


EMBEDDING_DIM_USER=64
EMBEDDING_DIM_PHOTO=64
eb_dim_rtime = 64
eb_dim_dtime = 64
eb_dim_time = 64
nb_users = data['user_id'].max()+1
nb_photos = data['photo_id'].max()+1
nb_rtime = data['relative_time'].nunique()
nb_dtime = data['duration_time'].nunique()
nb_time = data['time'].nunique()
print("num_users:"+str(nb_users))
print("num_photos"+str(nb_photos))
print("num_rtime"+str(nb_rtime))
print("num_dtime"+str(nb_dtime))
print("num_time"+str(nb_time))
nb_age_max = data['age_max'].nunique()
nb_woman_num = data['woman_num'].nunique()
nb_man_num = data['man_num'].nunique()
nb_rate_mean = data['face_rate_mean'].nunique()
print("num_age_max"+str(nb_age_max))
print("num_woamn_num"+str(nb_woman_num))
print("num_man_num"+str(nb_man_num))
print('num_face_rate_mean'+str(nb_rate_mean))
nb_visual_mean = data['visual_mean'].nunique()
print('num_visual'+str(nb_visual_mean))
########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation


#from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GRU, Embedding,SpatialDropout1D, Dropout, Activation,Bidirectional,TimeDistributed,CuDNNGRU
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop,Adam
from keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.layers import average,dot,maximum,multiply,add



# In[11]:


########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer   
from string import punctuation
from sklearn.metrics import roc_auc_score

#from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,Conv1D,AveragePooling1D,MaxPooling1D,Flatten,merge,TimeDistributed,ZeroPadding1D
from keras.layers.merge import concatenate,add
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler,Callback
from keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.layers import average,dot,maximum,multiply,add

class AucCallback(Callback):  #inherits from Callback
    
    def __init__(self, validation_data=(), patience=25,is_regression=False,best_model_name='best_keras.mdl',feval='roc_auc_score',batch_size=128,X_T=0):
        super(Callback, self).__init__()
        
        self.patience = patience
        self.X_test, self.y_test = validation_data  #tuple of validation X and y
        self.best = -np.inf
        self.wait = 0  #counter for patience
        self.best_model=None
        self.best_model_name = best_model_name
        self.is_regression = is_regression
        self.y_test = self.y_test#.astype(np.int)
        self.feval = feval
        self.batch_size = batch_size
        self.X_t =X_T
    def on_epoch_end(self, epoch, logs={}):
        # p = self.model.predict(self.X_test,batch_size=self.batch_size, verbose=0)#.ravel()
        p = []
        # for X_batch,y_batch in test_batch_generator(self.X_test,self.y_test,batch_size=self.batch_size):
        #     p.append(model.predict(X_batch,batch_size=batch_size))
        # p = np.concatenate(p).ravel()
        p = model.predict(self.X_test,batch_size=self.batch_size, verbose=1)
#        p1 = model.predict(self.X_test,batch_size=self.batch_size,verbose=1)
        current = 0.0
#        current1 = 0.0
        if self.feval=='roc_auc_score':

            current+= roc_auc_score(self.y_test.ravel(),p.ravel())
#            current1+= roc_auc_score(self.y_test.ravel(),p1.ravel())

        if current > self.best:
            self.best = current
            self.wait = 0
            print('Epoch %d Auc: %f | Best Auc: %f \n' % (epoch,current,self.best))
#            self.model.save_weights(self.best_model_name,overwrite=True)
            
            if(epoch >= 0 and epoch <=3):
                print('saving.....')
                y_sub = self.model.predict(self.X_t,batch_size=self.batch_size, verbose=1)
                submission = pd.DataFrame()
                submission['user_id'] = test_interaction['user_id']
                submission['photo_id'] = test_interaction['photo_id']
                submission['click_probability'] = y_sub
                submission['click_probability'].apply(lambda x:float('%.6f' % x))
                submission.to_csv('32epoch'+str(epoch)+'.txt',sep='\t',index=False,header=False)
            
        else:
            if self.wait >= self.patience:
                self.model.stop_training = True
                print('Epoch %05d: early stopping' % (epoch))
                
                
            self.wait += 1 #incremental the number of times without improvement
        print('Epoch %d Auc: %f | Best Auc: %f \n' % (epoch,current,self.best))
#        print('AUC1:'+str(current1))

# In[12]:

act = 'relu'


# In[13]:

########################################
## define the model structure
########################################
embedding_layer_user = Embedding(nb_users,
        EMBEDDING_DIM_USER)
#        weights=[embedding_matrix_user],
#        trainable=False)

embedding_layer_user2 = Embedding(nb_users,
        EMBEDDING_DIM_USER,
        trainable=True)

embedding_layer_rtime = Embedding(nb_rtime,
        eb_dim_rtime)

embedding_layer_dtime = Embedding(nb_dtime,
        eb_dim_dtime)

embedding_layer_time = Embedding(nb_time,
        eb_dim_time)

# In[60]:

MAX_SENTENCE_LENGTH = 30


# In[61]:


input_user = Input(shape=(1,), dtype='int32')
input_rtime = Input(shape=(1,), dtype='int32')
input_dtime = Input(shape=(1,),dtype='int32')
input_time = Input(shape=(1,),dtype='int32')
input_user_mean = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')

embedded_user= embedding_layer_user(input_user)
embedded_user2= embedding_layer_user2(input_user)
embedded_rtime = embedding_layer_rtime(input_rtime)
embedded_dtime = embedding_layer_dtime(input_dtime)
embedded_time = embedding_layer_time(input_time)
embedded_user2_agg = embedding_layer_user2(input_user_mean)



embedded_user = Flatten()(embedded_user)
embedded_user2 = Flatten()(embedded_user2)
embedded_rtime = Flatten()(embedded_rtime)
embedded_dtime = Flatten()(embedded_dtime)
embedded_time = Flatten()(embedded_time)

embedded_user2_mean = GlobalAveragePooling1D()(embedded_user2_agg)
embedded_user2_max = GlobalMaxPooling1D()(embedded_user2_agg)

eb_age_max_dim = 32
embedding_layer_age_max = Embedding(nb_age_max,
         eb_age_max_dim)
input_age_max = Input(shape=(1,), dtype='int32')
embedded_age_max= embedding_layer_age_max(input_age_max)

embedded_age_max = Flatten()(embedded_age_max)


eb_woman_num_dim = 16
embedding_layer_woman_num = Embedding(nb_woman_num,
         eb_woman_num_dim)
input_woman_num = Input(shape=(1,), dtype='int32')
embedded_woman_num= embedding_layer_woman_num(input_woman_num)
embedded_woman_num = Flatten()(embedded_woman_num)

eb_rate_mean_dim = 64
embedding_layer_rate_mean = Embedding(nb_rate_mean,
         eb_rate_mean_dim)
input_rate_mean = Input(shape=(1,), dtype='int32')
embedded_rate_mean= embedding_layer_rate_mean(input_rate_mean)
embedded_rate_mean = Flatten()(embedded_rate_mean)

eb_man_num_dim = 16
embedding_layer_man_num = Embedding(nb_man_num,
         eb_man_num_dim)
input_man_num = Input(shape=(1,), dtype='int32')
embedded_man_num= embedding_layer_man_num(input_man_num)
embedded_man_num = Flatten()(embedded_man_num)


eb_visual_mean_dim = 64
embedding_layer_visual_mean = Embedding(nb_visual_mean,
         eb_visual_mean_dim)
input_visual_mean = Input(shape=(1,), dtype='int32')
embedded_visual_mean= embedding_layer_visual_mean(input_visual_mean)
embedded_visual_mean = Flatten()(embedded_visual_mean)
input_text = Input(shape=(20,),dtype='float32')
ebd_text = Dense(64, activation=act)(input_text)

##fm

input_fm = Input(shape=(32,),dtype='float32')
ebd_fm = Dense(64,activation=act)(input_fm)



flatten_list = [
    embedded_user,
    #embedded_user2_mean,
    #embedded_user2_max,
    embedded_rtime,
    embedded_dtime,
    embedded_time,
    embedded_age_max,
    embedded_woman_num,
    embedded_rate_mean,
    #embedded_visual_mean,
    embedded_man_num,
    ebd_text,
    ebd_fm
]


# In[62]:

merged = concatenate(flatten_list,name='match_concat')
merged = BatchNormalization()(merged)
merged = Dense(280, activation=act)(merged)
merged = Dense(64,activation=act)(merged)
merged = Dropout(0.25)(merged)
preds = Dense(1, activation='sigmoid')(merged)
model = Model(inputs=[input_user,input_rtime,input_dtime,input_time,input_user_mean,
                       input_age_max,input_woman_num,input_rate_mean,input_man_num,input_visual_mean,input_text,input_fm],         outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
print(model.summary())



len_train = train_interaction.shape[0]
train = data[:len_train]
test = data[len_train:]
del(data)
gc.collect()
print(train.head(10))
print(test.head(10))
text_num = 10
text_name = []

for i in range(text_num):
    name = 'text_mean_'+str(i)
    text_name.append(name)

for i in range(text_num):
    name = 'text_max_'+str(i)
    text_name.append(name)


fm_num = 32
fm_name = []
for i in range(fm_num):
    name = 'fenmian_'+str(i)
    fm_name.append(name)

user_doc['photo_id'] = user_doc['photo_id'].astype(int)
user_doc['user_id_doc'] = user_doc['user_id_doc'].apply(lambda x:[int(s) for s in x.split(' ')])

train = pd.merge(train,user_doc,on='photo_id',how='left')
test = pd.merge(test,user_doc,on='photo_id',how='left')
del(user_doc)
gc.collect()


y = train_interaction['click'].values

skf =StratifiedKFold(n_splits=6, shuffle=True, random_state=1).split(train_interaction['user_id'],y)
for ind_tr, ind_te in skf:
    break
    
print('huafen')

train.fillna(0)
test.fillna(0)

print('huafen end')
train_user_mean = pad_sequences(train['user_id_doc'].values, maxlen=MAX_SENTENCE_LENGTH)
test_user_mean = pad_sequences(test['user_id_doc'].values, maxlen=MAX_SENTENCE_LENGTH)
del(train['user_id_doc'])
gc.collect()
del(test['user_id_doc'])
gc.collect()
print('Shape of train_user_mean tensor:', train_user_mean.shape)
print('Shape of test_user_mean tensor:', test_user_mean.shape)

del(train_interaction)
gc.collect()
X_train = [
    train['user_id'].values,
    train['relative_time'].values,
    train['duration_time'].values,
    train['time'].values,
    train_user_mean,
    train['age_max'].values,
    train['woman_num'].values,
    train['face_rate_mean'].values,
    train['man_num'].values,
    train['visual_mean'].values,
    train[text_name].values,
    train[fm_name].values
]
X_test = [
    train['user_id'].values[ind_te],
    train['relative_time'].values[ind_te],
    train['duration_time'].values[ind_te],
    train['time'].values[ind_te],
    train_user_mean[ind_te],
    train['age_max'].values[ind_te],
    train['woman_num'].values[ind_te],
    train['face_rate_mean'].values[ind_te],
    train['man_num'].values[ind_te],
    train['visual_mean'].values[ind_te],
    train[text_name].values[ind_te],
    train[fm_name].values[ind_te]
]
X_t = [
    test['user_id'].values,
    test['relative_time'].values,
    test['duration_time'].values,
    test['time'].values,
    test_user_mean,
    test['age_max'].values,
    test['woman_num'].values,
    test['face_rate_mean'].values,
    test['man_num'].values,
    test['visual_mean'].values,
    test[text_name].values,
    test[fm_name].values
#    test['test_time'].values
]

y_train = y
y_test = y[ind_te]
print(np.shape(y_train))
print(y_train)
print(np.shape(y_test))
print(y_test)
del(train)
del(test)
gc.collect()
# In[ ]:

STAMP = 'base32'
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=5)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

batch_size = 1024

def schedule(index):
    if index<10:
        return 0.001
    else:
        return 0.0001


lrs = LearningRateScheduler(schedule)
auc_callback = AucCallback(validation_data=(X_test,y_test),patience=5,best_model_name=bst_model_path,batch_size=batch_size,X_T=X_t)
callbacks = [

    auc_callback,
    ]



hist = model.fit(X_train, y_train,         validation_data=(X_test, y_test),         epochs=5, batch_size=batch_size, shuffle=True,          callbacks=callbacks)


# In[ ]:

model.load_weights(bst_model_path)

y_pred = model.predict(X_test)
score = roc_auc_score(y_test,y_pred)
print('auc score:%s'%score)

from sklearn.metrics import log_loss
score = log_loss(y_test,y_pred)
print('logloss score:%s'%score)


