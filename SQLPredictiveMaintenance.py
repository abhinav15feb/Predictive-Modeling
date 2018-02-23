
# coding: utf-8

# # Data Preprocessing
# Step 1: Conver datetime feild to_datetime for all datafiles
# Step 2: Merge Machines data to telemetry data

# In[2]:


import tensorflow, keras
import pandas
import numpy

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM


# In[ ]:


#errors = pandas.read_csv('/Users/abhi/Desktop/Sem 4/BNAD/SQL-Server-R-Services-Samples/PredictiveMaintanenceModelingGuide/Data/errors.csv')
#failures = pandas.read_csv('/Users/abhi/Desktop/Sem 4/BNAD/SQL-Server-R-Services-Samples/PredictiveMaintanenceModelingGuide/Data/failures.csv')
#machines = pandas.read_csv('/Users/abhi/Desktop/Sem 4/BNAD/SQL-Server-R-Services-Samples/PredictiveMaintanenceModelingGuide/Data/machines.csv')
maintain = pandas.read_csv('/Users/abhi/Desktop/Sem 4/BNAD/SQL-Server-R-Services-Samples/PredictiveMaintanenceModelingGuide/Data/maint.csv')
telemetry = pandas.read_csv('/Users/abhi/Desktop/Sem 4/BNAD/SQL-Server-R-Services-Samples/PredictiveMaintanenceModelingGuide/Data/telemetry.csv')


# In[ ]:


errors['datetime']=pandas.to_datetime(errors['datetime'])
telemetry['datetime']=pandas.to_datetime(telemetry['datetime'])
maintain['datetime']=pandas.to_datetime(maintain['datetime'])
failures['datetime']=pandas.to_datetime(failures['datetime'])


# # Maintain pivot

# In[ ]:


comp4 = maintain.iloc[numpy.where(maintain['comp']=='comp4')]
comp3 = maintain.iloc[numpy.where(maintain['comp']=='comp3')]
comp2 = maintain.iloc[numpy.where(maintain['comp']=='comp2')]
comp1 = maintain.iloc[numpy.where(maintain['comp']=='comp1')]


# In[ ]:


maintain_pivot_t = (pandas.merge(comp4, comp3, on = ['datetime', 'machineID'], how = 'outer'))
maintain_pivot_t = maintain_pivot_t.rename(columns={maintain_pivot_t.columns[2]:'comp4', maintain_pivot_t.columns[3]:'comp3'})


# In[ ]:


maintain_pivot_t2= pandas.merge(maintain_pivot_t, comp2, on = ['datetime', 'machineID'], how = 'outer')


# In[ ]:


maintain_pivot= pandas.merge(maintain_pivot_t2, comp1, on = ['datetime', 'machineID'], how = 'outer')


# In[ ]:


maintain_pivot.rename(columns={maintain_pivot.columns[2]:'comp4', maintain_pivot.columns[3]:'comp3'})


# In[ ]:


maintain_pivot = maintain_pivot.fillna(0)
maintain_pivot.loc[maintain_pivot['comp1'] == 'comp1', 'comp1'] = 1
maintain_pivot.loc[maintain_pivot['comp2'] == 'comp2', 'comp2'] = 1
maintain_pivot.loc[maintain_pivot['comp3'] == 'comp3', 'comp3'] = 1
maintain_pivot.loc[maintain_pivot['comp4'] == 'comp4', 'comp4'] = 1


# In[ ]:


maintain_pivot = maintain_pivot.rename(columns={maintain_pivot.columns[2]:'Mantain_comp4',maintain_pivot.columns[3]:'Mantain_comp3',maintain_pivot.columns[4]:'Mantain_comp2', maintain_pivot.columns[5]:'Mantain_comp1'})


# In[ ]:


maintain_pivot


# # Error Pivot

# In[ ]:


error4 = errors.iloc[numpy.where(errors['errorID']=='error4')]
error3 = errors.iloc[numpy.where(errors['errorID']=='error3')]
error2 = errors.iloc[numpy.where(errors['errorID']=='error2')]
error1 = errors.iloc[numpy.where(errors['errorID']=='error1')]


# In[ ]:


errors_pivot_t = (pandas.merge(error4, error3, on = ['datetime', 'machineID'], how = 'outer'))
errors_pivot_t = errors_pivot_t.rename(columns={errors_pivot_t.columns[2]:'error4', errors_pivot_t.columns[3]:'error3'})


# In[ ]:


errors_pivot_t2= pandas.merge(errors_pivot_t, error2, on = ['datetime', 'machineID'], how = 'outer')
errors_pivot= pandas.merge(errors_pivot_t2, error1, on = ['datetime', 'machineID'], how = 'outer')


# In[ ]:


errors_pivot = errors_pivot.rename(columns={errors_pivot.columns[4]:'error2', errors_pivot.columns[5]:'error1'})


# In[ ]:


errors_pivot = errors_pivot.fillna(0)
errors_pivot.loc[errors_pivot['error1'] == 'error1', 'error1'] = 1
errors_pivot.loc[errors_pivot['error2'] == 'error2', 'error2'] = 1
errors_pivot.loc[errors_pivot['error3'] == 'error3', 'error3'] = 1
errors_pivot.loc[errors_pivot['error4'] == 'error4', 'error4'] = 1


# In[ ]:


print(errors_pivot)


# # failures Pivot

# In[ ]:


comp4 = failures.iloc[numpy.where(failures['failure']=='comp4')]
comp3 = failures.iloc[numpy.where(failures['failure']=='comp3')]
comp2 = failures.iloc[numpy.where(failures['failure']=='comp2')]
comp1 = failures.iloc[numpy.where(failures['failure']=='comp1')]


# In[ ]:


failures_pivot_t = (pandas.merge(comp4, comp3, on = ['datetime', 'machineID'], how = 'outer'))
failures_pivot_t = failures_pivot_t.rename(columns={failures_pivot_t.columns[2]:'comp4', failures_pivot_t.columns[3]:'comp3'})


# In[ ]:


failures_pivot_t2= pandas.merge(failures_pivot_t, comp2, on = ['datetime', 'machineID'], how = 'outer')
failures_pivot= pandas.merge(failures_pivot_t2, comp1, on = ['datetime', 'machineID'], how = 'outer')


# In[ ]:


failures_pivot = failures_pivot.rename(columns={failures_pivot.columns[4]:'comp2', failures_pivot.columns[5]:'comp1'})


# In[ ]:


failures_pivot = failures_pivot.fillna(0)
failures_pivot.loc[failures_pivot['comp1'] == 'comp1', 'comp1'] = 1
failures_pivot.loc[failures_pivot['comp2'] == 'comp2', 'comp2'] = 1
failures_pivot.loc[failures_pivot['comp3'] == 'comp3', 'comp3'] = 1
failures_pivot.loc[failures_pivot['comp4'] == 'comp4', 'comp4'] = 1


# In[ ]:


failures_pivot = failures_pivot.rename(columns={failures_pivot.columns[2]:'fail_comp4',failures_pivot.columns[3]:'fail_comp3',failures_pivot.columns[4]:'fail_comp2', failures_pivot.columns[5]:'fail_comp1'})


# In[ ]:


failures_pivot


# # Data combining process 

# In[ ]:


fulldata = pandas.merge(telemetry, machines, on = ['machineID'], how = 'outer')


# In[ ]:


fulldata


# In[ ]:


fulldata1 = pandas.merge(fulldata, errors_pivot, on = ['datetime','machineID'], how = 'outer')
fulldata2 = pandas.merge(fulldata1, maintain_pivot, on = ['datetime','machineID'], how = 'outer')
fulldata3 = pandas.merge(fulldata2, failures_pivot, on = ['datetime','machineID'], how = 'outer')


# In[ ]:


fulldata3 = fulldata3.fillna(0)


# In[ ]:


print (telemetry.count())
print (errors.count())
print (maintain.count())
print (failures.count())
print(fulldata3.count())


# In[ ]:


fulldata3.to_csv('final_sql_data.csv', sep = ',',encoding='utf-8', index=False)


# # Read data and continue building Remaining Useful Life feature for failure and maintenance prediction

# In[4]:


final_data = pandas.read_csv('final_sql_data.csv')


# Process -
# 1. Add remaining useful life
#    if failure exists and last maintenance exist or assume start of time ie first entry of data for perticular machine in telemetry
#    Failure of component - Last Maintenance of component
# 2. generate label columns for training data
#    we will only make use of "label1" for binary classification, 
#    while trying to answer the question: is a specific engine going to fail within w1 cycles?
#    w1 = 30
#     w0 = 15
#     train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0 )
#     train_df['label2'] = train_df['label1']
#     train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2
#     

# In[7]:


final_data = final_data.sort_values('datetime')


# In[34]:


final_data['DateTillMaintain_comp1'] =final_data[(final_data['Mantain_comp1']==1)].datetime
final_data['DateTillMaintain_comp2'] =final_data[(final_data['Mantain_comp2']==1)].datetime
final_data['DateTillMaintain_comp3'] =final_data[(final_data['Mantain_comp3']==1)].datetime
final_data['DateTillMaintain_comp4'] =final_data[(final_data['Mantain_comp4']==1)].datetime

final_data['DateTillFailure_comp1'] =final_data[(final_data['fail_comp1']==1)].datetime
final_data['DateTillFailure_comp2'] =final_data[(final_data['fail_comp2']==1)].datetime
final_data['DateTillFailure_comp3'] =final_data[(final_data['fail_comp3']==1)].datetime
final_data['DateTillFailure_comp4'] =final_data[(final_data['fail_comp4']==1)].datetime


# In[35]:


final_data = final_data.fillna(method='bfill')


# In[36]:



final_data ['RULtillFail_comp1'] = (pandas.to_datetime(final_data['DateTillFailure_comp1'])- pandas.to_datetime(final_data['datetime'])).astype('timedelta64[h]')
final_data ['RULtillFail_comp2'] = (pandas.to_datetime(final_data['DateTillFailure_comp2'])- pandas.to_datetime(final_data['datetime'])).astype('timedelta64[h]')
final_data ['RULtillFail_comp3'] = (pandas.to_datetime(final_data['DateTillFailure_comp3'])- pandas.to_datetime(final_data['datetime'])).astype('timedelta64[h]')
final_data ['RULtillFail_comp4'] = (pandas.to_datetime(final_data['DateTillFailure_comp4'])- pandas.to_datetime(final_data['datetime'])).astype('timedelta64[h]')

final_data ['RULtillMaintain_comp1'] = (pandas.to_datetime(final_data['DateTillMaintain_comp1'])- pandas.to_datetime(final_data['datetime'])).astype('timedelta64[h]')
final_data ['RULtillMaintain_comp2'] = (pandas.to_datetime(final_data['DateTillMaintain_comp2'])- pandas.to_datetime(final_data['datetime'])).astype('timedelta64[h]')
final_data ['RULtillMaintain_comp3'] = (pandas.to_datetime(final_data['DateTillMaintain_comp3'])- pandas.to_datetime(final_data['datetime'])).astype('timedelta64[h]')
final_data ['RULtillMaintain_comp4'] = (pandas.to_datetime(final_data['DateTillMaintain_comp4'])- pandas.to_datetime(final_data['datetime'])).astype('timedelta64[h]')



# In[37]:


final_data.columns


# In[38]:


final_data.to_csv('predictors_sql_data.csv', sep = ',',encoding='utf-8', index=False)


# In[3]:


final_data= pandas.read_csv('predictors_sql_data.csv')


# In[4]:


final_data = final_data.loc[((final_data['datetime'])>='2015-01-01') & ((final_data['datetime'])<'2016-01-01')]


# In[5]:


w1 = 30
w0 = 15
final_data['comp1_1stwarning'] = numpy.where(final_data['RULtillFail_comp1'] <= w1, 1, 0 )
final_data['comp2_1stwarning'] = numpy.where(final_data['RULtillFail_comp2'] <= w1, 1, 0 )
final_data['comp3_1stwarning'] = numpy.where(final_data['RULtillFail_comp3'] <= w1, 1, 0 )
final_data['comp4_1stwarning'] = numpy.where(final_data['RULtillFail_comp4'] <= w1, 1, 0 )
final_data['comp1_2ndwarning'] = final_data['comp1_1stwarning']
final_data['comp2_2ndwarning'] = final_data['comp2_1stwarning']
final_data['comp3_2ndwarning'] = final_data['comp3_1stwarning']
final_data['comp4_2ndwarning'] = final_data['comp4_1stwarning']


final_data.loc[final_data['RULtillFail_comp1'] <= w0, 'comp1_2ndwarning'] = 2
final_data.loc[final_data['RULtillFail_comp2'] <= w0, 'comp2_2ndwarning'] = 2
final_data.loc[final_data['RULtillFail_comp3'] <= w0, 'comp3_2ndwarning'] = 2
final_data.loc[final_data['RULtillFail_comp4'] <= w0, 'comp4_2ndwarning'] = 2


# In[5]:


final_data.min()


# In[40]:


final_data.count()


# In[6]:


final_data.loc[(final_data['RULtillFail_comp1'])<0, ['datetime','DateTillMaintain_comp1',
       'DateTillMaintain_comp2', 'DateTillMaintain_comp3',
       'DateTillMaintain_comp4', 'DateTillFailure_comp1',
       'DateTillFailure_comp2', 'DateTillFailure_comp3',
       'DateTillFailure_comp4']]


# In[6]:


#final_data=final_data.loc[final_data['volt']>0]
final_data = final_data.dropna()


# In[66]:


final_data.columns


# In[7]:


cols =['datetime', 'machineID', 'volt', 'rotate', 'pressure', 'vibration',
       'model', 'age', 'error4', 'error3', 'error2', 'error1', 'Mantain_comp4',
       'Mantain_comp3', 'Mantain_comp2', 'Mantain_comp1', 'fail_comp4',
       'fail_comp3', 'fail_comp2', 'fail_comp1','RULtillFail_comp1', 'RULtillFail_comp2',
       'RULtillFail_comp3', 'RULtillFail_comp4', 'RULtillMaintain_comp1',
       'RULtillMaintain_comp2', 'RULtillMaintain_comp3',
       'RULtillMaintain_comp4', 'comp1_1stwarning', 'comp2_1stwarning',
       'comp3_1stwarning', 'comp4_1stwarning', 'comp1_2ndwarning',
       'comp2_2ndwarning', 'comp3_2ndwarning', 'comp4_2ndwarning']


# In[8]:


train_data = final_data.loc[(final_data['datetime'])<'2015-08-01',cols ]
test_data = final_data.loc[(final_data['datetime'])>='2015-08-01',cols]


# # Method 1: LSTM network

# In[78]:


cols_normalize = train_data.columns.difference(['datetime','machineID','model','RULtillFail_comp1', 'RULtillFail_comp2',
       'RULtillFail_comp3', 'RULtillFail_comp4', 'RULtillMaintain_comp1',
       'RULtillMaintain_comp2', 'RULtillMaintain_comp3',
       'RULtillMaintain_comp4','comp1_1stwarning', 'comp2_1stwarning',
       'comp3_1stwarning', 'comp4_1stwarning', 'comp1_2ndwarning',
       'comp2_2ndwarning', 'comp3_2ndwarning', 'comp4_2ndwarning'])
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df = pandas.DataFrame(min_max_scaler.fit_transform(train_data[cols_normalize]), 
                             columns=cols_normalize, 
                             index=train_data.index)
join_df = train_data[train_data.columns.difference(cols_normalize)].join(norm_train_df)
train_data = join_df.reindex(columns = train_data.columns)


# In[81]:


norm_test_df = pandas.DataFrame(min_max_scaler.transform(test_data[cols_normalize]), 
                            columns=cols_normalize, 
                            index=test_data.index)
test_join_df = test_data[test_data.columns.difference(cols_normalize)].join(norm_test_df)
test_data = test_join_df.reindex(columns = test_data.columns)
test_data = test_data.reset_index(drop=True)


# In[82]:


# pick a large window size of 50 cycles
sequence_length = 50

# function to reshape features into (samples, time steps, features) 
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,112),(50,192)
    # 0 50 -> from row 0 to row 50
    # 1 51 -> from row 1 to row 51
    # 2 52 -> from row 2 to row 52
    # ...
    # 111 191 -> from row 111 to 191
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]
        


# In[95]:


sequence_cols = ['volt', 'rotate', 'pressure', 'vibration', 'age']


# In[96]:


seq_gen = (list(gen_sequence(train_data[train_data['machineID']==id], sequence_length, sequence_cols)) 
           for id in train_data['machineID'].unique())


# In[97]:


seq_array = numpy.concatenate(list(seq_gen)).astype(numpy.float32)
seq_array.shape


# In[98]:


def gen_labels(id_df, seq_length, label):
    # For one id I put all the labels in a single matrix.
    # For example:
    # [[1]
    # [4]
    # [1]
    # [5]
    # [9]
    # ...
    # [200]] 
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target. 
    return data_matrix[seq_length:num_elements, :]


# In[100]:


# generate labels
label_gen = [gen_labels(train_data[train_data['machineID']==id], sequence_length, ['comp1_1stwarning']) 
             for id in train_data['machineID'].unique()]
label_array = numpy.concatenate(label_gen).astype(numpy.float32)
label_array.shape


# In[103]:


model_path = 'binary_model.h5'
nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]

model = Sequential()

model.add(LSTM(
         input_shape=(sequence_length, nb_features),
         units=100,
         return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
          units=50,
          return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=nb_out, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# fit the network


# In[ ]:


history = model.fit(seq_array, label_array, epochs=100, batch_size=200, validation_split=0.05, verbose=2,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
          )


# In[ ]:


# list all data in history
print(history.history.keys())

# summarize history for Accuracy
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig("model_accuracy.png")



# In[ ]:


fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig("model_loss.png")


# In[ ]:


# training metrics
scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
print('Accurracy: {}'.format(scores[1]))

# make predictions and compute confusion matrix
y_pred = model.predict_classes(seq_array,verbose=1, batch_size=200)
y_true = label_array


# # Method 2: random forest - not as time series

# In[10]:


from sklearn.ensemble import RandomForestClassifier


# In[11]:


cls = RandomForestClassifier()


# In[19]:


train_data.loc[:,['datetime', 'machineID', 'volt', 'rotate', 'pressure', 'vibration','model', 'age']]


# In[12]:


cls.fit(train_data[['machineID', 'volt', 'rotate', 'pressure', 'vibration', 'age']],train_data[['RULtillFail_comp1']])


# In[38]:


train_data.loc[:,['machineID', 'volt', 'rotate', 'pressure', 'vibration','model', 'age','RULtillFail_comp1']]


# In[13]:


print(cls.feature_importances_)


# In[ ]:


predictions = cls.predict(test_data[['machineID', 'volt', 'rotate', 'pressure', 'vibration', 'age']])


# In[ ]:


errors = abs(predictions - test_data[['RULtillFail_comp1']])


# In[ ]:


round(numpy.mean(errors), 2)

