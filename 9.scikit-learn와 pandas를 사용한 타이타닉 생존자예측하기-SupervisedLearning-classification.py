
# coding: utf-8

# In[1]:


import pandas as pd


# ## 1. Load Dataset (train, test)

# In[2]:


#훈련 데이터 가져오기
train = pd.read_csv('data/train.csv',index_col='PassengerId')
print(train.shape)
print(train.info())
train.head(2)


# In[3]:


#테스트 데이터 가져오기
test = pd.read_csv('data/test.csv',index_col='PassengerId')
print(test.shape)
print(test.info())
test.head(2)


# ## 2. Data PreProcessing

# In[4]:


# 성별(sex) 값이 male이면 열의 값을 0으로 , female이면 열의 값을 1로 set 
train.columns


# In[5]:


#Series 객체
print(type(train.loc[train['Sex'] == 'male','Sex']))
train.loc[train['Sex'] == 'male','Sex'] = 0
train.loc[train['Sex'] == 'female','Sex'] = 1


# In[6]:


test.loc[test['Sex'] == 'male','Sex'] = 0
test.loc[test['Sex'] == 'female','Sex'] = 1


# In[8]:


#DataFrame
#print(type(train.loc[train['Sex'] == 'male',['Sex','Name']]))
#train.loc[train['Sex'] == 'male',['Sex','Name']]
train.head()
test.head()


# In[11]:


#Embarked 컬럼 unique 값 
train['Embarked'].unique()
#C == 0, S == 1, Q == 2
#S + S = Q, 2 * S = Q (x)

#One Hot Encoding
#C=[True,False,False]
#S=[False,True,False]
#Q=[False,Fals,True]
#Embarked_C 컬럼을 추가하고 Embarked 값이 "C"이면 True추가 아니면 False를 추가한다.
train['Embarked_C'] = train['Embarked'] == "C"
train['Embarked_S'] = train['Embarked'] == "S"
train['Embarked_Q'] = train['Embarked'] == "Q"
train.loc[:,['Embarked','Embarked_C','Embarked_S','Embarked_Q']].head(10)


# In[12]:


test['Embarked_C'] = test['Embarked'] == "C"
test['Embarked_S'] = test['Embarked'] == "S"
test['Embarked_Q'] = test['Embarked'] == "Q"
test.loc[:,['Embarked','Embarked_C','Embarked_S','Embarked_Q']].head(10)


# In[14]:


#요금(Fare) 값이 null인 경우에 값을 0으로 변경
test.loc[test['Fare'].isnull(),'Fare'] = 0
test[test['Fare'].isnull()]


# ## 3. train & predict

# In[16]:


#필요한 컬럼을 선택
feature_names = ['Sex','Fare','Pclass','Embarked_C','Embarked_S','Embarked_Q']
feature_names


# In[17]:


#train Dataframe에서 feature_names의 컬럼을 가져와서 X_train DataFrame 저장
#훈련데이터 생성
X_train = train[feature_names]
print(X_train.shape)
X_train.head()


# In[18]:


#테스트 데이터 생성
X_test = test[feature_names]
print(X_test.shape)
X_test.head()


# In[19]:


#훈련데이터의 레이블(답) 생성
#train DataFrame에서 Survived 컬럼을 가져와서 y_train에 저장
label_name='Survived'
y_train = train[label_name]
print(y_train.shape)
y_train.head()


# In[20]:


# DecisionTreeClassifier 알고리즘 선택
from sklearn.tree import DecisionTreeClassifier

#의사 결정트리의 최대깊이를 5로 설정
model = DecisionTreeClassifier(max_depth=5,random_state=0)
model


# In[22]:


#학습하기
model.fit(X_train,y_train)


# In[23]:


#예측하기
predictions = model.predict(X_test)
print(predictions.shape)
predictions


# ## 3.1 Visualize

# In[29]:


from sklearn.tree import export_graphviz
import graphviz

export_graphviz(model,feature_names=feature_names,class_names=["Perish","Survived"],               out_file="decision-tree.dot")
with open("decision-tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)    


# ## 4. Submission

# In[24]:


# 제출하기 위해 제공된 gender_submission.csv 데이터를 load
submit = pd.read_csv('data/gender_submission.csv',index_col='PassengerId')
print(submit.shape)
submit.head()


# In[25]:


#submit dataframe의 Survived 컬럼을 predictions(예측한값)으로 변경
submit['Survived'] = predictions
print(submit.shape)
submit.head()


# In[26]:


#submit dataframe을 csv 파일로 저장해서 kaggle에 제출
submit.to_csv('data/result01.csv')

