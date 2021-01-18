

# 数据分析

### 特征分析

查看特征和标签的关系

```
sns.barplot(data=train,x='Pclass',y='Survived')
```

```
# 把 col 根据取值，分开为 n 张图
sns.factorplot('Pclass',col='Embarked',data=train,kind='count',size=3)
```



#### 缺失值填充

填充为 Unknown，众数，这个数据所属某个类别的平均数。

#### 特征粗粒化

根据名字的姓氏，提取阶层

根据其他特征的前缀，提取大类特征

同一票号的乘客数量可能不同，可能也与乘客生存率有关系

年龄可以使用模型预测填充



通过协方差矩阵，查看特征和标签的相关性系数，筛选特征。



















### 特征构造


### 查看特征的相关性
```python
train_corr_df=train_df.corr()
train_corr_df['label'].sort_values()
```

### 特征相关性的热力图
```python
import seaborn as sns
#热力图，查看label与其他特征间相关性大小
plt.figure(figsize=(9,9))
sns.heatmap(train_df.corr(),cmap='BrBG',annot=True,)
           # linewidths=)
plt.xticks(rotation=45)
```

### 初步模型交叉验证和对比
```python
#导入机器学习算法库
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold

#设置kfold，交叉采样法拆分数据集
kfold=StratifiedKFold(n_splits=10)

#汇总不同模型算法
classifiers=[]
classifiers.append(SVC())
classifiers.append(DecisionTreeClassifier())
classifiers.append(RandomForestClassifier())
classifiers.append(ExtraTreesClassifier())
classifiers.append(GradientBoostingClassifier())
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())
classifiers.append(LinearDiscriminantAnalysis())

#不同机器学习交叉验证结果汇总
cv_results=[]
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier,experData_X,experData_y,
                                      scoring='accuracy',cv=kfold,n_jobs=-1))

#求出模型得分的均值和标准差
cv_means=[]
cv_std=[]
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
#汇总数据
cvResDf=pd.DataFrame({'cv_mean':cv_means,
                     'cv_std':cv_std,
                     'algorithm':['SVC','DecisionTreeCla','RandomForestCla','ExtraTreesCla',
                                  'GradientBoostingCla','KNN','LR','LinearDiscrimiAna']})

cvResDf
```

可视化不同算法的表现情况
```python
cvResFacet=sns.FacetGrid(cvResDf.sort_values(by='cv_mean',ascending=False),sharex=False,
            sharey=False,aspect=2)
cvResFacet.map(sns.barplot,'cv_mean','algorithm',**{'xerr':cv_std},
               palette='muted')
cvResFacet.set(xlim=(0.7,0.9))
cvResFacet.add_legend()
```
选择准确率高，方差低的算法进行调参

### 选择模型进行调参
```python
#GradientBoostingClassifier模型
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }
modelgsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, 
                                     scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsGBC.fit(experData_X,experData_y)

#LogisticRegression模型
modelLR=LogisticRegression()
LR_param_grid = {'C' : [1,2,3],
                'penalty':['l1','l2']}
modelgsLR = GridSearchCV(modelLR,param_grid = LR_param_grid, cv=kfold, 
                                     scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsLR.fit(experData_X,experData_y)

#modelgsGBC模型
print('modelgsGBC模型得分为：%.3f'%modelgsGBC.best_score_)
#modelgsLR模型
print('modelgsLR模型得分为：%.3f'%modelgsLR.best_score_)
```



### 模型集成
查看 staking.py 和 stacking_class.py








**参考**

- [知乎-泰坦尼克号](https://zhuanlan.zhihu.com/p/50194676)
- [CSDN-泰坦尼克号](https://blog.csdn.net/weiyongle1996/article/details/78038350)

