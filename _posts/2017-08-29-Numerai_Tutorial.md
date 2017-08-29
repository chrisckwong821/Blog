---
layout: post
header:
  teaser: /assets/images/my-awesome-post-teaser.jpg
title: Understanding HSI - III - The Best EWMA Pair
categories: [Machine Learning]
tags: [Numerai, Machine Learning, Data Science, Python]
fullview: true
comments: true
---

Numerai is an online machine learning tournament which is operated by numerai. You may refer to this article for an introduction. In this article, I want to describe how I approach the problem, and build up a set of tools that helps me to rapidly iterate over different algorithms for testing, modification and ensembing.


```python
import os.path
import pandas as pd
class models:
    def __init__(self,tournament=70):
        parentdir = os.path.join(os.path.abspath(os.getcwd()),os.pardir)
        path = os.path.join(parentdir,"T{}/".format(tournament))
        
        self.training_data = pd.read_csv(path + "numerai_training_data.csv",header=0)
        self.test_data = pd.read_csv(path + "numerai_tournament_data.csv",header=0)
print(models().training_data.head(1))        
```

          id   era data_type  feature1  feature2  feature3  feature4  feature5  \
    0  22364  era1     train   0.52781   0.48414   0.61717   0.41186   0.38068   
    
       feature6  feature7   ...    feature13  feature14  feature15  feature16  \
    0   0.46056   0.60864   ...      0.41597    0.66218     0.5372    0.42039   
    
       feature17  feature18  feature19  feature20  feature21  target  
    0    0.57638    0.62859    0.54002    0.52455    0.51074       1  
    
    [1 rows x 25 columns]


The data contains 21 features range from 0 to 1, with the binary target(0,1). Numerai claims that they encrypted financial data into the dataset so it is more than simple time-series data. Each row contains a unique id, an era which label its type, and whether it belongs to train/test data.


```python
#training data
        self.X_train = self.training_data[[f for f in list(self.training_data) if "feature" in f]]
        self.y_train = self.training_data['target']
#test data(part of prediction data)
        self.X_test = self.test_data[[f for f in list(self.training_data) if "feature" in f]][:16686]
        self.y_test = self.test_data['target'][:16686]
#prediction data
        self.X_prediction = self.test_data[[f for f in list(self.training_data) if "feature" in f]]
```


```python

```
