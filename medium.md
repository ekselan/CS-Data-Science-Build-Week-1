# K-Nearest Neighbors with base Python
Functional and Object-Oriented Implementations for Classification

*Link to blog post on medium: (https://medium.com/@aaron.watkinsjr/k-nearest-neighbors-with-base-python-72a2a7b5f3cb?sk=fea45f84b5d1a67748fbc8c9feb9fe42)*

---

This post will explore two implementations of the K-Nearest Neighbors algorithm in base python (without scikit-learn), and compare classification results on the iris dataset with those of a scikit-learn implementation.

Nearest neighbors is a relatively simple, but versatile algorithm that can be used for both regression and classification problems. In the case of classifying labels for example, the concept behind the algorithm is to compare the distance from a new observation to that of each observation in a training set, and return the "closest" k-neighbors of the new observation.

The "k" in k-neighbors would represent the number of neighbors to pull, and this is where some of the optionality would begin. There's variation in how a nearest neighbors model determines distance, generates a prediction, and more. The implementations in this post will explore basic setups and use Euclidian Distance as the distance measure.

---

### Functional Approach

The first function we'll need is distance calculator to find the euclidian distance, which is "a measure of the true straight line distance between two points in Euclidean space." We can calculate this by finding the sum of squared differences between the two points, and getting the square root of that. So in effect, a value of 0 would indicate no difference between the two points, and the greater the value, the greater the difference.

```py
from math import sqrt
def euclidean_distance(row1, row2):
    
    # 0.0 so that distance will be float
    distance = 0.0
    
    # loop for columns
    for i in range(len(row1) - 1):
        # squared difference between the two vectors
        distance += (row1[i] - row2[i])**2
        
    return sqrt(distance)
```

We'll also need a function to find the neighbors to a new observation. Keeping in mind how the nearest neighbor algorithm works, this function will need to receive a training dataset as input, as well as a new observation, and the number of neighbors to retrieve. It'll also prove useful to prep this function to handle another datatype besides an array. The function will work by collecting the `euclidean_distance` of the new observation and each observation in the training set. It will then sort those distances and return the k-lowest distances.

```py
def get_neighbors(train, new_obs, k):
    """
    Locates most similar neighbors via euclidian distance.
    
    Params: 
        
        train: a dataset
        
        new_obs: observation for which neighbors are to be found
        
        k: k-neighbors; the number of neighbors to be found (int)
    """
    
    distances = []
    neighbors = []
    # Rules for whether or not train is a pandas.DataFrame
    if type(train) == pd.core.frame.DataFrame:
        
        for i,row in train.iterrows():
            # calculate distance
            d = euclidean_distance(new_obs, list(row))
            
            # fill list with tuples of row index and distance
            distances.append((i, d))
            # sort distances by second value in tuple
            distances.sort(key=lambda tup: tup[1])
    
    else:
        
        for i,row in enumerate(train):
            # calculate distance
            d = euclidean_distance(new_obs, row)
            # fill list with tuples of row index and distance
            distances.append((i, d))
            # sort distances by second value in tuple
            distances.sort(key=lambda tup: tup[1])
    for i in range(k):
        # Grabs k-records from distances list
        neighbors.append(distances[i])
return neighbors
```

The last two functions we'll need are prediction and scoring functions. To get a "prediction" from a nearest neighbors algorithm, two common approaches are to either use the label from the closest neighbor as the prediction, or to use a "voting" system to get the most popular label among the closest k-neighbors. In a binary classification example, if the closest 5 neighbors had labels of `0,0,1,1,1`, then the "voting" system would return `1` as the prediction. In that same example, simply using the label for the closest neighbor would return `0` as the prediction. 

Which approach is most applicable will depend on the problem, but if the training set is large enough and well groomed, using the single-closest neighbor label can achieve high accuracies. This is the approach demonstrated here.

```py
def predict_classification(train, new_obs, k):
    """
    Predicts class lbl on new_obs from provided training data.
    
    Params: 
        
        train: a pandas.DataFrame, or array
        
        new_obs: observation for which neighbors are to be found
        
        k: k-neighbors; the number of neighbors to be found (int)
    """
    # Compile list of neighbors
    neighbors = get_neighbors(train, new_obs, k)
    
    # Grab index of the closest neighbor
    n_index = neighbors[0][0]
    
    # Add rules for if train is a pandas.DataFrame
    if type(train) == pd.core.frame.DataFrame:
        # Assumes labels are in last column of dataframe
        loc = train.columns[-1]
        pred = train[loc][n_index]
    else:
        # Prediction is the label from train record at n_index  
        # Assumes label is at end of record.
        pred = train[n_index][-1]
    
    return pred
```

The accuracy can be determined by simply counting the number of correct predictions, dividing that by the number of total predictions, and multiplying by 100 to get a score between 0 and 100.

```py
def accuracy_metric(x, y):
    """
    Calculates accuracy of predictions (on classification problems).
    
    Params:
        
        x: actual, or correct labels
        
        y: predicated labels
    """
    
    correct = 0
    
    for i in range(len(x)):
        # Rules for if `x` is a pandas.Series
        if type(x) == pd.core.series.Series:
            if x.iloc[i] == y[i]:
                correct += 1
            
        else:
            if x[i] == y[i]:
                correct += 1
                
    return correct / float(len(x)) * 100.0
```

To test this implementation, the next step was to load the iris dataset, split it into train and test sets, define X-feature matrices and y-target vectors, and generate predictions to gauge performance.

```py
import pandas as pd
from sklearn.model_selection import train_test_split
# Load iris dataset
cols = [
    "sepal_len",
    "sepal_wid",
    "petal_len",
    "petal_wid",
    "class"
]
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=cols)
# Cleanup class names
names = []
for x in df["class"]:
    x = x.replace("Iris-","")
    names.append(x)
    
df["class"] = names
# Encode class names
labels = []
for x in df["class"]:
    x = x.replace("versicolor","0")
    x = x.replace("virginica","1")
    x = x.replace("setosa","2")
    x = int(x)
    labels.append(x)
    
df["class"] = labels
# Train test split
train, test = train_test_split(df, train_size=0.70, test_size=0.30, random_state=5)
target = "class"
X_test = test.drop(target, axis=1)
y_test = test[target]
# Generate Predictions
predictions = []
for _, obs in X_test.iterrows():
    pred = predict_classification(train, list(obs), 3)
    predictions.append(pred)
print(f"ABW KNearestNeighbors (Functional-Approach) Accuracy: {accuracy_metric(y_test, predictions):.2f}")
```

```py
>>> ABW KNearestNeighbors (Functional-Approach) Accuracy: 95.56
```

In using the `predict_classification` function to generate predictions, an accuracy of about 95% was achieved. The result makes sense on the iris data, since the class labels are evenly distributed: 50 observations for `versicolor`, 50 for `virginica`, and 50 for `setosa`. In most practical applications however, the data is far less neat-and-tidy, and achieving high accuracy with the nearest neighbors algorithm could involve tuning how distance is calculated, modifying weights to make certain neighbors more impactful than others, and pre-processing the data with normalization, standardization, or vectorization techniques. 


---

### Object-Oriented Approach

With this functional approach working properly, the next step was to turn this nearest neighbor algorithm into an object-oriented implementation using a python class. 
The main difference between the two is that the OOP (object-oriented programming) version will house all of the previous functions as methods inside the class. This brings about some minor syntax and logic changes, but for the most part the code is the same. Another key difference is that the OOP version includes `.fit()` and `.predict()` methods to mirror the familiar usage provided by scikit-learn models. Can view code for the `KNearestNeighbor` class here.

We can test the OOP implementation on the same data and with a similar process as the functional implementation, but this time use a `KNearestNeighbor` class instance.

```py
nn = KNearestNeighbor(n_neighbors=3)
predictions = []

for _, obs in X_test.iterrows():
    pred = nn.predict(train, list(obs))
    predictions.append(pred)
print(f"ABW KNearestNeighbors (OOP) Accuracy: {nn.score(y_test, predictions):.2f}")
```
```py
>>> ABW KNearestNeighbors (OOP) Accuracy: 95.56
```
As expected, accuracy was identical to that of the functional implementation. 

---

### Compare with `sklearn.neighbors.KNeighborsClassifier` 

For kicks, we can benchmark these from-scratch implementations to the performance of scikit-learn's `KNeighborsClassifier`. 

```py
# Split train into x and y for use with sklearn model

X_train = train.drop(target, axis=1)
y_train = train[target]

# Import KNeighborsClassifier for comparison to my own
from sklearn.neighbors import KNeighborsClassifier

# create instance object
sk_nn = KNeighborsClassifier(n_neighbors=3) #> to match

# fit model
sk_nn.fit(X_train, y_train)

# Generate predictions
sk_preds = sk_nn.predict(X_test)

print(f"Sklearn KNeighborsClassifier Accuracy: {sk_nn.score(X_test, y_test):.2f}")
```
```py
>>> Sklearn KNeighborsClassifier Accuracy: 0.96
```
The scikit-learn `.score` method rounds differently than the `.score` method in my OOP implementation, but we can still see that the performance is practically the same. Again, this is largely thanks to the rare class-balance provided by the iris data. When tackling classification problems on messier data, the additional parameters available in the scikit-learn version become much more valuable.

---

This post walked through functional and object-oriented implementations of the nearest neighbors algorithm for classification, but much of the logic works the same way in a regression problem. And while nearest neighbor models are relatively simple and don't truly "train" the way other models would, they can still provide powerful results in applications like recommendation engines and label classification.

---

*Resources:*
- *UCI (https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)*
- *Scikit-Learn (https://scikit-learn.org/stable/modules/neighbors.html#classification)*
- *Machine Learning Mastery (https://scikit-learn.org/stable/modules/neighbors.html#classification)*