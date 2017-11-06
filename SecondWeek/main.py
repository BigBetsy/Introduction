import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
import operator


def k_neighbors_task():
    df = pd.read_csv('Data/wine_data.csv', header=None)
    x = df.iloc[:, 1:].values
    y = df.iloc[:, :1].values
    generator = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for i in range(1, 51):
        cls = KNeighborsClassifier(n_neighbors=i)
        score = cross_val_score(estimator=cls, X=x, y=y.ravel(), cv=generator, scoring='accuracy')
        results[i] = score.mean()
        sorted_x = sorted(results.items(), key=operator.itemgetter(1))
        sorted_x.reverse()
    print sorted_x

    for i in range(1, 51):
        cls = KNeighborsClassifier(n_neighbors=i)
        score = cross_val_score(estimator=cls, X=scale(x), y=y.ravel(), cv=generator, scoring='accuracy')
        results[i] = score.mean()
        sorted_x = sorted(results.items(), key=operator.itemgetter(1))
        sorted_x.reverse()
    print sorted_x

if __name__ == '__main__':
    k_neighbors_task()
