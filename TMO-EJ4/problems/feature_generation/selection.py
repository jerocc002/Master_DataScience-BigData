import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import ExtraTreeClassifier
from problems.feature_generation.problem import EvolEquation
pd.options.mode.chained_assignment = None

def inv_accuracy_score(y, pred):
    return 1 - accuracy_score(y, pred)

if __name__ == '__main__':
    df = pd.read_csv('data/australian/australian_expanded.csv')
    X, y = df[[c for c in df.columns if c != 'y']], df.y

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train)

    rfc = RandomForestClassifier(max_depth=5, random_state=2017, n_estimators=100)
    rfc.fit(X_train, y_train)
    current_val_score = accuracy_score(y_val, rfc.predict(X_val))
    print("Initial validation score: %f" % current_val_score)
    print("Initial test score: %f" % accuracy_score(y_test, rfc.predict(X_test)))

    for i in range(1000):
        ee = EvolEquation(X_train, y_train, max_depth=4,
                          num_equations=100, mutation_prob=0.2,
                          metric=inv_accuracy_score)
        best, best_fitness, ev = ee.run(50)

        X_train['__NEW_FEATURE__'] = ee._create_feature(best, X_train, False)
        X_test['__NEW_FEATURE__'] = ee._create_feature(best, X_test, False)
        X_val['__NEW_FEATURE__'] = ee._create_feature(best, X_val, False)
        X_train['__NEW_FEATURE__'].fillna(-9999, inplace=True)
        X_test['__NEW_FEATURE__'].fillna(-9999, inplace=True)
        X_val['__NEW_FEATURE__'].fillna(-9999, inplace=True)
        rfc = RandomForestClassifier(max_depth=5, random_state=2017, n_estimators=100)
        rfc.fit(X_train, y_train)

        val_score = accuracy_score(y_val, rfc.predict(X_val))
        print("\tValidation score: %f" % val_score)
        print("\tTest score: %f" % accuracy_score(y_test, rfc.predict(X_test)))
        print("\tNew feature: %s" % best)
        print("")
    
        print("\tAdded new feature")
        n_features = X_train.shape[1]
        X_train['V%d' % n_features] = X_train['__NEW_FEATURE__']
        X_test['V%d' % n_features] = X_test['__NEW_FEATURE__']
        X_val['V%d' % n_features] = X_val['__NEW_FEATURE__']

        current_val_score = val_score

        X_train['y'] = y_train
        X_test['y'] = y_test
        X_val['y'] = y_val

        del X_train['__NEW_FEATURE__']
        del X_test['__NEW_FEATURE__']
        del X_val['__NEW_FEATURE__']


        df = pd.concat((X_train, X_test, X_val), axis=0)
        df.to_csv("data/australian/australian_expanded2.csv", index=False)

        del X_train['y']
        del X_test['y']
        del X_val['y']









