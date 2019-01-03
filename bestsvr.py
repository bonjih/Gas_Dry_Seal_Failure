from sklearn import svm
from sklearn.metrics import mean_squared_error
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures


class BestSVRClassifier:
    classifier_types = ['linear', 'rbf', 'sigmoid', 'poly']

    def __init__(self, X, X_train, y, y_train, X_test, y_test):
        """"
        :param X
        :param X_train
        :param y
        :param y_train
        :param X_test
        :param y_test
        """
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_best_classifier(self):

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(BestSVRClassifier.__async_classify, self, klassif): klassif for klassif in self.classifier_types}
            print(futures)

            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            print(results)
            return self.__best_classifier(results)

    def __async_classify(self,classifier):
        clf = svm.SVR(kernel=classifier)
        clf.fit(self.X_train, self.y_train)
        confidence = clf.score(self.X_test, self.y_test)
        mse = mean_squared_error(self.y_test, clf.predict(self.X_test))
        return {"type": classifier, "MSE": mse, "confidence": confidence, "classifier": clf}

    def __best_classifier(self, results):
        best = 0
        for i in range(1, len(results)):
            if results[i]['MSE'] <= results[best]['MSE'] and results[i]['confidence'] >= results[best]['confidence']:
                best = i

        return results[best]
