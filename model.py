
from sklearn.metrics import accuracy_score, precision_score,\
    f1_score, fbeta_score, recall_score, precision_recall_curve,\
    median_absolute_error, confusion_matrix, roc_curve
import random, os
from itertools import product
from math import sqrt
import numpy as np
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from csv import DictWriter

from data import DataHandle, categories
from utils import EncodingMaps, is_notanitem, pickle_classifier, unpickle_classifier
from run import BASEDIR

_format_score = lambda x: round(float(x) * 100, 4)


class Preprocessor(DataHandle):

    def __init__(self,  testing_file, **kwargs):
        super().__init__()
        self.text_encoder = kwargs.get("xencoder", "huffman")

        self.encoderMaps = EncodingMaps(type=self.text_encoder)

        self._target_col = None
        self.pct_train = kwargs.get("perc_train", .7)
        self.pct_valid = kwargs.get("perc_validation", 0)
        self.selection_method = kwargs.get('selector', "random")
        assert (self.selection_method in ["random", "inline", "all"]), "invalid selector method must be random (DEFAULT), inline, or all"

        self.training_ids = self.set_trainingids()
        self.testing_ids = self.set_testids(datafile=testing_file)

        self.testing_file = testing_file

    @property
    def text_encoder(self):
        return self._text_encoder

    @text_encoder.setter
    def text_encoder(self, x):
        self._text_encoder = x

    @property
    def target_col(self):
        return self._target_col

    @target_col.setter
    def target_col(self, col):
        self._target_col = col

    def set_trainingids(self):
        selected = random.choices(population=[i["id"] for i in self.labeled_items], k=int(self.pct_train*len(self.labeled_items)))
        return [int(s) for s in selected]

    def set_testids(self, datafile):
        IDs = []
        if datafile.find("labeled") > -1:
            for x in self.labeled_items:
                if int(x["id"]) not in self.training_ids:
                    IDs.append(int(x["id"]))
                else:
                    pass
        else:
            IDs.extend([int(x["id"]) for x in self.unlabeled_items])
        return IDs

    def exclude_rare_categories(self, orig_items, min_freq):
        included = list(filter(lambda x: float(x["pct"]) >= min_freq, [_ for _ in categories]))
        return [x for x in orig_items if x["category"] in [i['name'] for i in included]]

    def preassign_noneitem(self, item):
        if is_notanitem(item) is True:
            return item

    def remove_notitem_cases(self, x, y):
        rmids = list(filter(lambda i: y[i]==26, range(0, len(y))))
        trainx = np.delete(arr=np.array(x), obj=np.array(rmids), axis=0)
        trainy = np.delete(arr=np.array(y), obj=np.array(rmids))
        return list(np.vstack(tuple([_ for _ in trainx]))), list([_ for _ in trainy])

    def training_items(self, min_category_freq, **kwargs):
        IDs = kwargs.get("ids", self.training_ids)
        items = [i for i in self.labeled_items if int(i["id"]) in IDs]
        return self.exclude_rare_categories(orig_items=items, min_freq=min_category_freq)

    def testing_items(self, min_category_freq, **kwargs):
        IDs = kwargs.get("ids", self.testing_ids)
        if self.testing_file.find('labeled') > -1:
            items = self.exclude_rare_categories(orig_items=[x for x in self.labeled_items if int(x["id"]) in IDs], min_freq=min_category_freq)
        else:
            items = [x for x in self.unlabeled_items if int(x["id"]) in IDs]
        return items

    def _process(self, xdata, ydata, *args, **kwargs):
        blocks = list(self.encoderMaps.encodedTextGenerator(string_lst=xdata))
        rowlens = list(map(lambda x: len(x), blocks))
        xtrarows = 40 - max(rowlens)
        diffs = [int(max(rowlens) + xtrarows) - x for x in rowlens]
        addtl = [self.encoderMaps.features["fillvalue"]]
        encx = list(map(lambda x: list([int(i) for i in blocks[x]]), range(len(blocks))))
        for x in range(len(encx)):
            encx[x].extend(addtl*diffs[x])
        x = np.vstack(tuple(encx))

        return x, ydata, blocks

    def training_set(self, items, target=None):
        if target is not None:
            col = target
        else:
            col = self.target_col

        trainx = [x["product_txt"] for x in items]
        trainy = [x["%s" % col] for x in items]
        return self._process(trainx, trainy)

    def testing_set(self, items, target=None):
        if target is not None:
            col = target
        else:
            col = self.target_col
        testx = [x["product_text"] for x in items if int(x["id"]) in [i for i in self.testing_ids]]
        actual = [x["%s" % col] for x in items if int(x["id"]) in [i for i in self.testing_ids]]
        return self._process(testx, actual)


class Performance:

    def __init__(self, predicted, actual, **kwargs):
        self.predicted = predicted
        self.actual = actual

        self.avg_method = kwargs.get("avg_method", "weighted")
        assert (self.avg_method in ["weighted", "macro", "micro", "binary"]), "invalid averaging method. see sklearn docs"

        self.labels = kwargs.get('labels', None)
        self.probs = kwargs.get("probs")

        self.ctab = self.ContingencyTable(rtn_object="dict")
        self.TP, self.TN, self.FP, self.FN = self.ctab["TruePos"], self.ctab["TrueNeg"], self.ctab["FalsePos"], \
                                             self.ctab["FalseNeg"]
        self._metrics = {"Accuracy": self.accuracy, "Precision": self.precision,
                         "Recall": self.recall, "F1": self.f1, "F1Beta": self.f1Beta,
                         "ContingencyRatios": self.ContingencyRatios}
        self.results = {}

    @property
    def PLR(self):
        """
        Positive Likelihood Ratio
        :return: 
        """
        try:
            x = float(self.recall/100)/self.TNR
        except ZeroDivisionError:
            x = 0
        return x

    @property
    def NLR(self):
        """
        Negative Likelihood Ratio
        :return: 
        """
        try:
            x = self.FNR/self.TNR
        except ZeroDivisionError:
            x = 0
        return x

    @property
    def DiagRatio(self):
        try:
            x = self.PLR / self.NLR
        except ZeroDivisionError:
            x = 0
        return x

    @property
    def NPV(self):
        """
        Negative Predictive Value
        :return: 
        """
        try:
            x = float(self.TN/(self.TN+self.FN))
        except ZeroDivisionError:
            x = 0
        return x

    @property
    def PPV(self):
        """
        Positive Predictive Value
        :return: 
        """
        try:
            x = float(self.TP/(self.TP+self.FP))
        except ZeroDivisionError:
            x = 0
        return x

    @property
    def FOR(self):
        """
        False Omission Rate
        :return: 
        """
        try:
            x = float(self.FN/(self.FN+self.TN))
        except ZeroDivisionError:
            x = 0
        return x

    @property
    def TNR(self):
        """
        True Negative Rate / Specificity
        :return: 
        """
        try:
            x = float(self.TN / (self.TN + self.FP))
        except ZeroDivisionError:
            x = 0
        return x

    @property
    def FPR(self):
        """
        False Positive Rate / Fall-Out
        :return: 
        """
        try:
            x = float(self.FP/(self.FP+self.TN))
        except ZeroDivisionError:
            x = 0
        return x

    @property
    def FDR(self):
        """
        False Discovery Rate (FDR)
        :return: 
        """
        try:
            x = float(self.FP/(self.FP+self.TP))
        except ZeroDivisionError:
            x = 0
        return x

    @property
    def FNR(self):
        """
        False Negative Rate / Miss Rate
        :return: 
        """
        try:
            x = float(self.FN/(self.FN+self.TP))
        except ZeroDivisionError:
            x = 0
        return x

    @property
    def accuracy(self):
        score = accuracy_score(self.actual, self.predicted)
        return _format_score(score)

    @property
    def precision(self):
        score = precision_score(y_true=self.actual, y_pred=self.predicted,
                                labels=self.labels, average=self.avg_method)
        return _format_score(score)

    @property
    def recall(self):
        """
        Recall / Hit Rate / Sensitivity / True Positive Rate
        :return: 
        """
        score = recall_score(y_true=self.actual, y_pred=self.predicted,
                             labels=self.labels, average=self.avg_method)
        return _format_score(score)

    @property
    def f1(self):
        score = f1_score(y_true=self.actual, y_pred=self.predicted,
                         labels=self.labels, average=self.avg_method)
        return _format_score(score)

    @property
    def ContingencyRatios(self):
        return {
            "TPR": self.recall, "TNR": self.TNR, "PPV": self.precision,
            "NPV": self.NPV, "FNR": self.FNR, "FPR": self.FPR,
            "FDR": self.FDR, "FOR": self.FOR, "PLR": self.PLR,
            "NLR": self.NLR, "DOR": self.DiagRatio,"MCC": self.matthews_corr_coef,
            "BM": self.informedness, "MK": self.markedness
        }

    @property
    def precision_recall_curve(self):
        return precision_recall_curve(y_true=self.actual, probas_pred=self.probs)

    @property
    def f1Beta(self):
        score = fbeta_score(y_true=self.actual, y_pred=self.predicted, labels=self.labels,
                            average=self.avg_method, beta=.5)
        return _format_score(score)

    @property
    def mad(self):
        return median_absolute_error(y_true=self.actual, y_pred=self.predicted)

    def roc_curve(self, positive_class):
        return roc_curve(y_true=self.actual, y_score=self.probs, pos_label=positive_class)

    @property
    def informedness(self):
        return float(float(self.recall/100) + self.TNR) - 1

    @property
    def markedness(self):
        return float(self.PPV + self.NPV) - 1

    @property
    def matthews_corr_coef(self):
        numer = (self.TP*self.TN) - (self.FP*self.FN)
        denom = (self.TP+self.FP)*(self.TP+self.FN)*(self.TN+self.FP)*(self.TN+self.FN)
        if denom == 0:
            x = 0
        else:
            x = numer/sqrt(denom)
        return x

    def _plot_cnf_mat(self, cat, show=False, normalize=False, **kwargs):
        import matplotlib.pyplot as plt

        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        cm = self.ContingencyTable(rtn_object="array")

        cmap = plt.cm.get_cmap("Blues")

        title = "Contingency Table -- %s" % cat
        classes = [i for i in range(len(cm[0]))]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize is True:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        ff = kwargs.get("savefile", "/".join([BASEDIR, "plots"]))
        if os.path.isdir(ff):
            plt.savefig("/".join([ff, title+".png"]), format="png")
        else:
            raise OSError("Invalid File Path %s" % ff)
        if show is True:
            plt.show()
        else:
            pass

    def ContingencyTable(self, rtn_object, **kwargs):
        assert (rtn_object in ["dict", "array", "plot"]), ValueError("Invalid Return Object. Must be 'dict', 'array' or 'plot'")
        cmat = confusion_matrix(y_true=self.actual, y_pred=self.predicted, labels=self.labels)

        if rtn_object == "dict":
            try:
                tab = {"TrueNeg": cmat[0][0], "FalseNeg": cmat[1][0],
                       "TruePos": cmat[1][1], "FalsePos": cmat[0][1]}
                return tab
            except IndexError:
                tab = {"TrueNeg": cmat[0][0], "FalseNeg":0,
                       "TruePos": 0, "FalsePos": 0}
                return tab
        elif rtn_object == "array":
            return cmat
        elif rtn_object == "plot":
            self._plot_cnf_mat(show=kwargs.get("show", False), cat=kwargs.get("cat"))

    def fetchMetrics(self, names):
        if hasattr(names, "__iter__"):
            for name in names:
                assert (name in [k for k in self._metrics.keys()]), "invalid metric %s" % name
                self.results.update({name: self._metrics[name]})

        elif type(names) is str:
            if names == "all":
                self.fetchMetrics(names=[k for k in self._metrics.keys()])
            else:
                assert (names in [k for k in self._metrics.keys()]), "invalid metric %s" % names
                self.results.update({names: self._metrics[names]})

    def dumpReport(self, reportfile, modelname=None):
        pass


class Test:

    def __init__(self, testing_file, target_col="category_encoded", pct_train=.65):
        self.testing_file = testing_file
        self.prep = Preprocessor(testing_file=testing_file, perc_train=pct_train)
        self.cats = sorted([(c["name"], float(c["pct"]), int(c["id"])) for c in categories], key=lambda x: x[1], reverse=True)
        self.trainids = [int(i) for i in self.prep.training_ids]
        self.testids = [int(i) for i in self.prep.testing_ids]
        self.prep.target_col = target_col

    def RandomSelection(self, target=None, show_perform=True):
        print("Running Random Selection")
        testx, testy, htest = self.prep.testing_set(items=self.prep.testing_items(min_category_freq=0.))
        selected = [n for n in np.random.choice(a=[c[2] for c in self.cats], size=len(testy))]
        print("Random Selection Model Performance")
        perf = Performance(predicted=selected, actual=testy)
        perf.fetchMetrics(["Accuracy", "Precision", "Recall", "F1"])
        if show_perform is True:
            print(perf.results)

    def _get_performance(self, predicted, actual):
        perf = Performance(predicted=[int(p) for p in predicted], actual=[int(_) for _ in actual])
        perf.fetchMetrics(["Accuracy", "Precision", "Recall", "F1"])
        ctab = perf.ContingencyTable(rtn_object="dict")
        perf.results.update(ctab)
        return perf.results

    def _truncate_datasets(self, ids, min_freq=0.):
        training_items = self.prep.training_items(min_category_freq=min_freq, ids=ids)
        testing_items = self.prep.testing_items(min_category_freq=min_freq, ids=ids)
        return training_items, testing_items

    def setup_binary_classifier(self, estimator, category_id, category_name, show_performs, ids):
        trainx, trainy, htrain = self.prep.training_set(items=self.prep.training_items(ids=self.trainids, min_category_freq=0.),
                                                        target="_".join([str(category_id), "bin"]))
        testx, testy, htest = self.prep.testing_set(items=self.prep.testing_items(ids=ids, min_category_freq=0.),
                                                    target="_".join([str(category_id), "bin"]))
        print("Status -- Fitting {0} Model for Category: {1} to Binary Target Column".format(type(estimator).__name__, str(category_name).upper()))
        try:
            estimator.fit(trainx, [int(t) for t in trainy])
        except ValueError:
            altclf = RandomForestClassifier(n_estimators=35)
            print(
                "Encountered an issue due to an underpopulated class. Using a {0} as the model for this category.".format(
                    str(type(altclf).__name__)))
            print("Status -- Fitting {0} Model for Category: {1} to Binary Target Column".format(type(altclf).__name__, str(category_name).upper()))
            altclf.fit(X=trainx, y=[int(y) for y in trainy])
            est = altclf
        else:
            est = estimator
        p = est.predict(X=testx)
        perf = Performance(predicted=p, actual=testy)
        perf.fetchMetrics(["Accuracy", "Recall", "Precision", "F1"])
        ctab = perf.ContingencyTable(rtn_object="dict")
        perf.results.update(ctab)

        if show_performs is True:
            print(perf.results)
        return est, testx, testy, p, perf.results

    def _performs2csv(self, file, data):
        f = "/".join([BASEDIR, ".data", "performance", file])
        if os.path.isfile(f) is False:
            with open(f, mode='w') as fx:
                writr = DictWriter(fx, fieldnames=["Category","Accuracy", "Precision", "Recall", "F1", "TrueNeg",
                                                   "FalseNeg", "TruePos", "FalsePos"])
                writr.writeheader()
                writr.writerows(data)
            fx.close()

    def BinaryEnsemble(self, base_estimator, save=False, recursive=True, classify_by="category", *args, **kwargs):
        print("Running Custom Ensemble Method Classifier")
        binary_est_file = os.curdir + "%s_binary_models.pkl" % type(base_estimator).__name__
        stack_est_file = os.curdir + "stack_model.pkl"
        master_kvlog = {}
        run_ = 0
        perf_data = []

        if len(master_kvlog.keys()) == 0:
            [master_kvlog.update({ID: None}) for ID in self.testids]

        if os.path.isfile(binary_est_file):
            estimators_ = unpickle_classifier(file=binary_est_file)
            for e in estimators_:
                IDs = [i for i in self.testids if master_kvlog[i] is None]
                testx, testy, htest = self.prep.testing_set(items=self.prep.testing_items(ids=IDs, min_category_freq=0.),
                                                            target="_".join([str(e[2]), "bin"]))

                pred = e[2].predict(testx)
                for p in range(len(pred)):
                    if int(pred[p]) != -1:
                        master_kvlog.update({self.testids[p]: pred[p]})
                perf = Performance(predicted=pred, actual=testy)
                perf.fetchMetrics(names=["Accuracy", "Recall", "Precision", "F1"])
                ctab = perf.ContingencyTable(rtn_object="dict")
                perf.results.update(ctab)
                perf.results.update({"Category": e[1].upper()})
                show_perf = kwargs.get("show_binary_performs", True)
                if show_perf is True:
                    print(perf.results)
                perf_data.append(perf.results)
        else:
            estimators_collection = []
            if classify_by.find("category") > -1:
                for i in range(len(self.cats)):
                    IDs = [i for i in self.testids if master_kvlog[i] is None]
                    est, testx, testy, pred, perform = self.setup_binary_classifier(estimator=base_estimator, category_id=self.cats[i][2],
                                                                                    category_name=str(self.cats[i][0]),
                                                                                    show_performs=kwargs.get("show_binary_performs", True),
                                                                                    ids=IDs)
                    estimators_collection.append((run_, self.cats[i][0], self.cats[i][2], est))

                    for p in range(len(pred)):
                        if int(pred[p]) != -1:
                            master_kvlog.update({self.testids[p]: pred[p]})
                    perform.update({"Category": self.cats[i][0].upper()})
                    perf_data.append(perform)
            estimators_ = [e[3] for e in estimators_collection]
            if save is True:
                pickle_classifier(clf=estimators_, file=binary_est_file)

        self._performs2csv(file="%s_binary_models.csv" % type(base_estimator).__name__, data=perf_data)

    def PreBuiltModel(self, estimator):
        print("Running %s Scikit-Learn Classifier" % type(estimator).__name__)
        trainx, trainy, htrain = self.prep.training_set(items=self.prep.training_items(ids=self.trainids, min_category_freq=0.))
        testx, testy, htest = self.prep.testing_set(items=self.prep.testing_items(ids=self.testids, min_category_freq=0.))
        estimator.fit(X=trainx, y=trainy)
        pred = estimator.predict(testx)
        probs = estimator.predict_proba(testx)

        print("%s Model Performance" % type(estimator).__name__)
        perf = Performance(predicted=pred, actual=testy, probs=probs)
        perf.fetchMetrics(names=["Accuracy", "Recall", "Precision", "F1"])
        res = perf.results
        
        return res
