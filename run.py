import os

BASEDIR = os.getcwd()
DATADIR = "/".join([BASEDIR, ".data"])


if __name__ == "__main__":
    import os
    import pip
    pkgfile = os.getcwd() + '/requirements.txt'
    try:
        packages = [i.replace("\n", "") for i in open(pkgfile, "r").readlines()]
    except:
        pass
    else:
        for pkg in packages:
            pip.main(["install", pkg])
        os.remove(pkgfile)

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from model import Test
    import warnings

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        model = Test(testing_file="labeled")

        model.RandomSelection()
        model.PreBuiltModel(estimator=RandomForestClassifier(n_estimators=35))
        model.PreBuiltModel(estimator=SVC(C=10.0, gamma=.001, probability=True, decision_function_shape="ovr"))
        model.BinaryEnsemble(base_estimator=DecisionTreeClassifier())
        model.BinaryEnsemble(base_estimator=SVC(C=10.0, gamma=.001, probability=True, decision_function_shape="ovr"))
