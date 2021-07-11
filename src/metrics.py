import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


def optimal_cutoff(estimator, X, y, ax, name):
    y_probs = pd.Series(estimator.predict_proba(X)[:, 1], index=y.index)

    y_pred = pd.DataFrame()
    y_pred["true"] = y
    y_pred["probs"] = y_probs
    cutoffs = np.linspace(0, 0.99, 20)
    for i in cutoffs:
        y_pred[i] = np.where(y_probs > i, 1, 0)
    cutoff_columns = ["prob", "accuracy", "sensitivity", "specificity"]
    cutoff_df = pd.DataFrame(columns=cutoff_columns)
    total = X.shape[0]
    for i in cutoffs:
        cm = metrics.confusion_matrix(y_pred["true"], y_pred[i])
        (tn, fp), (fn, tp) = cm
        accuracy = (tp+fp)/total
        specificity = tn/(tn+fp+1e-4)
        sensitivity = tp/(tp+fp+1e-4)
        cutoff_df.loc[i] = [i, accuracy, sensitivity, specificity]
    cutoff_df.plot.line(
        x="prob", y=["accuracy", "sensitivity", "specificity"], ax=ax)


def show_performance(estimator, X_train, y_train, X_test, y_test):
    # Predict
    y_train_pred = estimator.predict(X_train)
    y_train_probs = estimator.predict_proba(X_train)

    # Predict probabilities
    y_test_pred = estimator.predict(X_test)
    y_test_probs = estimator.predict_proba(X_test)

    # Display scores
    train_score = estimator.score(X_train, y_train)
    test_score = estimator.score(X_test, y_test)
    print("\nTrain Score: {}\nTest Score: {}".format(train_score, test_score))

    # Display scores
    train_auc_score = metrics.roc_auc_score(y_train, y_train_probs[:, 1])
    test_auc_score = metrics.roc_auc_score(y_test, y_test_probs[:, 1])
    print("\nTrain AUC Score: {}\nTest AUC Score: {}".format(train_auc_score,
                                                             test_auc_score))

    # Display scores
    train_recall_score = metrics.recall_score(y_train, y_train_pred)
    test_recall_score = metrics.recall_score(y_test, y_test_pred)
    print("\nTrain Recall Score: {}\nTest Recall Score: {}".format(
        train_recall_score, test_recall_score))

    # Show classification report
    print("\nTrain classification Report")
    print(metrics.classification_report(y_train, y_train_pred))

    print("\nTest classification Report")
    print(metrics.classification_report(y_test, y_test_pred))

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    # Show ROC
    _ = metrics.plot_roc_curve(estimator, X_train, y_train, ax=axes[0],
                               name="Training")
    _ = metrics.plot_roc_curve(estimator, X_test, y_test, ax=axes[0],
                               name="Testing")
    axes[0].set(title='Receiver operating characteristic')
    # Show Precision-Recall
    _ = metrics.plot_precision_recall_curve(estimator, X_train, y_train,
                                            ax=axes[1], name="Training")
    _ = metrics.plot_precision_recall_curve(estimator, X_test, y_test,
                                            ax=axes[1], name="Testing")
    axes[1].set(title='Precision-Recall')
    # Show Detection Error Tradeoff
    _ = metrics.plot_det_curve(estimator, X_train, y_train, ax=axes[2],
                               name="Training")
    _ = metrics.plot_det_curve(estimator, X_test, y_test, ax=axes[2],
                               name="Testing")
    axes[2].set(title='Detection Error TradeOff')
    # Optimal cutoff
    optimal_cutoff(estimator, X_test, y_test, ax=axes[3], name="Testing")
    axes[3].set(title='Optimal Cutoff')

    for ax in axes:
        _ = ax.grid(linestyle='--')
    # plt.tight_layout()
    plt.show()
