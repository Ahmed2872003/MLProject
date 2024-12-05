from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import matplotlib.pyplot as plt


def evaluate(model, Y_test, Y_pred, X_test):

    eval_info = { "auc_curve_info": { "Y_test_bin": None, "Y_score": None } }

    eval_info["acc_sc"] = round(accuracy_score(Y_test, Y_pred), 2)
    
    eval_info["cm"] = confusion_matrix(Y_test, Y_pred)

    eval_info["precision"] = round(precision_score(Y_test, Y_pred, average="micro"), 2)
    
    eval_info["recall"] = round(recall_score(Y_test, Y_pred, average="micro"), 2)
    


    eval_info["auc_curve_info"]["Y_score"] = y_score = np.array(model.predict_proba(X_test))

    eval_info["auc_curve_info"]["Y_test_bin"] = y_test_bin = label_binarize(Y_test, classes=np.unique(np.array(Y_test))) 

    eval_info["AUC"] = round(roc_auc_score(y_test_bin, y_score, multi_class="ovr"), 2)

    return eval_info




def show_roc_graph(title, Y_test_bin, Y_score):
    for i in range(Y_score.shape[1]): 
        fpr, tpr, _ = roc_curve(Y_test_bin[:, i], Y_score[:, i])
        plt.plot(fpr, tpr, label=f"Class {i}")

    plt.plot([0, 1], [0, 1], 'k--', label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def show_logestic_Reg_loss_curve(X_train, Y_train):
    logreg_cv = LogisticRegressionCV(max_iter=1000, random_state=42, solver='liblinear')
    logreg_cv.fit(X_train, Y_train)

    plt.plot(logreg_cv.scores_[1].mean(axis=0))
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.title('Logistic Regression Loss Curve')
    plt.show()