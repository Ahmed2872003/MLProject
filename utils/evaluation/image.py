from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss


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



def show_loss_curve(X_train, Y_train, model = "Logistic"):
    if model == "Logistic":
        show_logistic_Reg_loss_curve(X_train, Y_train)
    elif model == "KNN":
        show_KNN_loss_curve(X_train, Y_train)


def show_logistic_Reg_loss_curve(X_train, Y_train):
    losses = []
    max_iter = 100

    logreg = LogisticRegression(solver='lbfgs', warm_start=True, max_iter=500, random_state=42)

    for i in range(max_iter):
        logreg.fit(X_train, Y_train)
        y_prob = logreg.predict_proba(X_train)
        loss = log_loss(Y_train, y_prob)
        losses.append(loss)

    plt.plot(range(1, max_iter + 1), losses)
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.title('Logistic Regression Loss Curve')
    plt.show()


def show_KNN_loss_curve(X_train, Y_train, max_k=20):
    train_errors = []
        
    for k in range(1, max_k + 1):
        # Train KNN with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, Y_train)
        
        # Predict on the training set
        Y_train_pred = knn.predict(X_train)
        
        # Calculate training error rate
        train_error = 1 - accuracy_score(Y_train, Y_train_pred)
        train_errors.append(train_error)
    
    # Plot the training error curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), train_errors, label='Training Error', marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Error Rate')
    plt.title('KNN Training Loss Curve')
    plt.legend()
    plt.grid()
    plt.show()