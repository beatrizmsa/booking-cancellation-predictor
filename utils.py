import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel

## Data Exploration and Preparation functions
def boxplot_by_type_visualization(data, columns, title=''):
    columns = [col for col in columns if col != 'is_canceled']
    n_cols = min(4, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.boxplot(x='is_canceled', y=col, data=data, ax=axes[i])
        axes[i].set_xlabel('is_canceled')
        axes[i].set_ylabel(col)

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.suptitle(title, fontsize=16)
    plt.show()

def crosstab_by_type_visualization(data, columns, title):
    n_cols = min(4, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10 * n_rows))

    axes = axes.flatten()
    for i, col in enumerate(columns):
        crosstab_result = pd.crosstab(data[col], data['is_canceled'])
        crosstab_result.plot(kind="bar", ax=axes[i])
        axes[i].set_xlabel('is_canceled')
        axes[i].set_ylabel(col)
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.suptitle(title, fontsize=16)
    plt.show()


def boxplot_visualization(data, columns, title):
    n_cols = min(5, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.boxplot(data=data, y=col, ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_ylabel("")

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.suptitle(title, fontsize=16)
    plt.show()


def barplot_visualization(data, columns, title):
    n_cols = min(5, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    axes = axes.flatten()
    for i, col in enumerate(columns):
        label = data.groupby(col).size()
        sns.barplot(x=label.index, y=label.values, ax=axes[i])
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.suptitle(title, fontsize=16)
    plt.show()


def best_neural_network(X_train, y_train):
    # Define the parameter grid for the Neural Network
    param_grid = {
        'activation': ['relu', 'tanh', 'logistic'],
        'alpha': [0.0001, 0.001, 0.01],
    }
    # Initialize the MLPClassifier (Neural Network model)
    nn = MLPClassifier(max_iter=200, random_state=42)

    # Set up GridSearchCV to find the best parameters
    grid_search = GridSearchCV(nn, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    print("Best parameters", grid_search.best_params_)
    print("Best Score", grid_search.best_score_)

    return grid_search.best_estimator_
def best_random_forest(X_train, y_train):
    rfm = RandomForestClassifier()
    param_rfm = {'n_estimators': [100, 200, 300, 400, 500],
                 'max_depth': [None, 10, 20, 30],
                 'max_features': ['sqrt', 'log2']}
    rfm_gs = GridSearchCV(rfm, param_rfm, cv=5, scoring='roc_auc')
    rfm_gs.fit(X_train, y_train)

    print("Best parameters", rfm_gs.best_params_)
    print("Best Score", rfm_gs.best_score_)
    return rfm_gs.best_estimator_
def best_k_for_KNN(X_train, y_train):
    param_grid = {'n_neighbors': np.arange(3,152,2)}

    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Best parameters", grid_search.best_params_)
    print("Best Score", grid_search.best_score_)
    return grid_search.best_estimator_

## Lasso
def lasso_regularization(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    scaler.fit(X_train)

    lasso = LogisticRegression(C=0.5, penalty='l1', solver='liblinear')
    sel_ = SelectFromModel(lasso)
    sel_.fit(scaler.transform(X_train), y_train)

    selected_feat = X_train.columns[sel_.get_support()]
    print("Number of removed features :", np.sum(sel_.estimator_.coef_ == 0))

    removed_feats = X_train.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]
    print("Removed features:", removed_feats)

    X_lasso = pd.DataFrame(sel_.transform(scaler.transform(X)), columns=selected_feat)
    df_lasso = pd.concat([X_lasso, y.reset_index(drop=True)], axis=1)

    return X_lasso, df_lasso

def get_feature_importances_text(column_names, importances):
    features = list(zip(column_names, importances))

    features_sorted = sorted(features, key=lambda x: x[1], reverse=True)
    print("Feature Importances:")
    for feature, importance in features_sorted:
        print(f"- {feature}: {importance:.3f}")

## Evaluation
def classification_report_print(y_test, y_pred):
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()

    # Calcular a acurácia
    accuracy = accuracy_score(y_test, y_pred)

    df = df.round(2)
    # Adicionar a linha de accuracy
    df.loc['accuracy'] = ["", "", round(accuracy, 2), len(y_test)]

    # Adicionar linha vazia
    df.loc[''] = ["", "", "", ""]  # Linha vazia

    # Reorganizar as linhas
    df = df.reindex(['0', '1', '', 'accuracy', 'macro avg', 'weighted avg'])

    return df


def plot_confusion_matrix(y_true, y_pred, suffix=''):
    title = 'Confusion Matrix '
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(2, 2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title(title + suffix)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()



def roc_curve_data(models_dict, X_test, y_test):
    roc_data = {}
    for model_name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        roc_data[model_name] = (fpr, tpr, auc_score)
    return roc_data


def roc_curve_visualization(roc_data):
    plt.figure(figsize=(10, 8))

    for model_name, (fpr, tpr, auc_score) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')

    # Random guessing line
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guessing')

    # Labels and legend
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Ensemble and Individual Models')
    plt.legend(loc='lower right')
    plt.show()