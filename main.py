import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def evaluation(model, x_train_std, y_train, x_test, y_test, train=True):
    if train:
        pred = model.predict(x_train_std)
        classifier_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"F1 Score: {round(f1_score(y_train, pred), 2)}")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{classifier_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")

    if not train:
        pred = model.predict(x_test)
        classifier_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"F1 Score: {round(f1_score(y_test, pred), 2)}")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{classifier_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


def data_analyse():
    df = pd.read_csv('diabetes.csv')
    diabetic = df[df['Outcome'] == 1].count().loc['Outcome']
    non_diabetic = df[df['Outcome'] == 0].count().loc['Outcome']
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    plt.suptitle('Analyse of Data')
    # Соотношение больных диабетом и небольных
    ax1 = ax[0, 0]
    labels = ['Diabetic',
              'Non-Diabetic']
    ax1.set_title('Outcomes', fontdict={'weight': 'bold'})
    percentages = [diabetic, non_diabetic]
    explode = (0.1, 0)
    ax1.pie(percentages, explode=explode, labels=labels, autopct='%1.0f%%',
            shadow=False, startangle=0,
            pctdistance=1.2, labeldistance=1.4)
    ax1.legend(frameon=False, bbox_to_anchor=(1.5, 0.8))
    # Большинство небольны диабетом

    # Хитмап взаимоотношения переменных
    ax2 = ax[0, 1]
    corr = df.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns,
                ax=ax2)
    ax2.set_title("Dependecies of values", fontdict={'weight': 'bold'})
    # Диабетом чаще всего болеют при высоком уровне глюкозы

    # Предугадывания
    x = df['Glucose']
    y = df['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25)

    # Преобразование данных в 2Д
    x_train = x_train.values.reshape(-1, 1)
    x_test = x_test.values.reshape(-1, 1)

    # Предугадывание
    log = LogisticRegression()
    log.fit(x_train, y_train)
    y_pred = log.predict(x_test)

    ax4 = ax[1, 1]
    ax4.set_title('Accuracy of prediction', fontdict={'weight': 'bold'})
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')

    # pred_proba() - вероятность угадывания
    y_pred_proba = log.predict_proba(x_test)[::, 1]
    # fpr - ошибочные предикты относильтельно всех
    # tpr - правильные предикты относительно всех
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    # Качество классификации, т.е. предугадываний
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    ax4.plot(fpr, tpr, label="auc=" + str(auc))
    ax4.legend(loc=4)
    # График зависимости уровня глюкозы и конечного результата
    ax3 = ax[1, 0]
    ax3.grid()
    accuracy_scores = []
    knn = ''
    for i in range(1, 10):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        accuracy_scores.append(accuracy_score(y_test, knn.predict(x_test)))

    ax3.set_title('Accuracy of prediction', fontdict={'weight': 'bold'})
    ax3.plot(accuracy_scores)
    plt.show()
    print('KNN CLASSIFICATION')
    evaluation(knn, x_train, y_train, x_test, y_test, True)
    print('LOGISTIC REGRESSION')
    evaluation(log, x_train, y_train, x_test, y_test, True)


def main():
    data_analyse()


if __name__ == '__main__':
    main()
