import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


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

    # График зависимости уровня глюкозы и конечного результата
    ax3 = ax[1, 0]
    ax3.grid()
    ax3.set_title('Average number of pregnancies in different ages', fontdict={'weight': 'bold'})
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Number of pregnancies')
    ax3.bar('A', df[df['Age'].between(20, 30)]['Pregnancies'].mean(), width=0.2)
    ax3.bar('B', df[df['Age'].between(30, 40)]['Pregnancies'].mean(), width=0.2)
    ax3.bar('C', df[df['Age'].between(40, 50)]['Pregnancies'].mean(), width=0.2)
    ax3.bar('D', df[df['Age'].between(50, 60)]['Pregnancies'].mean(), width=0.2)
    ax3.bar('E', df[df['Age'].between(60, 70)]['Pregnancies'].mean(), width=0.2)
    ax3.legend(['Between 20 and 30', 'Between 30 and 40',
                'Between 40 and 50', 'Between 50 and 60',
                'Between 60 and 70'])
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
    print(y_pred)

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
    plt.show()


def main():
    data_analyse()


if __name__ == '__main__':
    main()
