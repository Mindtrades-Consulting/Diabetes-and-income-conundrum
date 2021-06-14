import matplotlib.pyplot as plt
import pandas as pd
import scipy
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from functools import reduce


def plot_correlation(df, output_factors):

    # Convert non-numeric columns to numeric factors and then calculate correlation
    df_num = df.apply(lambda x: x.factorize()[0])

    print(df_num)

    '''
    # How to plot the correlation of every column with every other column in the dataframe
    df_num = df_num.corr()
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df_num.corr(), fignum=f.number)
    plt.xticks(range(df_num.select_dtypes(['number']).shape[1]), df_num.select_dtypes(['number']).columns, fontsize=10, rotation=45)
    plt.yticks(range(df_num.select_dtypes(['number']).shape[1]), df_num.select_dtypes(['number']).columns, fontsize=10)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.title('Correlation Matrix', fontsize=16);

    plt.show()
    '''

    # All columns except 'Dalc' and 'Walc'
    list_1 = [col for col in df_num.select_dtypes(['number']).columns if col not in output_factors]
    list_2 = output_factors

    plotDict = {}
    # Loop across each of the two lists that contain the items you want to compare
    for gene1 in list_1:
        for gene2 in list_2:
            # Do a spearmanR comparison between the two items you want to compare
            tempDict = {(gene1, gene2): scipy.stats.spearmanr(df_num[gene1], df_num[gene2])}
            # Update the dictionary each time you do a comparison
            plotDict.update(tempDict)

    # Unstack the dictionary into a DataFrame
    dfOutput = pd.Series(plotDict).unstack()

    # Take just the spearmanR value out of the output tuple
    dfOutputPearson = dfOutput.apply(lambda x: x.apply(lambda x:x[0]))
    print(dfOutputPearson)

    # generate a heatmap
    sns.heatmap(dfOutputPearson, xticklabels=True, yticklabels=True)

    # Plot the correlation
    plt.show()


    '''
    # Another way to plot the correlation matrix based on list_1 and list_2
    # initiate empty dataframe
    corr = pd.DataFrame()
    for a in list_1:
        for b in list_2:
            corr.loc[a, b] = df_num.corr().loc[a, b]
    print(corr)
    sns.heatmap(corr, xticklabels=True, yticklabels=True)
    plt.show()
    '''


def filtered_data_condition(df, column_name, value):
    return df.loc[df[column_name] == value]


def build_model_and_predict(df, column_to_drop):
    # Evaludate algorithms, X-axis will contain all parameters except the alcoholic consumption parameters
    X = df.drop(columns = ['Walc', 'Dalc']).values

    # 26 is the index of alcoholic consumption parameter
    y = df.drop(column_to_drop, axis=1).values[:,26]

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    # Compare Algorithms
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.show()

    # We can fit the model on the entire training dataset and make predictions on the validation dataset.
    # Make predictions on validation dataset
    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    # We can evaluate the predictions by comparing them to the expected results in the validation set, then calculate classification accuracy, as well as a confusion matrix and a classification report.
    # We can see that the accuracy on the hold out dataset.
    # The confusion matrix provides an indication of the errors made.
    # Finally, the classification report provides a breakdown of each class by precision, recall, f1-score and support

    # Evaluate predictions
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions, zero_division=1))


if __name__ == '__main__':

    csv_file = 'student-mat.csv'
    df = pd.read_csv(csv_file)

    '''plot_correlation(df, ['Dalc', 'Walc'])

    df_res = filtered_data_condition(df, 'sex', 'M')
    plot_correlation(df_res, ['Dalc', 'Walc']) #school goout

    df_res = filtered_data_condition(df, 'sex', 'F')
    plot_correlation(df_res, ['Dalc', 'Walc']) #address schoolsup

    df_res = filtered_data_condition(df, 'famsize', 'LE3')
    plot_correlation(df_res, ['Dalc', 'Walc']) #absences Pstatus

    df_res = filtered_data_condition(df, 'famsize', 'GT3')
    plot_correlation(df_res, ['Dalc', 'Walc']) #sex/failures school

    df_res = filtered_data_condition(df, 'activities', 'yes')
    plot_correlation(df_res, ['Dalc', 'Walc']) #goout goout

    df_res = filtered_data_condition(df, 'activities', 'no')
    plot_correlation(df_res, ['Dalc', 'Walc']) #sex/failures sex/failures

    csv_file = 'student-por.csv'
    df = pd.read_csv(csv_file)

    plot_correlation(df, ['Dalc', 'Walc'])

    df_res = filtered_data_condition(df, 'sex', 'M')
    plot_correlation(df_res, ['Dalc', 'Walc']) #goout goout

    df_res = filtered_data_condition(df, 'sex', 'F')
    plot_correlation(df_res, ['Dalc', 'Walc']) #guardian absences'''

    # From this point onwards, all the code works for each of the CSVs, so execute it for each CSV one by one for getting the graphs
    # Note that the Dataframe below is referring to CSV opened earlier for reading
    df = df.apply(lambda x: x.factorize()[0])

    '''# Size of the dataset
    print(df.shape)

    # First 20 rows of the dataset
    print(df.head(20))

    # Now we can take a look at a summary of each attribute.
    # This includes the count, mean, the min and max values as well as some percentiles.
    print(df.describe())

    # Group the data by and attribute and find number of occurences of each value. e.g. in below case the attribute is 'Mjob'
    print(df.groupby('Mjob').size())

    # box and whisker plots
    df.plot(kind='box', subplots=True, layout=(11,3), sharex=False, sharey=False)
    plt.show()

    # histograms
    df.hist()

    # Launch the subplots tool to adjust spacing between subplots
    #plt.subplot_tool()
    plt.subplots_adjust(left=0.1,
                        bottom=0.2, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)
    plt.show()

    # scatter plot matrix
    axes = scatter_matrix(df, alpha=0.2)
    for ax in axes.flatten():
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')

    plt.tight_layout()
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    plt.show()

    # Build Model and Predict Daily Alcoholic Consumption
    build_model_and_predict(df, "Walc")

    # Build Model and Predict Weekend Alcoholic Consumption
    build_model_and_predict(df, "Dalc")'''

    df_res = df[['Dalc', 'Walc', 'G1', 'G2', 'G3']]
    plot_correlation(df_res, ['G1', 'G2', 'G3'])
