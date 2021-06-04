'------------Import Modules------------'

# DS Modules
import random
import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import seaborn as sns

# Statistical Tests
from scipy.stats import normaltest, ttest_ind, chi2_contingency, chi2

# Modelling
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


"---------------------- File Column Parameters -------------------------"

"""
Taken from ../data/census_income_metadata.txt
"""

# Data keys
keys = ["AAGE", "ACLSWKR", "ADTIND", "ADTOCC", "AHGA", "AHRSPAY", "AHSCOL", 
       "AMARITL", "AMJIND", "AMJOCC", "ARACE", "AREORGN", "ASEX", "AUNMEM", "AUNTYPE",
       "AWKSTAT", "CAPGAIN", "CAPLOSS", "DIVVAL",  "FILESTAT", "GRINREG", "GRINST",
       "HHDFMX", "HHDREL", "MARSUPWT", "MIGMTR1", "MIGMTR3", "MIGMTR4", "MIGSAME", "MIGSUN",
       "NOEMP", "PARENT", "PEFNTVTY", "PEMNTVTY", "PENATVTY", "PRCITSHP",
        "SEOTR", "VETQVA", "VETYN", "WKSWORK", "YEAR", "INCOME"]

# Data values
values = ["age", "class of worker", "industry code", "occupation code",  "education", "wage per hour", 
"enrolled in edu inst last wk", "marital status", "major industry code", "major occupation code", 
"race", "hispanic Origin", "sex", "member of a labor union", "reason for unemployment", "full or part time employment stat",
"capital gains", "capital losses", "dividends from stocks", 
"tax filer status", "region of previous residence", "state of previous residence", "detailed household and family stat",
"detailed household summary in household", "instance weight", "migration code-change in msa", 
"migration code-change in reg", "migration code-move within reg", "live in this house 1 year ago", 
"migration prev res in sunbelt", "num persons worked for employer", "family members under 18", 
 "country of birth father", "country of birth mother", "country of birth self",
"citizenship",  "own business or self employed", 
"fill inc questionnaire for veteran's admin", "veterans benefits", "weeks worked in year", "year",
         "income classification"]

# check to see if columns exist:
# for cols in df.columns:
#     print(cols, df[cols].unique().tolist(), "\n")
# "adjusted gross income",, "AGI"

# Create mapping dictionary
column_dict = dict()
for i in range(len(keys)):
    column_dict[keys[i]] = values[i]


"---------------------- Assesment functions -------------------------"

def univariateContinuous(dataframe, numerical_features):
    """
    Functions returns summary dataframe of numerical features and histogram plots
    Params:
        dataframe: type(DataFrame), pandas DataFrame
        numerical_features: type(List), list of column names for continuous features
    """
    n_bins=20

    # generate plots of histograms
    fig, axs = plt.subplots(2, 4, sharey=False, sharex=False, tight_layout=True, figsize=(20,8))
    axs = axs.ravel()
    # We can set the number of bins with the `bins` kwarg
    for idx,ax in enumerate(axs):
        ax.grid()
        ax.set_title("Histogram of {}".format(numerical_features[idx]))
        ax.hist(dataframe[numerical_features[idx]], bins=n_bins, color=["red"])

    # create feature summary and include some additional metrics
    summary_df = dataframe[numerical_features].describe()\
    .round()\
    .append(pd.DataFrame(columns= ["index"] + numerical_features,\
                        data=[["median"] + [round(dataframe[cols].median()) for cols in numerical_features],
                            ["skew"] + [round(dataframe[cols].skew(), 2) for cols in numerical_features],
                            ["kurtosis"] + [round(dataframe[cols].kurtosis(), 2) for cols in numerical_features],
                             ["normality: statistic"] + [round(normaltest(dataframe[cols]).statistic, 2) for cols in numerical_features],
                             ["normality: p-value"] + [round(normaltest(dataframe[cols]).pvalue, 2) for cols in numerical_features]])\
    .set_index("index"))
    return summary_df


def categoricalCountPlot(dataframe, categorical_feature):
    # generate count plots
    plt.figure(figsize=(20,5))
    ax = sns.countplot(y=categorical_feature, data=dataframe)
    ax.set_title("Count plot of {}".format(categorical_feature))
    return plt.show()


def univariateCategorical(dataframe, categorical_features):
    """
    Functions returns plots of feature categories
    Params:
        dataframe: type(DataFrame), pandas DataFrame
        categorical_features: type(List), list of column names for categorical features
    """
    for categorical_feature in categorical_features:
        # extract majority class over entire proportion
        print("{f}: \nNumber of distinct categories = {d} \nMajority class = '{a}' with {b}% of samples."\
              .format(f=categorical_feature,
                      a=dataframe[categorical_feature].mode()[0], d=dataframe[categorical_feature].nunique(),
                           b=round(100*len(dataframe.loc[dataframe[categorical_feature] == dataframe[categorical_feature].mode()[0]])/len(dataframe))))

        # reduce what we plot if feature has > 10 categories
        if dataframe[categorical_feature].nunique() > 10:
             dataframe = dataframe.loc[dataframe[categorical_feature].isin(\
                                                    dataframe[categorical_feature].value_counts()[:5].index.tolist()\
                                                       + dataframe[categorical_feature].value_counts()[-5:].index.tolist())]
        # plot the count plot
        categoricalCountPlot(dataframe, categorical_feature)
    
    return


def bivariateContinuous(dataframe, numerical_features, target):
    """
    Functions returns boxplot of numerical_feature vs target feature
    Params:
        dataframe: type(DataFrame), pandas DataFrame
        numerical_features: type(List), list of column names for continuous features
        target: y feature comparing to
    """
    # generate subplot figure 
    fig, axs = plt.subplots(2, 4, sharey=False, sharex=False, tight_layout=True, figsize=(20,14))
    axs = axs.ravel()
    # generate boxplots
    for idx,ax in enumerate(axs):
        ax.set_title("{y} \n vs. \n {x}".format(x=numerical_features[idx], y=target))
        sns.boxplot(ax=ax, y=dataframe[numerical_features[idx]], x=dataframe[target])

    return


def bivariateContinuous(dataframe, numerical_features, target, seed):
    """
    Functions returns boxplot of numerical_feature vs target feature, plus info on two classes (t-test)
    Params:
        dataframe: type(DataFrame), pandas DataFrame
        numerical_features: type(List), list of column names for continuous features
        target: y feature comparing to
    """
   
    # generate subplot figure 
    fig, axs = plt.subplots(2, 4, sharey=False, sharex=False, tight_layout=True, figsize=(20,14))
    axs = axs.ravel()
    # generate boxplots
    for idx,ax in enumerate(axs):
        
        # split the data between the two classes
        class0 = dataframe.loc[dataframe[target] == ' - 50000.'][numerical_features[idx]]
        class1 = dataframe.loc[dataframe[target] == ' 50000+.'][numerical_features[idx]]
        
        # plot boxplots with t-test results in title header
        ax.set_title("""{y} \n vs. \n {x} \n Mean values: <50k = {l}, >50K = {m} \n t-test result: statistic={s}, pvalue={p}"""\
                     .format(x=numerical_features[idx],
                             y=target,
                             l=round(class0.mean(), 2),
                             m=round(class1.mean(), 2),
                             s=round(ttest_ind(random.sample(class1.values.tolist(), 300),
                                               random.sample(class0.values.tolist(), 300),
                                               equal_var=False).statistic, 2),
                             p=round(ttest_ind(random.sample(class1.values.tolist(), 300),
                                               random.sample(class0.values.tolist(), 300),
                                               equal_var=False).pvalue, 3)))
        sns.boxplot(ax=ax, y=dataframe[numerical_features[idx]], x=dataframe[target])

    return


def categoricalComparisonPlot(dataframe, categorical_feature, target):
    # generate count plots
    plt.figure(figsize=(20,5))
    ax = sns.countplot(y=categorical_feature, hue=target, data=dataframe)
    ax.set_title("Comparison plot plot of {a} per {b}".format(a=categorical_feature, b=target))
    return plt.show()

def bivariateCategorical(dataframe, categorical_features, target, random_seed):
    """
    Functions returns bivariate analysis of target vs feature categories and chi square test
    Params:
        dataframe: type(DataFrame), pandas DataFrame
        categorical_features: type(List), list of column names for categorical features
        target: target y
    """
    for categorical_feature in categorical_features:
        
        # sample the data for the chi2 test
        sample_df_feat = dataframe[categorical_feature].sample(frac=0.02, replace=True, random_state=random_seed).values
        sample_df_target = dataframe[target].sample(frac=0.02, replace=True, random_state=random_seed).values
        
        print("\n" + categorical_feature)
        # contingency table
        sampled_cross_df = pd.crosstab(sample_df_feat,
                          sample_df_target)
        
        # create chi2 contingency table
        s, p, d, e = chi2_contingency(sampled_cross_df)
        critical_value = chi2.ppf(p, d)
        if abs(s) >= critical_value:
            print("Reject null hypothesis between {a} and {b}".format(a=target, b=categorical_feature))
        else:
            print("Do not reject null hypothesis between {a} and {b}".format(a=target, b=categorical_feature))
        
        # plot the comparison plot
        if dataframe[categorical_feature].nunique() <= 10:
            categoricalComparisonPlot(dataframe, categorical_feature, target)
        
    return 


def fill_nulls_mode(df, cols):
    """
    Fill null values with the mode of the column cols
    Params:
        df = pandas DataFrame
        cols = column name
    """
    
    return df[cols].fillna(df[cols].mode()[0])


def dataPreparation(df, target, categorical_features):
    """
    Performs data preparation steps to dataframe
    """
    
    # step 1: Map classification values to binary
    df[target] = df[target].map({' - 50000.': 0, ' 50000+.': 1})
    
    # step 2: imputing missing values
    
    # First, we replace question marks with nulls
    df = df.replace(' ?', np.nan)

    # Let us display which columns have null values:
    print("Columns which have missing values before imputation:")
    print(df.isnull().sum().sort_values(ascending=False).head(8))
    
    # impute nulls with the mode
    for cols in df.columns:
        df[cols] = fill_nulls_mode(df, cols)
    
    return df


def featurePrep(df, df_test, categorical_features):
    """
    Preparation of features according to explanation in section 2 of notebook
    Params:
        df : train dataframe
        df_test : test dataFrame
        categorical_features : list
    """

    # remove y from categorical features
    categorical_features_no_y = [cols for cols in categorical_features if cols != "income classification"]

    # Apply feature preparation to train_data
    train = dataPreparation(df.drop(columns="instance weight"), "income classification",
                        categorical_features_no_y)
    test = dataPreparation(df_test.drop(columns="instance weight"), "income classification",
                            categorical_features_no_y)
    
    # Creating training and test features and target variable

    X, y = train.drop(columns=["income classification"]), \
    train[["income classification"]].rename(columns={"income classification" : "y"})["y"] # train

    X_t, y_t = test.drop(columns=["income classification"]), \
    test[["income classification"]].rename(columns={"income classification" : "y"})["y"] # test


    # encoding categorical data
    le = LabelEncoder()
    for cols in categorical_features_no_y:
        X[cols] = le.fit_transform(X[cols])
        X_t[cols] = le.transform(X_t[cols])
        
    # standard scaling features
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_t), columns=X.columns)

    return X_train, y, X_test, y_t


def classificationModel(X_train, y):
    """
    Applies three classifiers to problem as mentioned in section 3 of notebook
    """

    # Applying the Logistic Regression algorithm
    print("Logistic Regression Model: \n")
    logreg = LogisticRegression(solver="lbfgs") # define model
    score_logreg = cross_val_score(logreg, X_train, y, cv=5) # obtain cross validation score
    print("CV scores = {}".format(score_logreg))
    print("Mean CV score = {}".format(round(np.mean(score_logreg), 3)))
    model_logreg = logreg.fit(X_train, y) # fit to training
    logreg_importance = model_logreg.coef_[0]
    print("Feature importances:")
    for i,v in enumerate(logreg_importance):
        print(X_train.columns[i] + ' Score: %.5f' % (v))

    print("\n\n")

    # Applying the Gaussian Naive Bayes Classifier
    print("Gaussian Naive Bayes Model: \n")
    gnb = GaussianNB()
    score_gnb = cross_val_score(gnb, X_train, y, cv=5)
    print("CV scores = {}".format(score_gnb))
    print("Mean CV score = {}".format(round(np.mean(score_gnb), 3)))
    model_gnb = gnb.fit(X_train, y)
    print("\n\n")

    # Applying Random forest classifier
    print("Random Forest Classifier Model: \n")
    rf_parameters  = {"n_estimators" : [5, 10]}
    print("Grid search parameters = {}".format(str(rf_parameters)))
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, rf_parameters, cv=5)
    model_rf = clf.fit(X_train, y)
    score_rf = clf.cv_results_["mean_test_score"]
    print("Mean CV scores = {}".format(np.round(score_rf, 3)))
    print("Best number of estimators = {}".format(clf.best_estimator_.n_estimators))
    print("Feature importances:")
    for i, v in enumerate(clf.best_estimator_.feature_importances_):
        print(X_train.columns[i] + ' Score: %.5f' % (v))

    return model_logreg, model_gnb, model_rf


def modelAssessment(model_logreg, model_gnb, model_rf, X_test, y_t):
    """
    Gets score when applying to test data
    """
    result = pd.DataFrame(data={"Model" : ["Score: Logistic Regression",
                                           "Score: Gaussian Naive Bayes",
                                           "Score: Random Forest Classifier (trees=10)"],
                                "Score" : [round(model_logreg.score(X_test, y_t) * 100, 2),
                                           round(model_gnb.score(X_test, y_t) * 100, 2),
                                           round(model_rf.score(X_test, y_t) * 100, 2)]})
    return result.set_index("Model")