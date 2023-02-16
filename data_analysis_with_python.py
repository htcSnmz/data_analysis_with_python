import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

def check_df(dataframe, head=5):
    print("#################### Shape ####################")
    print(dataframe.shape)
    print("#################### Dtypes ####################")
    print(dataframe.dtypes)
    print("#################### NA ####################")
    print(dataframe.isnull().sum())
    print("#################### Head ####################")
    print(dataframe.head())
    print("#################### Tail ####################")
    print(dataframe.tail())
    print("#################### Describe ####################")
    print(dataframe.describe().T)

check_df(df)


def grab_col_names(dataframe, cat_th=10, car_th=30):
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["object", "category", "bool"]]
    num_but_cat = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["int64", "float64"] and
                   dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in cat_cols if dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_bat_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    df = pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": dataframe[col_name].value_counts() / len(dataframe) * 100
        # "Ratio" : dataframe[col_name].value_counts(normalize=True) * 100
    })
    print(df)
    print("######################################")
    if plot:
        sns.countplot(x = col_name, data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        #df[col] = df[col].astype("int64")
        cat_summary(df, df[col].astype("int64").name, True)
    else:
        cat_summary(df, col, True)

def num_summary(dataframe, col_name, plot=False):
    quantiles = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(dataframe[col_name].describe(quantiles).T)
    print("######################################")
    if plot:
        #sns.histplot(x=col_name, data=dataframe)
        dataframe[col_name].hist()
        plt.title(col_name)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)


def target_summary_with_cat_cols(dataframe, target_col, cat_col):
    df = pd.DataFrame({"TARGET_MEAN": dataframe.groupby(cat_col)[target_col].mean()})
    print(df)
    print("################################")

for col in cat_cols:
    target_summary_with_cat_cols(df, "survived", col)


def target_summary_with_num_cols(dataframe, target_col, num_col):
    df = pd.DataFrame({num_col.upper() + "_MEAN": dataframe.groupby(target_col)[num_col].mean()})
    print(df)
    print("################################")

for col in num_cols:
    target_summary_with_num_cols(df, "survived", col)

#Korelasyon Analizi

df2 = pd.read_csv("datasets/breast_cancer.csv")
df2 = df2.iloc[:, 1:-1]
df2.head()

#num_cols2 = [col for col in df2.columns if df2[col].dtypes in [int, float]]



def high_corr_cols(dataframe, plot=False, corr_th=0.9):
    corr = dataframe.corr()
    corr_matrix = corr.abs()
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype("bool"))
    drop_cols = [col for col in upper_triangle_matrix.columns if (upper_triangle_matrix[col] > corr_th).any()]
    if plot:
        sns.set(rc={"figure.figsize": (12, 12)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_cols

high_corr_cols(df2)
