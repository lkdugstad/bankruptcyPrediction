# -----------------------------------------------------------
# This is the Exploratory Data Analysis class.
# Here we create graphs and do the data analysis
#
# 2022 Lars Kristian Dugstad, Mainz, Germany
# email lkdugstad@googlemail.com
# -----------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import project


class EDA:

    def __init__(self, df):
        self.df = df

    def distribution(self, object='is_bankrupt_after_years', turnLabel=False, title='Title', barplot=True, path='images/testimage.png'):
        df = self.df
        if turnLabel:
            plt.xticks(rotation=90)
        plt.title(title, fontsize=18, color='black')
        if barplot:
            sns.countplot(x=df[object])
        else:
            df.groupby(object).size().plot(kind='pie', autopct='%1.1f%%', pctdistance=0.75)
            plt.ylabel("")
            plt.legend(['0: 24,785', '1: 410', '2: 515', '3: 495'], loc=2)
            plt.legend(['0: 24,785', '1: 24,785', '2: 24,785', '3: 24,785'], loc=2)
        #plt.show()
        plt.savefig(path, bbox_inches='tight')


if __name__ == '__main__':
    data = project.DataPreparer()
    data.impute_missing_values()
    data.synthetic_minority_oversampling()
    df = data.df[data.df['is_bankrupt_after_years'] > 0]
    EDA(data.df).distribution(object='is_bankrupt_after_years', turnLabel=False, title='Distribution bankrupt years', barplot=False)
