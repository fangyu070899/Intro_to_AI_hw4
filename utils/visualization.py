import matplotlib.pyplot as plt
import pandas as pd


class Data:
    def __init__(self) -> None:
        # 所有 column 的標題
        cols = ['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username', 'description length', 'external URL', 'private', '#posts', '#followers', '#follows', 'fake']
        # 讀取 training data     
        self.df_train = pd.read_csv("new_data/train.csv", usecols=cols)
        # 將讀取進來的資料拆成真帳號與假帳號兩個 dataset -> 為了分別作圖
        self.df_notfake = self.df_train[self.df_train['fake']==0]
        self.df_fake = self.df_train[self.df_train['fake']==1]

    # 畫 histogram
    def histogram(self,col):
        plt.hist([self.df_notfake[col], self.df_fake[col]], label = ['not fake', 'fake'],color=['lightblue','rosybrown'], stacked=True)
        plt.legend()
        plt.xlabel(col)
        plt.show()

    # 畫 boxplot
    def boxplot(self,col):
        plt.boxplot([self.df_notfake[col], self.df_fake[col]],  labels = ['not fake', 'fake'])
        plt.ylabel(col)
        plt.show()

    # 畫 pie chart
    def pie(self,col):
        # 統計有多少個1和0
        counts_notfake = self.df_notfake[col].value_counts()
        counts_fake = self.df_fake[col].value_counts()

        # 繪製圓餅圖
        plt.pie(counts_notfake, labels=counts_notfake.index, autopct='%1.1f%%',colors=['thistle', 'slategrey'])
        plt.title('Distribution of being private - notfake')
        plt.show()

        plt.pie(counts_fake, labels=counts_fake.index, autopct='%1.1f%%',colors=['thistle', 'slategrey'])
        plt.title('Distribution of being private - fake')
        plt.show()

    def print_data(self):
        print(self.df_notfake)
