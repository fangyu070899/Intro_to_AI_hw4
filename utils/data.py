from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Data:
    def __init__(self) -> None:
        # 所有 column 的標題
        cols = ['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username', 'description length', 'external URL', 'private', '#posts', '#followers', '#follows', 'fake']
        
        print("==================== read data ====================")
        # 讀取 training data     
        self.df_train = pd.read_csv("new_data/train.csv", usecols=cols)
        # 讀取 validating data     
        self.df_train = pd.read_csv("new_data/test.csv", usecols=cols)
        print("Done")

        self.train_rows, self.train_columns = self.df_train.shape

        self.new_df_train = self.discrete(self.df_train)

        # 將讀取進來的資料拆成真帳號與假帳號兩個 dataset -> 為了分別作圖
        self.df_notfake = self.new_df_train[self.new_df_train['fake']==0]
        self.df_fake = self.new_df_train[self.new_df_train['fake']==1]

        # 切 training dataset
        print("==================== parse data ====================")
        self.parse_data(self.new_df_train)
        print("Done")

    def parse_data(self, df):
        # 將數據切割成訓練集和測試集，test_size=0.2 表示測試集占總數據的 20%
        self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split( df.drop('fake', axis=1), df['fake'], test_size=0.2, random_state=42)

    """
    將連續資料離散化並回傳新的 dataframe
    """
    def discrete(self,df):
        print("==================== discrete data ====================")
        df_copy = df.copy()

        for i in range(self.train_rows):

            # 令 fullname_words 長度大於 5 者皆等於 5
            if df_copy.loc[i,'fullname words'] > 5 :
                df_copy.loc[i,'fullname words'] = 5 

            # 每 0.2 為一區間
            df_copy.loc[i,'nums/length username'] = np.floor(df_copy.loc[i,'nums/length username']/0.2) 
            df_copy.loc[i,'nums/length fullname'] = np.floor(df_copy.loc[i,'nums/length fullname']/0.2) 

            df_copy.loc[i,'description length'] = np.floor(df_copy.loc[i,'description length']/20)
            if df_copy.loc[i,'description length'] > 7:
                df_copy.loc[i,'description length'] = 7
                
            if df_copy.loc[i,'#posts'] < 10:
                df_copy.loc[i,'#posts'] = 0
            elif df_copy.loc[i,'#posts'] < 100:
                df_copy.loc[i,'#posts'] = 10
            elif df_copy.loc[i,'#posts'] < 1000:
                df_copy.loc[i,'#posts'] = 100
            else:
                df_copy.loc[i,'#posts'] = 1000

            if df_copy.loc[i,'#followers'] < 100:
                df_copy.loc[i,'#followers'] = 0
            elif df_copy.loc[i,'#followers'] < 500:
                df_copy.loc[i,'#followers'] = 100
            elif df_copy.loc[i,'#followers'] < 1000:
                df_copy.loc[i,'#followers'] = 500
            else:
                df_copy.loc[i,'#followers'] = 1000

            if df_copy.loc[i,'#follows'] < 100:
                df_copy.loc[i,'#follows'] = 0
            elif df_copy.loc[i,'#follows'] < 500:
                df_copy.loc[i,'#follows'] = 100
            elif df_copy.loc[i,'#follows'] < 1000:
                df_copy.loc[i,'#follows'] = 500
            else:
                df_copy.loc[i,'#follows'] = 1000

        print("Done")
        return df_copy

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
