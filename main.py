from utils.visualization import Data
from utils.decision_tree import DecisionTree
import pandas as pd
import numpy as np

"""
cols = ['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username', 'description length', 'external URL', 'private', '#posts', '#followers', '#follows', 'fake']
"""
if __name__ == '__main__':
    # 創建示例數據集
    data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
        'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
    }

    df = pd.DataFrame(data)

    # 提取特徵和目標變量
    X = df.drop('PlayTennis', axis=1)
    y = df['PlayTennis']

    # 構建決策樹
    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)

    # 新數據進行預測
    new_data = {
        'Outlook': ['Rain'],
        'Temperature': ['Mild'],
        'Humidity': ['High']
    }

    new_df = pd.DataFrame(new_data)

    # 進行預測
    predictions = tree.predict(new_df)

    print("Predictions:", predictions)


