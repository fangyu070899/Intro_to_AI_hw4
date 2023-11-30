from utils.data import Data
from utils.decision_tree import DecisionTree
import pandas as pd
import graphviz

if __name__ == '__main__':

    data = Data()

    # 構建決策樹
    print("==================== consturct decision tree ====================")
    tree = DecisionTree(max_depth=3)
    tree.fit(data.X_train, data.y_train)
    print("Done")

    print("==================== validation data test ====================")
    # validation data 測試
    predictions = tree.predict(data.X_validate)
    print("Validation data predictions:", predictions)

    # validation data 計算 accuracy
    accuracy = tree.accuracy(predictions,data.y_validate)
    print(f"Validation data accuracy = {accuracy}")


    print("==================== test data predict ====================")
    # 新數據進行預測
    test_df = pd.read_csv("new_data/test.csv", usecols=['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username', 'description length', 'external URL', 'private', '#posts', '#followers', '#follows', 'fake'])
    X_test = test_df.drop('fake', axis=1)
    y_test = test_df['fake']

    # 資料離散化
    new_X_test = data.discrete(X_test)

    # 新數據進行預測
    predictions = tree.predict(new_X_test)
    print("Predictions:", predictions)

    # test data 計算 accuracy
    accuracy = tree.accuracy(predictions,y_test)
    print(f"Test data accuracy = {accuracy}")

    # visualize tree
    graph = graphviz.Digraph(comment='Decision Tree')
    tree.visualize_tree(tree.tree, 'Root', graph)
    graph.render("decision_tree", format="png")

