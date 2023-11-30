import pandas as pd
import numpy as np
from graphviz import Digraph

class DecisionTree:
    """
    X : feature data set (特徵資料)
    y : target data set (真帳號或假帳號)
    """
    def __init__(self, max_depth=None):
        self.tree = None
        self.max_depth = max_depth

    """
    計算 entropy (亂度)
    """
    def entropy(self, y):
        labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        # entropy 公式，1e-10是為了避免 probabilities==0 時取log會發生的錯誤
        entropy_value = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy_value
    
    """
    計算 attribute 的 Information gain
    """
    def importance(self, X, y, attribute):
        total_entropy = self.entropy(y)

        # 計算 attribute 的 weighted entropy
        values = set(X[attribute])
        weighted_entropy = 0
        for value in values:
            sub_indices = X[attribute] == value
            sub_entropy = self.entropy(y[sub_indices])
            weighted_entropy += (np.sum(sub_indices) / len(X)) * sub_entropy

        # 計算信息增益
        info_gain = total_entropy - weighted_entropy
        return info_gain

    """
    選擇 Information gain 最大的 attribute
    """
    def choose_attribute(self, X, y, attributes):
        best_attribute = None
        max_info_gain = -float('inf') # 將 max_info_gain 初始化成負無窮

        for attribute in attributes:
            current_info_gain = self.importance(X, y, attribute)
            if current_info_gain > max_info_gain:
                max_info_gain = current_info_gain
                best_attribute = attribute

        return best_attribute

    """
    target 中最多的值 (多數為真帳號或假帳號)
    """
    def plurality_value(self, y):
        labels, counts = np.unique(y, return_counts=True)
        major_label = labels[np.argmax(counts)]
        return major_label

    """
    構建 decision tree
    """
    def fit(self, X, y, attributes=None):
        if attributes is None:
            attributes = list(X.keys())
        self.tree = self.decision_tree_learning(X, y, attributes)

    def decision_tree_learning(self, X, y, attributes, parent_examples=None, depth=0):
        
        # 如果 examples 為空，返回 parent_examples target 的眾數
        if X.empty:
            return self.plurality_value(parent_examples)
        
        # 如果所有 target 都相同，return 該值(真帳號或假帳號)
        labels = np.unique(y)
        if len(labels) == 1:
            return {'label': labels[0]}

        # 如果 feature data 為空或深度達到限制，return 當前 target 的眾數
        if not attributes or (self.max_depth is not None and depth >= self.max_depth):
            return {'label': self.plurality_value(y)}

        # 選擇 information gain 最大的 attribute
        best_attribute = self.choose_attribute(X, y, attributes)

        # 刪除這次選擇的 best_attribute
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]

        # 根據 best_attribute 的不同結果分割 dataset
        subsets = {}
        for value in set(X[best_attribute]):
            sub_indices = X[best_attribute] == value
            subsets[value] = (X.loc[sub_indices], y[sub_indices])

        # 遞迴構建子樹
        sub_trees = {}
        for value, (sub_X, sub_y) in subsets.items():
            sub_trees[value] = self.decision_tree_learning(sub_X, sub_y, remaining_attributes, y, depth=depth + 1)

        return {'attribute': best_attribute, 'sub_trees': sub_trees}
    
    """
    遞迴預測單個 instance
    """
    def predict_instance(self, instance, tree):
        # 檢查現在是不是在 leaf，是的話就回傳 label (真帳號or假帳號)
        if 'label' in tree:
            return tree['label']
        else:
            attribute_value = instance[tree['attribute']]
            if attribute_value not in tree['sub_trees']:
                return None  # 未知的屬性值
            else:
                return self.predict_instance(instance, tree['sub_trees'][attribute_value])
    """
    對 feature dataset 進行預測
    """
    def predict(self, X):
        return np.array([self.predict_instance(X.iloc[i], self.tree) for i in range(len(X))])
    
    """
    計算 accuracy
    """
    def accuracy(self, predictions, answers):
        answers_list = answers.to_numpy()
        num=0
        correct = 0
        for i in range(len(predictions)):
            num+=1
            if answers_list[i] == predictions[i]:
                correct+=1
        return correct/num

    """
    將 tree 視覺化
    """
    def visualize_tree(self, tree, parent_name, graph):
        if 'label' in tree:
            graph.node(str(parent_name), f'Label: {tree["label"]}', shape='circle', color='lightblue2', style='filled')
        else:
            graph.node(str(parent_name), f'Attribute: {tree["attribute"]}', shape='box')
            for value, subtree in tree['sub_trees'].items():
                self.visualize_tree(subtree, f'{parent_name}_{value}', graph)
                graph.edge(str(parent_name), f'{parent_name}_{value}', label=str(value))