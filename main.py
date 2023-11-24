from utils.visualization import Data


"""
cols = ['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username', 'description length', 'external URL', 'private', '#posts', '#followers', '#follows', 'fake']
        
"""
if __name__ == '__main__':
    data = Data()
    data.pie('private')

