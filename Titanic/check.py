import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

g = pd.read_csv("./data/gender_submission.csv")
t1 = pd.read_csv("./data/xgb_out_new.csv")
t2 = pd.read_csv("./data/xgb_out_new5.csv")

print("accuracy: {:.4f}".format(accuracy_score(t1.Survived, t2.Survived)))

import re
key = r"mat cat hat pat"
p1 = r"[^c|p]at"
pattern1 = re.compile(p1)
print(pattern1.findall(key))


# train = pd.read_csv("D:\ying\Pycharm\ying\Titanic\data\\train.csv")
# cabin = pd.DataFrame(train['Cabin'])
# print(type(cabin))
# print(cabin.iloc[2])