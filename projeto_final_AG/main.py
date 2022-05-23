import csv
from collections import defaultdict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

data_set = pd.read_csv('dataset/data_car.csv')

lb_buying = preprocessing.LabelEncoder()
data_set["buying"] = lb_buying.fit_transform(data_set["buying"])

lb_maint = preprocessing.LabelEncoder()
data_set["maint"] = lb_maint.fit_transform(data_set["maint"])

lb_doors = preprocessing.LabelEncoder()
data_set["doors"] = lb_doors.fit_transform(data_set["doors"])

lb_persons = preprocessing.LabelEncoder()
data_set["persons"] = lb_persons.fit_transform(data_set["persons"])

lb_lub_boot = preprocessing.LabelEncoder()
data_set["lug_boot"] = lb_lub_boot.fit_transform(data_set["lug_boot"])

lb_safety = preprocessing.LabelEncoder()
data_set["safety"] = lb_safety.fit_transform(data_set["safety"])

lb_class = preprocessing.LabelEncoder()
data_set["class"] = lb_class.fit_transform(data_set["class"])

my_dict = defaultdict(list)

with open('dataset/data_set.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for line in csv_reader:
        for key, value in line.items():
            my_dict[key].append(value)

print(my_dict)
# dick = pd.read_csv('dataset/data_set.csv').to_string()

# dados_filtrados = data_set[["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]]
# sns.set_theme(style="ticks")
# grafico = sns.pairplot(dados_filtrados)
#
# plt.show()
