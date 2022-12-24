# Synthetic Data 3: 10 features,
# 6 is irrelevant to class, 4 relevant
# random values in range [0,20]
# class is by biggest feature value (class 0,1,2,3)
# 500 rows
import copy
import pandas as pd
import numpy as np
import math
import time
import datetime
import itertools
import random
# Calculating  gini impurity for the attiributes
def gini_split_a(df1,attribute_name):
    attribute_values = df1.value_counts([attribute_name])
    gini_A = 0
    for key in attribute_values.keys():
      dict = {}
      for j in Data[y]:
        dict.update({j: 0})
      k = key[0]
      df_new = df1.loc[df1[attribute_name] == k]
      df_new = df_new.values.tolist()
      for i in df_new:
          dict[i[1]] = dict[i[1]] + 1
      n_k = attribute_values[k]
      n = df1.shape[0]
      gini_A = gini_A + (( n_k[0] / n) * gini_impurity(dict))
    return gini_A
# information : entropy calc

def gini_impurity(value_counts):
  n = 0
  for i in value_counts.keys():
    n = n + value_counts[i]
  p_sum = 0
  for key in value_counts.keys():
    p_sum = p_sum + (value_counts[key] / n) * (value_counts[key] / n)
  gini = 1 - p_sum
  return gini

# information : entropy calc
def pi(data):
  dict = {}
  for row in data:
    if tuple(row) not in dict.keys():
      dict.update({tuple(row):1})
    else:
      dict[tuple(row)] = dict[tuple(row)] + 1
  for i in dict.keys():
    dict[i] = dict[i]/len(data)
  return(dict)

def filtering(df, limits_dic):
  cond = None
  # Build the conjunction one clause at a time
  for key, val in limits_dic.items():
      if cond is None:
          cond = df[key] == val
      else:
          cond = cond & (df[key] == val)
  return(df.loc[cond])

def data_split(data,*n):
  d = data[list(n)]
  d = [row for row in d.values]
  dsplit = []
  uniqe = []
  l = 0
  for row in d:
    if (not any(list(map(lambda x : list(row) == list(x), uniqe)))):
      uniqe.append(row)
  for u in uniqe:
    limits_dic = {}
    for i in n:
      limits_dic.update({i :{}})
    for index,i in enumerate(u):
      for indexj,j in enumerate(n):
        if index==indexj:
          limits_dic[j]=i
    dsplit.append(filtering(data, limits_dic))
  return(dsplit)

def H (data,xi,*x):
  H = 0
  if(x == ()):
    data_sets = data_split(data,xi)
  else:
    data_sets = data_split(data,*x)
  for data_set in data_sets:
    p_data_set = len(data_set)/len(data)
    if(x == ()):
      H = H - p_data_set*math.log(p_data_set,2)
    else:
      data_i = data_set[[xi,*x]]
      data_i = [row for row in data_i.values]
      p_i = pi(data_i)
      Hj = 0
      for j in p_i:
        Hj = Hj - p_i[j]*math.log(p_i[j],2)
      H = H + p_data_set*Hj
  return(H)

# mutual information
#calc the mutual informaion betweem columns x1,..,xn and column y in data
def I(data,y,*x):
  if (len(x) == 1):
    return (H(data,y) - H(data,y,*x)) # I(x1;y) = H(y)+H(y|x1)
  else:
    Info = I(data,y,*x[:-1]) + H(data,x[-1],*x[:-1]) - H(data,x[-1],*(*x[:-1], y))
    return Info

#calc the coverage value of selecting features x1,..,xn from data with label column y
def Coverage(f,x): # x is a list contain the coverage value of any of the rq readings
  coverage = 0
  z = np.zeros(y)
  for i in x:
    z[i] = 1
  for inj,j in enumerate(f):
    coverage = coverage + math.log(np.sum(np.multiply(f[inj],z)) + 1)
  return coverage

# create dict of readings and labels
def Reading_to_labels(X,TR,label):
  data_readings = [tuple(row) for row in X.values]
  for index_i, i in enumerate(data_readings):
    TR.update({i: {}})
    for index_j, j in enumerate(label):
      if index_j == index_i:
        TR[i] = j
  return (TR)

# pairs of different readings
def P(TR):
  P = []
  for index_i,i in enumerate(TR):
    for index_j,j in enumerate(TR):
      if index_j > index_i:
        if TR[i] != TR[j]:
          P.append([i,j])
  print("end pairs")
  return(P)

# calc how many different features between pair of reading with diff class
def F(p,y):
  # dict of arrays with size = number of features
  f = {}
  for ind,i in enumerate(p):
    f.update({ind: np.zeros(y)})
  for j,rq in enumerate(p):
    for index,i in enumerate(rq[0]):
      if (i != rq[1][index]):
          f[j][index] = 1
  print("end f")
  return(f)

# feature selection by information function value
start = time.time()
# define B = number of features we want to select
Features_number = 4
B = 4
y = 10 #features number
runs_number = 100 #number of runs of the same process
# open file to save rum results
filename1 = r'Synthetic_Data_3_Information_Coverage_10f B='+ str(B) + r' ' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
outFileName=r"C:\Users\hila\Desktop\runs\ " + filename1 + r".txt"
outFile=open(outFileName, "w")

# number of features
random.seed(10)
N = list(range(0, y)) #N = [0,..,features number-1]
# features classes
#Class_features = {0:0,3:1,6:2,9:3}
#F_group= [0,3,6,9] # The set of features that determines the class
find_F_info = 0 # number of Successful experiments
find_F_coverage = 0 # number of Successful experiments
find_F_gini = 0  # number of Successful experiments


for j in range(runs_number):
  label = []
  data = []
  Data = []
  # randomly choose features that decide the class
  Class_features = {}
  F_group = random.sample(N, Features_number)
  F_group.sort()
  for i in range(Features_number):
    Class_features.update({F_group[i]: i})
  print(Class_features)
  for i in range (0,500):#create rows
    row = [int(random.random()*20) for i in range(y)]# 10 features with values [0,20]
    r = np.asarray(row)
    max = -1
    for i in Class_features:
      if row[i] > max:
        max = row[i]
        maxC = Class_features[i]
    label.append(maxC)  # class
    row.append(maxC)
    Data.append(row)
  # Create DataFrame.
  data = copy.deepcopy(Data)
  data = pd.DataFrame(data)
  X = data.drop(y, axis=1)
  Data = pd.DataFrame(Data)
  # Print the output.
  #class distribution
  D = copy.deepcopy(Data)
  print(D[y].value_counts().sort_index(ascending=True))
  print(sorted(D[y].unique()))
  D['class_name'] = pd.DataFrame(sorted(D[y].unique()))
  D['dist_of_class'] = D[y].value_counts().sort_index(ascending=True)
  D.to_csv(r'C:\Users\hila\Desktop\runs\Highest value model'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+r'.csv' , index = False)

  # function create to get all N subsets size B
  subsets = list(itertools.combinations(N, B))
  print("subsets")

  #Coverage Value calc
  TR = {}
  TR = Reading_to_labels(X,TR,label)
  p = P(TR) # create pairs of different readings (with diff class)
  # dict of arrays with size = number of features
  f = F(p,y) # calc how many different features between pair of reading with diff class

  max_information = 0
  max_coverage = 0
  max_gini_gain = 0
  max_inforamation_set = []
  max_coverage_set = []
  max_gini_gain_set = []
  for s in subsets:
    coverage = Coverage(f,s)
    info = I(Data,y,*s)
    df = Data.drop(y, axis=1)
    for i in N:
      if i not in s:
        df = df.drop(i, axis=1)
    records = df.to_records(index=False)
    df = pd.DataFrame(records).apply(tuple, axis=1)
    df = pd.DataFrame(df)
    df.insert(1, "1", Data[y])
    #df = df.drop(0, axis=0)
    gini_gain = 1-gini_split_a(df,0)
    if (gini_gain > max_gini_gain):
      max_gini_gain = gini_gain
      max_gini_gain_set = s
    if (info > max_information):
      max_information = info
      max_inforamation_set = s
    if (coverage/len(p) > max_coverage):
      max_coverage = coverage/len(p)
      max_coverage_set = s
  print(max_information)
  print(max_inforamation_set)
  print(max_coverage)
  print(max_coverage_set)
  print(max_gini_gain)
  print(max_gini_gain_set)
  # Check if the selected features are contained in the set of features that determines the class
  check = all(item in max_inforamation_set for item in F_group)
  if check:
    find_F_info = find_F_info + 1
  check = all(item in max_coverage_set for item in F_group)
  if check:
    find_F_coverage = find_F_coverage + 1
  check = all(item in max_gini_gain_set  for item in F_group)
  if check:
    find_F_gini = find_F_gini + 1

#print(max_set)
#print(max)
print("info")
print(find_F_info/runs_number*100)# Success rate
print("cover")
print(find_F_coverage/runs_number*100)# Success rate
print("gini")
print(find_F_gini/runs_number*100)# Success rate
"""
#find the group of good feauters
max_group = []
for s in subsets:
  info = I(data,y,*s)
  if info >= max:
    max_group.append(s)
print(max_group)
"""
outFile.write("""Information success rate is """)
outFile.write(str(find_F_info/runs_number*100))
outFile.write("""\n """)

outFile.write("""Coverage success rate is """)
outFile.write(str(find_F_coverage/runs_number*100))
outFile.write("""\n """)

outFile.write("""Gini index success rate is """)
outFile.write(str(find_F_gini/runs_number*100))
outFile.write("""\n """)
#outFile.write("""maximum information features are \n""")
#outFile.write(str(max_set))
#outFile.write("""\n maximum information value is \n""")
#outFile.write(str(max))
#outFile.write("""\n maximum information group is \n""")
#outFile.write(str(max_group))

# calc run time
end = time.time()

# total time taken
outFile.write(r"Runtime of the program is " + str(round(end - start)) + " sec")
print(f"Runtime of the program is " + str(round(end - start)) + " sec")

#close file
outFile.close()
