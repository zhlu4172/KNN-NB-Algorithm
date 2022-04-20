import numpy as np
from ctypes import sizeof
import random
import math


def classify_nn(training_filename, testing_filename, k):
  train = open(training_filename, "r").readlines()
  test = open(testing_filename, "r").readlines()
  
  training = []
  testing = []
  
  parameter_size = 0
  # training input
  for l in train:
    append_line = []
    l = l.strip().split(",")
    # append all numbers
    for n in range(0, len(l)-1):
      append_line.append(float(l[n]))
    # append class
    append_line.append(l[len(l)-1])
    # append the entire line
    training.append(append_line)
    
  # testing input
  for l in test:
    append_line = []
    l = l.strip().split(",")
    parameter_size = len(l)
    # append all numbers
    for n in l:
      append_line.append(float(n))
    # append the entire line
    testing.append(append_line)


  # transform the data
  
  result = []
  for single_test in testing:

    distance_store = []
    for single_training in training:
      i = 0
      dis = 0
      while i < parameter_size:
        diff = (single_test[i] - single_training[i])
        dis += math.pow(diff, 2)
        i += 1
      label = single_training[parameter_size]
      distance_pair = [dis, label]
      distance_store.append(distance_pair)
    distance_store.sort(key=lambda x:x[0])
    count_yes = 0
    count_no = 0
    j = 0
    while j < k:
      getting_label = distance_store[j][1]
      if getting_label == "yes":
        count_yes += 1
      elif getting_label == "no":
        count_no += 1
      j += 1
    if count_yes > count_no:
      result.append("yes")
    elif count_yes < count_no:
      result.append("no")
    else:
      result.append("yes")

      

  return result

def classify_nb(training_filename, testing_filename):
  
  # access the data
  train = open(training_filename, "r").readlines()
  test = open(testing_filename, "r").readlines()
  
  training = []
  testing = []
  
  # training input
  for l in train:
    append_line = []
    l = l.strip().split(",")
    # append all numbers
    for n in range(0, len(l)-1):
      append_line.append(float(l[n]))
    # append class
    append_line.append(l[len(l)-1])
    # append the entire line
    training.append(append_line)
    
  # testing input
  for l in test:
    append_line = []
    l = l.strip().split(",")
    # append all numbers
    for n in l:
      append_line.append(float(n))
    # append the entire line
    testing.append(append_line)
    
  # transform the data
  yes = []
  no = []
  for data in training:
    if data[len(data)-1].lower() == "yes":
      yes.append(data)
    else:
      no.append(data)
  
  # calculate P(y) & P(n)
  yes_prob = len(yes) / (len(training))
  no_prob = len(no) / (len(training))
  
  # get yes
  # to find every column
  yes_columns = []
  for i in range(len(testing[0])):
    column = []
    for d in yes:
      column.append(d[i])
    yes_columns.append(column)
  
  # get each col's np.mean and np.std
  yes_mean = []
  yes_std = []
  for x in yes_columns:
    yes_mean.append(np.mean(x))
    yes_std.append(np.std(x))
  
  # get prob density
  yes_probs = []
  for a in testing:
    m = yes_prob
    for b in range(len(a)):
      m *= 1/(yes_std[b]*np.sqrt(2*np.pi)) * np.exp(-1*np.square(a[b]-yes_mean[b])/(2*np.square(yes_std[b])))
    yes_probs.append(m)

  # calculate no
  no_columns = []
  for i in range(len(testing[0])):
    column = []
    for d in no:
      column.append(d[i])
    no_columns.append(column)
  
  # get each col's np.mean and np.std
  no_mean = []
  no_std = []
  for x in no_columns:
    no_mean.append(np.mean(x))
    no_std.append(np.std(x))
  
  # get prob density
  no_probs = []
  for a in testing:
    m = no_prob
    for b in range(len(a)):
      m *= 1/(no_std[b]*np.sqrt(2*np.pi)) * np.exp(-1*np.square(a[b]-no_mean[b])/(2*np.square(no_std[b])))
    no_probs.append(m)
  
  # compare and out put
  return_ls = []
  
  for m in range(len(yes_probs)):
    if yes_probs[m] >= no_probs[m]:
      return_ls.append('yes')
    else:
      return_ls.append('no')
 
  return return_ls


def create_new_csv(foldname,fold_num):
  ten_folds = open(foldname, "r").readlines()

  new_training_file = open("new_training.csv", "w")
  new_testing_file = open("new_testing.csv","w")
  reach_testing_folder = False
  for l in ten_folds:
    if len(l.split()) == 0:
      continue
    # print(l)
    getting_string = "fold" + str(fold_num)
    if "fold" in l and reach_testing_folder == True:
      reach_testing_folder = False
    if getting_string in l:
      reach_testing_folder = True
    if "fold" not in l and reach_testing_folder == False and len(l) != 0:
      new_training_file.write(l)
    if "fold" not in l and reach_testing_folder == True and len(l) != 0:
      new_testing_file.write(l)
    
  new_training_file.close()
  new_testing_file.close()

def knn(train_ls, test_ls, k): # knn
  output = []
  for current in test_ls:
    dist_record = []
    for t in train_ls:
      record = [t[len(t)-1]] # record the class
      # calculate distance 
      dist = 0
      for j in range(len(current)):
        dist += np.square(current[j] - t[j])
      dist = np.sqrt(dist)
      record.append(dist)
      dist_record.append(record)
    # sorted by distance in ascending order
    dist_record.sort(key=lambda x: x[1]) 
    # choose the closest k neighbors
    neighbors = dist_record[:k]
    # yes - no count
    y_count = 0
    n_count = 0
    for n in neighbors:
      if n[0].lower() == "yes":
        y_count += 1
      else:
        n_count += 1
    # output
    if y_count >= n_count:
      output.append("yes")
    else:
      output.append("no")
  return output

def test_my_knn(k):
  # train = open(training_filename, "r").readlines()
  new_test = open("new_testing.csv", "r").readlines()
  correct = []
  for l in new_test:
    l = l.strip().split(",")
    correct.append(l[len(l)-1])
  getting_result = classify_nn("new_training.csv", "final_testing.csv", k)

  # train = open("new_training.csv", "r").readlines()
  # test = open("final_testing.csv", "r").readlines()
  
  # training = []
  # testing = []
  
  # # training input
  # for l in train:
  #   append_line = []
  #   l = l.strip().split(",")
  #   # append all numbers
  #   for n in range(0, len(l)-1):
  #     append_line.append(float(l[n]))
  #   # append class
  #   append_line.append(l[len(l)-1])
  #   # append the entire line
  #   training.append(append_line)
    
  # # testing input
  # for l in test:
  #   append_line = []
  #   l = l.strip().split(",")
  #   # append all numbers
  #   for n in l:
  #     append_line.append(float(n))
  #   # append the entire line
  #   testing.append(append_line)

  # getting_result = knn(training, testing, k)
    




  total = len(correct)
  correct_num = 0
  i = 0
  while i < total:
    if correct[i] == getting_result[i]:
      correct_num += 1
    i += 1
  return correct_num/total * 100


def test_my_nb():
  new_test = open("new_testing.csv", "r").readlines()
  correct = []
  for l in new_test:
    l = l.strip().split(",")
    correct.append(l[len(l)-1])
  getting_result = classify_nb("new_training.csv", "final_testing.csv")

  # train = open("new_training.csv", "r").readlines()
  # test = open("final_testing.csv", "r").readlines()
  
  # training = []
  # testing = []
  
  # # training input
  # for l in train:
  #   append_line = []
  #   l = l.strip().split(",")
  #   # append all numbers
  #   for n in range(0, len(l)-1):
  #     append_line.append(float(l[n]))
  #   # append class
  #   append_line.append(l[len(l)-1])
  #   # append the entire line
  #   training.append(append_line)
    
  # # testing input
  # for l in test:
  #   append_line = []
  #   l = l.strip().split(",")
  #   # append all numbers
  #   for n in l:
  #     append_line.append(float(n))
  #   # append the entire line
  #   testing.append(append_line)

  # getting_result = nb(training, testing)

  total = len(correct)
  correct_num = 0
  i = 0
  while i < total:
    if correct[i] == getting_result[i]:
      correct_num += 1
    i += 1
  print(correct_num)
  print(total)
  return correct_num/total * 100

def strip_last_para(foldname):
  current_test = open(foldname, "r").readlines()
  final_testing_file = open("final_testing.csv", "w")
  for l in current_test:
    if "yes" in l:
      new_adding = l.replace(",yes","")
    if "no" in l:
      new_adding = l.replace(",no","")
    final_testing_file.write(new_adding)


def nb(train_ls, test_ls): # nb
  output = []
  # separate into yes and no list
  y = []
  n = []
  for t in train_ls:
    if t[len(t)-1].lower() == "yes":
      y.append(t)
    else:
      n.append(t)
  # calcualte P(yes) and P(no)
  p_yes = len(y) / (len(train_ls))
  p_no = len(n) / (len(train_ls))
 
  # calculate yes
  # get all columns
  y_cols = []
  for i in range(len(test_ls[0])):
    col = []
    for t in y:
      col.append(t[i])
    y_cols.append(col)
  # calculate mean and standard deviation for each column
  y_mean = []
  y_sd = []
  for c in y_cols:
    y_mean.append(np.mean(c))
    y_sd.append(np.std(c))
  # calculate probability density
  y_probs = []
  for t in test_ls:
    mult = p_yes
    for i in range(len(t)):
      mult *= 1/(y_sd[i]*np.sqrt(2*np.pi)) * np.exp(-1*np.square(t[i]-y_mean[i])/(2*np.square(y_sd[i])))
    y_probs.append(mult)
  # calculate no
  # get all columns
  n_cols = []
  for i in range(len(test_ls[0])):
    col = []
    for t in n:
      col.append(t[i])
    n_cols.append(col)
  # calculate mean and standard deviation for each column
  n_mean = []
  n_sd = []
  for c in n_cols:
    n_mean.append(np.mean(c))
    n_sd.append(np.std(c))
  # calculate no probability density
  n_probs = []
  for t in test_ls:
    mult = p_no
    for i in range(len(t)):
      mult *= 1/(n_sd[i]*np.sqrt(2*np.pi)) * np.exp(-1*np.square(t[i]-n_mean[i])/(2*np.square(n_sd[i])))
    n_probs.append(mult)
  # output
  for i in range(len(y_probs)):
    if y_probs[i] >= n_probs[i]:
      output.append("yes")
    else:
      output.append("no")
  return output


if __name__ == '__main__':
    # Example function calls below, you can add your own to test the task6 function
    # print(classify_nb("training.csv","testing.csv"))
    # create_new_csv("pima-folds.csv",1)
    # strip_last_para("new_testing.csv")
    
    i = 1
    total_accuracy_knn = 0
    total_accuracy_nb = 0
    while i <= 10:
      create_new_csv("pima-CFS-10folds.csv",i)
      strip_last_para("new_testing.csv")
      print(test_my_knn(5))
      total_accuracy_knn += test_my_knn(1)
      print(test_my_nb())
      total_accuracy_nb += test_my_nb()
      i += 1
    print(total_accuracy_knn/10)
    print(total_accuracy_nb/10)

    # print(test_my_knn(5))

    # print(test_my_nb())
    
    
    
