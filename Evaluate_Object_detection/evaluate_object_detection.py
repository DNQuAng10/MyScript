import argparse
import cv2 as cv 
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from classes_name import classes_name


# _------------------------------------------------------------------------------
# Calculate IoU between predictions and truth 
def IoU(truth_classes_bbox, pred_classes_bbox):
  # pred_classes_bbox:  is a results of yolov5
  # form:               class conf xmin ymin xmax ymax
  pred_xmin = int(pred_classes_bbox[2])
  pred_xmax = int(pred_classes_bbox[4])
  pred_ymin = int(pred_classes_bbox[3])
  pred_ymax = int(pred_classes_bbox[5])
  # truth_classes_bbox: is a ground-truth bbox
  # form:               class xmin ymin xmax ymax
  true_xmin = int(truth_classes_bbox[1])
  true_xmax = int(truth_classes_bbox[3])
  true_ymin = int(truth_classes_bbox[2])
  true_ymax = int(truth_classes_bbox[4])
  S_pred = (pred_xmax - pred_xmin)*(pred_ymax - pred_ymin)
  S_true = (true_xmax - true_xmin)*(true_ymax - true_ymin)
  
  # Intersection area
  # 
  intersect_xmin = pred_xmin if pred_xmin > true_xmin else true_xmin
  intersect_xmax = pred_xmax if pred_xmax < true_xmax else true_xmax
  intersect_ymin = pred_ymin if pred_ymin > true_ymin else true_ymin
  intersect_ymax = pred_ymax if pred_ymax < true_ymax else true_ymax
  S_intersect = (intersect_xmax - intersect_xmin)*(intersect_ymax - intersect_ymin)
  #print("Intersection area: xmin xmax ymin ymax S_intersect:", intersect_xmin, intersect_xmax, intersect_ymin, intersect_ymax, S_intersect)

  # IoU = S_intersect/(S_true + S_pred - S_intersect)
  iou = S_intersect/(S_true + S_pred - S_intersect)
  return iou

def compare_iou_truth_pred(true_txt_path, pred_txt_path, iou_thresh = 0.5):
  print("true_txt_path:", true_txt_path)
  print("pred_txt_path:", pred_txt_path)
  with open(true_txt_path, "r") as file:
    true_classes_bbox = file.read().split()
  
  with open(pred_txt_path, "r") as file:
    pred_classes_bbox = file.read().split()
  
  true_classes_bbox_1 = [true_classes_bbox]
  if len(true_classes_bbox) != 5:
    x = len(true_classes_bbox)//5
    true_classes_bbox_1 = []
    for i in range(x):
      true_classes_bbox_1.append(true_classes_bbox[i*5:(i*5 + 5)])
  
  pred_classes_bbox_1 = [pred_classes_bbox]
  if len(pred_classes_bbox) != 6:
    x = len(pred_classes_bbox)//6
    pred_classes_bbox_1 = []
    for i in range(x):
      pred_classes_bbox_1.append(pred_classes_bbox[i*6:(i*6 + 6)])

  print("--Ground-truth:", true_classes_bbox_1)
  print("--Predictions :", pred_classes_bbox_1)
  
  print("--IoU pred-true")
  Dict_pred_correspond2_true = {}
  pred_correspond2_true = []
  if len(true_classes_bbox_1) != 0 and len(pred_classes_bbox_1) != 0:
    for pred_ in pred_classes_bbox_1:
      for true_ in true_classes_bbox_1:
        iou = IoU(true_, pred_)
        print("-->IoU between %s-%s ="%(pred_[0], true_[0]), iou)
        #if iou > iou_thresh:
        pred_correspond2_true.append([iou, pred_, true_])
      print("-"*100)
      Dict_pred_correspond2_true[os.path.basename(true_txt_path)] = pred_correspond2_true
  elif len(true_classes_bbox_1) == 0:
    for pred_ in pred_classes_bbox_1:
      pred_correspond2_true.append([0, pred_, [None, None, None, None]])
      Dict_pred_correspond2_true[os.path.basename(true_txt_path)] = pred_correspond2_true
  elif len(pred_classes_bbox_1) == 0:
    for true_ in true_classes_bbox_1:
      pred_correspond2_true.append([0, [None, None, None, None, None, None], true_])
      Dict_pred_correspond2_true[os.path.basename(true_txt_path)] = pred_correspond2_true
  return Dict_pred_correspond2_true
  
def eval_true_pred(true_dir, pred_dir):
  List_true_txt = glob.glob("%s/*.txt"%true_dir)

  List_eval_true_pred = []
  for true_txt_path in List_true_txt:
    pred_txt_path = os.path.join(pred_dir, os.path.basename(true_txt_path))
    Dict_pred_correspond2_true = compare_iou_truth_pred(true_txt_path, pred_txt_path)
    List_eval_true_pred.append(Dict_pred_correspond2_true)
  return List_eval_true_pred


# -_------------------------------------------------------------------------------
# Create statistic table 
def create_pred_table(List_eval_true_pred):
  List_txt = []
  List_eval = []
  for eval_true_pred in List_eval_true_pred:
    for k, v in eval_true_pred.items():
      #print(k)
      #print(len(v))
      for i in v:
        List_txt.append(k)
        List_eval.append(i)
  #print(len(List_txt), len(List_eval))
  #print(List_eval)
  #List_txt

  data = pd.DataFrame(data = {
      "text file":List_txt,
      "IoU":[i[0] for i in List_eval],
      "Confidence":[i[1][1] for i in List_eval],
      "predict label":[i[1][0] for i in List_eval],
      "truth label":[i[2][0] for i in List_eval],
      "predict bbox":[i[1][2:] for i in List_eval],
      "truth bbox":[i[2][1:] for i in List_eval]
  })
  return data

# --_-------------------------------------------------_____----------------------
# ---Statistic classes---
# Count number of label in each classes
def truth_classes_count(List_txt_path, classes_name):
  List_classes = []
  for i in List_txt_path:
    #print(i)
    with open(i, "r") as file:
      #print(file)
      classes_bbox = file.read().split()
    #print(classes_bbox)
    if len(classes_bbox) != 0:
      if len(classes_bbox) == 5:
        List_classes.append(classes_bbox[0])
      else:
        x = len(classes_bbox)//5
        for i1 in range(x):
          List_classes.append(classes_bbox[i1*5])
  data = pd.Series(List_classes, dtype = str)
  
  #count = 0
  Dict_true_classes_count = {}
  for v in classes_name.values():
    count = 0
    if v in data.unique():
      #print(v)
      count = data.value_counts()[v]
    #else:
      #count = 0
    Dict_true_classes_count[v] = count 
  return Dict_true_classes_count

# Count number of prediction labels
def pred_classes_label(data, classes_name):
  data = data[data["IoU"] > 0.5]
  Dict_pred_classes_count = {}
  for v in classes_name.values():
    count = 0
    if v in data["predict label"].unique():
      count = data["predict label"].value_counts()[v]
    #else:
      #count = 0
    Dict_pred_classes_count[v] = count
  return Dict_pred_classes_count

# Plot number of label in each classes histogram
def label_on_bar(bars, pos = 3, color = (0, 0, 0)):
  for bar in bars:
    height = bar.get_height()
    width = bar.get_width()
    xloc = bar.get_x() + bar.get_width()/pos
    ax.annotate("{}".format(height), 
                xy = (xloc, height + 0.2),
                xytext = (0, 3), 
                textcoords = "offset points",
                ha = "left", va = "center",
                color = color)

def plot_classes_count(Dict_true_classes_count, Dict_pred_classes_count, width_figsize = 12, height_figsize = 10, save_dir = None):
  global ax
  x = np.arange(len(Dict_pred_classes_count))
  width = 0.35

  fig, ax = plt.subplots(figsize = (width_figsize, height_figsize))
  p1 = ax.bar(x - width/2, list(Dict_true_classes_count.values()), width, label = "Truth labels")
  p2 = ax.bar(x + width/2, list(Dict_pred_classes_count.values()), width, label = "Prediction labels")
  label_on_bar(p1)
  label_on_bar(p2)
  ax.set_xticks(x)
  ax.set_xticklabels(list(Dict_true_classes_count.keys()))
  ax.set_xlabel("Classes")
  ax.set_ylabel("Number")
  plt.title("Number of truth and prediction label in each classes with IoU threshold = 0.5")
  plt.legend(loc = "best")

  save_path = os.path.join(save_dir, "classes_count.jpg")
  plt.savefig(save_path)
  plt.show()

# Plot true false positive and true positive-false negative
def true_positive_false_negative(data, classes_names):
  data_1 = data[data["IoU"] > 0.5]
  Dict_true_positive_false_negative = {}
  for label in data_1["truth label"].unique():
    data_2 = data_1[data_1["truth label"] == label]
    tp, fn = 0, 0
    for i in range(len(data_2)):
      #print(i)
      if data_2["predict label"].iloc[i] == data_2["truth label"].iloc[i]:
        tp += 1
      else:
        fn += 1
    #print(tp)
    #fp = len(data_2) - tp
    Dict_true_positive_false_negative[label] = [tp, fn]
  return Dict_true_positive_false_negative

def true_false_positive(data, classes_names):
  data_1 = data[data["IoU"] > 0.5]

  Dict_true_false_positive = {}
  for label in data_1["truth label"].unique():
    tp, fp = 0, 0
    for i in range(len(data_1)):
      if data_1["predict label"].iloc[i] == label:
        if data_1["truth label"].iloc[i] == data_1["predict label"].iloc[i]:
          tp += 1
        elif data_1["predict label"].iloc[i] != data_1["truth label"].iloc[i]:
          fp += 1
    Dict_true_false_positive[label] = [tp, fp]
  return Dict_true_false_positive 

def label_on_bar_tfp(bars, values, pos, pos_1, colors = (0, 1, 0)):
  for i, bar in enumerate(bars):
    #print(i)
    value = " %s"%values[i]
    height = bar.get_height()
    width = bar.get_width()
    xloc = bar.get_x() + bar.get_width()/pos + pos_1
    ax.annotate("{}".format(value), 
                xy = (xloc, height + 0.2),
                xytext = (0, 3), 
                textcoords = "offset points",
                ha = "left", va = "center",
                color = colors)

def plot_true_false_positive(Dict_true_false_positive, size = (10, 8), titles = ["True positive", "False positive"], save_dir = None):
  global ax
  x = np.arange(len(Dict_true_false_positive))
  width = 0.35
  print(x)

  y1 = [(v[0] + v[1]) for v in Dict_true_false_positive.values()]
  y2 = [v[1] for v in Dict_true_false_positive.values()]
  print(y1)
  print(y2)
  fig, ax = plt.subplots(figsize = size)
  p1 = ax.bar(x, y1, width, color = "g", label = titles[0])
  p2 = ax.bar(x, y2, width, color = "r", label = titles[1])
  label_on_bar_tfp(p1, [v[0] for v in Dict_true_false_positive.values()], 10, 0)
  label_on_bar_tfp(p1, y2, 1, 0.1, colors = (1, 0, 0))
  ax.set_xticks(x)
  ax.set_xticklabels(Dict_true_false_positive.keys())
  ax.set_xlabel("Classes")
  ax.set_ylabel("Number of label")
  plt.title("%s and %s in each classes"%(titles[0], titles[1]))
  plt.legend()
  save_path = os.path.join(save_dir, "%s_%s.jpg"%(titles[0], titles[1]))
  plt.savefig(save_path)
  plt.show()

# Calculate confusion matrix
def calculate_confusion_matrix(data, classes_name, save_dir = None):
  data_1 = data[data["IoU"] > 0.5]
  preds = data_1["predict label"]
  truths = data_1["truth label"]
  print(truths.value_counts())
  print(preds.value_counts())
  cl_report = classification_report(truths, preds, output_dict = False)
  print(cl_report)
  
  cl_report = classification_report(truths, preds, output_dict = True)
  #cl_report = dict(cl_report)
  #print(list(cl_report.values())[0]["precision"])
  #print(cl_report)
  #print(cl_report.keys())
  #print([i["precision"] for i in list(cl_report.values()) if type(i) != float])
  result_data = pd.DataFrame(data = {
    "":[i for i in cl_report.keys() if i != "accuracy"],
    "precision":[i["precision"] for i in list(cl_report.values()) if type(i) != float],
    "recall":[i["recall"] for i in list(cl_report.values()) if type(i) != float],
    "f1-score":[i["f1-score"] for i in list(cl_report.values()) if type(i) != float],
    "support":[i["support"] for i in list(cl_report.values()) if type(i) != float],

  })
  result_data.to_csv(os.path.join(save_dir, "result_table.csv"))

  cm = confusion_matrix(truths, preds, labels = list(classes_name.values()))
  #print(cm)
  plt.figure(figsize = (12, 10))
  cm_plot = sns.heatmap(cm, 
                        xticklabels = list(classes_name.values()), 
                        yticklabels = list(classes_name.values()),
                        annot = True, fmt = "d")
  #fig = cm_plot.get_fig()
  
  plt.xlabel("Truth labels")
  plt.ylabel("Prediction labels")
  plt.title("Statistic between true and prediction labels with IoU threshold = 0.5")
  save_path = os.path.join(save_dir, "confusion_matrix.jpg")
  plt.savefig(save_path)
  plt.show()

# Plot mAP that is calculated by mAP Cartucho_repository
#   Use output file 
def Plot_mAP(output_file_path, classes_name):
  with open(output_file_path, "r") as file:
    data = file.read().split("\n")
  #print(data)
  
  end = 0
  for i, text in enumerate(data):
    #print(text)
    if text == "":
      data.remove(text)
    if text.startswith("mAP =") == True:
      end = i
  #print(end)
    
  map_classes = data[1:end]
  #print(map_classes)

  Dict_map_classes = {}
  x = len(map_classes[:-2])//3
  #print(x)
  for i in range(x):
    #print(i)
    ap = map_classes[(i*3)].split()[0][:-1]
    classes = map_classes[i*3].split()[2]
    Dict_map_classes[classes] = ap
    #break
  #print(Dict_map_classes)
  map = map_classes[-1].split()[-1]
  #print(map)

  map_data = pd.DataFrame(data = {
      "":list(Dict_map_classes.keys()),
      "Average Precision":list(Dict_map_classes.values()),
  })
  #map_data = pd.concat([map_data, ["mAP", map]])
  print(map_data)
  print("Plot histogram")

  plt.figure(figsize = (8, 6))
  plt.plot([v for v in classes_name.values() if v in Dict_map_classes.keys()],
          [float(Dict_map_classes[v]) for v in classes_name.values() if v in Dict_map_classes.keys()], 
          "r*-", label = "Average Precision")
  plt.title("mAP = %s"%map)
  plt.legend(loc = "best")
  plt.grid()
  plt.show()

def main(
  ground_truth_classes_bbox, 
  pred_classes_bbox, 
  classes_name, 
  width_figsize = 12, 
  height_figsize = 8, 
  save_dir = None
):
  if os.path.exists(save_dir) == False:
    os.makedirs(save_dir)

  #with open(classes_name, "r") as file:
  #  classes_name = dict(file.read())
  
  # Calculate IoU between predict and ground truth all bounding boxes
  List_eval_true_pred = eval_true_pred(ground_truth_classes_bbox, pred_classes_bbox)
  
  # Create a statical table 
  data = create_pred_table(List_eval_true_pred)
  data.to_csv(os.path.join(save_dir, "iou_data.csv"))
  
  # Count truth and predict label of each class
  List_truth_txt_path = glob.glob("%s/*"%ground_truth_classes_bbox)
  Dict_true_classes_count = truth_classes_count(List_truth_txt_path, classes_name)
  Dict_pred_classes_count = pred_classes_label(data, classes_name)

  plot_classes_count(Dict_true_classes_count, Dict_pred_classes_count, width_figsize, height_figsize, save_dir = save_dir)

  # True False positive
  Dict_true_false_positive = true_false_positive(data, classes_name)
  plot_true_false_positive(Dict_true_false_positive, size = (width_figsize, height_figsize), save_dir = save_dir)

  # True positive False negative
  Dict_true_positive_false_negative = true_positive_false_negative(data, classes_name)
  plot_true_false_positive(Dict_true_positive_false_negative, size = (width_figsize, height_figsize), titles = ["True positive", "False negative"], save_dir = save_dir)

  # Precision and Recall and Confusion matrix
  calculate_confusion_matrix(data, classes_name, save_dir = save_dir)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "Evaluate Object Detection result")
  parser.add_argument("--truth_dir", help = "Directory contain Ground truth object detection YOLO", type = str)
  parser.add_argument("--pred_dir", help = "Directory contain Predict object detection", type = str)
  #parser.add_argument("--classes_name", help = "Text file save classes name", type = str)
  parser.add_argument("--width_figsize", help = "Width histogram", default = 12)
  parser.add_argument("--height_figsize", help = "Height histogram", default = 8)
  parser.add_argument("--save_dir", help = "save evaluate result", default = "save_evaluate_object_detection_result")
  args = parser.parse_args()

  truth_dir = args.truth_dir
  pred_dir = args.pred_dir
  #classes_name_path = args.classes_name   
  width_figsize = args.width_figsize   
  height_figsize = args.height_figsize   
  save_dir = args.save_dir

  main(truth_dir, pred_dir, classes_name, width_figsize, height_figsize, save_dir)
