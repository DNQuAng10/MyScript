import argparse
import cv2 as cv 
import os

from classes_name import classes_name

# --------------------------------------------------------------------
# classes_bbox_form_text = 'classes xmin([0, 1]) ymin([0, 1]) xmax([0, 1]) ymax([0, 1])
# Unnormalize bbox 
# new_classes_bbox_form_text  = 'classes xmin([0, width_img]) ymin([0, height_img]) xmax([0, width_img]) ymax([0, height_img])'
def Unnormalize_bbox(classes_bbox, width, height, classes_name):
  # bbox: x_center, y_center, width, height relative to width and height of img
  # xmin = x_center*width_img - width*width_img/2
  # ymin = y_center*height_img - height*height_img/2
  # xmax = xmin + width*width_img
  # ymax = ymin + height*height_img
  x_center, y_center, w_rate, h_rate = float(classes_bbox[1]), float(classes_bbox[2]), float(classes_bbox[3]), float(classes_bbox[4])
  #print("Bbox:", left, top, right, bottom)
  #print(width, height)
  xmin, ymin = round(x_center*width - w_rate*width/2), round(y_center*height - h_rate*height/2) 
  xmax, ymax = round(xmin + w_rate*width), round(ymin + h_rate*height)
  #print("Unnormalized bbox:", left, top, right, bottom)
  
  new_classes_bbox = [classes_name[classes_bbox[0]], xmin, ymin, xmax, ymax]
  return new_classes_bbox

def List2String(List):
  string = List[0]
  for i in range(1, len(List)):
    string += " " + str(List[i])
  return string

def change_bbox2unnormalize_bbox(txt_file_path, width, height, classes_name):
  with open(txt_file_path, "r") as file:
    classes_bbox = file.read().split()
  print("Old classes bbox:", classes_bbox)

  non_classes_bbox = None
  new_classes_bbox_str = ""
  if len(classes_bbox) != 0:
    if len(classes_bbox) == 5:
      new_classes_bbox = Unnormalize_bbox(classes_bbox, width, height, classes_name)
      new_classes_bbox_str = List2String(new_classes_bbox)
    else:
      #print(txt_file_path)
      x = len(classes_bbox)//5
      classes_bbox_1 = []
      for i in range(x):
        classes_bbox_1.append(classes_bbox[i*5:(i*5 + 5)])
      #print(classes_bbox_1)
      
      new_classes_bbox_0 = Unnormalize_bbox(classes_bbox_1[0], width, height, classes_name)
      new_classes_bbox_str = List2String(new_classes_bbox_0)
      for i in range(1, len(classes_bbox_1)):
        new_classes_bbox = Unnormalize_bbox(classes_bbox_1[i], width, height, classes_name)
        new_classes_bbox_str += "\n" + List2String(new_classes_bbox)
  else:
    non_classes_bbox = txt_file_path
  print("New classes unnormalized bbox:", new_classes_bbox_str)
  return new_classes_bbox_str, non_classes_bbox
  
def save_new_txt(new_classes_bbox_str, txt_file_path, is_new_save = False, save_dir = None):
  if is_new_save:
    if os.path.exists(save_dir) == False:
      os.mkdir(save_dir)
    
    txt_new_path = os.path.join(save_dir, os.path.basename(txt_file_path))
    with open(txt_new_path, "w") as file:
      file.write(new_classes_bbox_str)
  else:
    txt_new_path = os.path.join(save_dir, os.path.basename(txt_file_path))
    with open(txt_new_path, "w") as file:
      file.write(new_classes_bbox_str)

# -------------------------------------------------------------
# Save new unnormalize bbox 
def save_unnormalize_bbox(dir, classes_name, is_new_save = True, save_dir = None):
  List_img_path = []
  List_txt_path = []
  for path, subdir, files in os.walk(dir):
    for file in files:
      if file.endswith(".jpg") == True or file.endswith(".jpeg") == True:
        img_path = os.path.join(path, file)
        List_img_path.append(img_path)
      elif file.endswith(".txt") == True:
        txt_path = os.path.join(path, file)
        List_txt_path.append(txt_path)
  
  List_non_classes_bbox = []
  for i in List_txt_path:
    img_path = i[:-4] + ".jpg"
    if img_path not in List_img_path:
      img_path = i[:-4] + ".jpeg"
    
    print(img_path)

    img = cv.imread(img_path)
    w, h = img.shape[1], img.shape[0]

    new_classes_bbox_str, non_classes_bbox = change_bbox2unnormalize_bbox(i, w, h, classes_name)
    save_new_txt(new_classes_bbox_str, i, is_new_save, save_dir)
    List_non_classes_bbox.append(non_classes_bbox)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "Unnormalize bounding box and convert digit class to class name in YOLO ground-truth text file")
  parser.add_argument("--dir", help = "directory contain ground truth file", type = str)
  parser.add_argument("--save_dir", help = "save new classes bbox", type = str)

  args = parser.parse_args()
  dir = args.dir
  save_dir = args.save_dir

  save_unnormalize_bbox(dir, classes_name, save_dir = save_dir)
  

