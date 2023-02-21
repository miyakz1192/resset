#!/usr/bin/env python3

#一つのレコードが"(class, 確信度(float))"　形式のリストを分析する
#例：
#(1002, 0.8584231734275818)
#(1009, 0.5545655488967896)
#(200 , 0.5545655488967896)
#class = 1000以上のものを一つのclassとみなして分類していく(gathering_class_thanで指定)
#TODO: 幅指定や複数指定など
#現状の利用範囲では1000以上をcloseとしてみなしたいので、とりあえずは、幅なし、単数指定でOK

#仕様改善後
#calc_targetオプションで集計する対象となるラベルを正規表現で指定する。exact matchも可能。
#calc_asオプションで集計を集約する対象のラベルをexact matchな表現で指定

import argparse
from collections import Counter

import re
import sys
sys.path.append("./dataset")
from gaa import *

parser = argparse.ArgumentParser()
parser.add_argument("file_name", type=str)
parser.add_argument("calc_target", type=str)
parser.add_argument("calc_as", type=str)
args = parser.parse_args()

records = []

class LabelDB:
	def __init__(self):
		dataset = GAADataSet()
		print("dataset size = %d" % (len(dataset)))
		print("dataset classses = %d" % (dataset.classes()))
		self.db = dataset.label_ids()

	def target_ids(self, target_label_pattern):
		res = []
		for label, _id in self.db.items():
			if re.match(target_label_pattern, label):
				print("INFO: %s,%d" % (label, _id))
				res.append(_id)

		return res


with open(args.file_name) as f:
	raw = f.read().splitlines()
	temp = [x for x in raw if x.startswith("(")] 

for x in temp:
	x = x.strip("()").split(",")
	records.append((int(x[0]), float(x[1])))

def summer(threshold, targets, invert_ratio=False):
	sum_res = Counter()
	total = len(records)
	for x in records:
		if x[1] < threshold: continue
		if x[0] in targets:
			sum_res[args.calc_as] += 1
		else:
			sum_res[x[0]] += 1
	
	for label, score in sum_res.most_common():
		if label == args.calc_as:
			if invert_ratio is False:
				print("%f, %s, %d, %d" % (threshold, label, score, int(score/float(total)*100.0)))
			else:
				print("%f, %s, %d, %d" % (threshold, label, score, int((total-score)/float(total)*100.0)))
				
label_db = LabelDB()
print("### CALC targets as label=%s,id=%d" % (args.calc_as,label_db.db[args.calc_as]))
targets = label_db.target_ids(args.calc_target)
	
print("=====RECORD INFO=====")
print("total = %d" % (len(records)))
print("=====SUM=====")
threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.87, 0.88, 0.89, 0.9, 1.0]
for threshold in threshold_list:
	summer(threshold, targets)
print("=====SUM(INVERT RAITIO)=====")
for threshold in threshold_list:
	summer(threshold, targets, invert_ratio=True)

