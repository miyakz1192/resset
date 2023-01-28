#!/usr/bin/env python3

#一つのレコードが"(class, 確信度(float))"　形式のリストを分析する
#例：
#(1002, 0.8584231734275818)
#(1009, 0.5545655488967896)
#(200 , 0.5545655488967896)
#class = 1000以上のものを一つのclassとみなして分類していく(gathering_class_thanで指定)
#TODO: 幅指定や複数指定など
#現状の利用範囲では1000以上をcloseとしてみなしたいので、とりあえずは、幅なし、単数指定でOK

import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("file_name", type=str)
parser.add_argument("--gathering_class_than", type=int, default=1000)
parser.add_argument("--gathering_class_as", type=int, default=1000)
parser.add_argument("--calc_target", type=int, default=1000)
args = parser.parse_args()


records = []

with open(args.file_name) as f:
	raw = f.read().splitlines()
	temp = [x for x in raw if x.startswith("(")] 

for x in temp:
	x = x.strip("()").split(",")
	records.append((int(x[0]), float(x[1])))

def summer(threshold, target, invert_ratio=False):
	sum_res = Counter()
	total = len(records)
	for x in records:
		if x[1] < threshold: continue
		if x[0] >= args.gathering_class_than:	
			sum_res[args.gathering_class_than] += 1
		else:
			sum_res[x[0]] += 1
	
	for cls, score in sum_res.most_common():
		if cls == target:
			if invert_ratio is False:
				print("%f, %d, %d, %d" % (threshold, cls, score, int(score/float(total)*100.0)))
			else:
				print("%f, %d, %d, %d" % (threshold, cls, score, int((total-score)/float(total)*100.0)))
				
	
print("INFO: gathering class than %d as %d" % (args.gathering_class_than, args.gathering_class_as))

print("=====RECORD INFO=====")
print("total = %d" % (len(records)))
print("=====SUM=====")
threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for threshold in threshold_list:
	summer(threshold, args.calc_target)
print("=====SUM(INVERT RAITIO)=====")
for threshold in threshold_list:
	summer(threshold, args.calc_target, invert_ratio=True)

