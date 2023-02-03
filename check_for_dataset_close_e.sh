ls test_data/dataset_20230125/close/* | while read line
do
	echo ${line}
	python3 /home/a/pytorch_ssd/bin/edged.py ${line}
	python3 core/resnet34.py single ./edged.jpg
done
