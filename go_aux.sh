python3 ./core/resnet34.py train ; python3 ./core/resnet34.py test

./check_for_dataset_close_e.sh  > check_res_close_edge.log
./check_for_dataset_not_close_e.sh  > check_res_not_close_edge.log
./bin/calc_exp.py check_res_close_edge.log ".*close.*" close > calc_exp_res_close.txt
./bin/calc_exp.py check_res_not_close_edge.log ".*close.*" close > calc_exp_res_not_close.txt

./check_for_dataset_close.sh  > check_res_close.log
./check_for_dataset_not_close.sh  > check_res_not_close.log
./bin/calc_exp.py check_res_close.log ".*close.*" close > calc_exp_res_close_not_edge.txt
./bin/calc_exp.py check_res_not_close.log ".*close.*" close > calc_exp_res_not_close_not_edge.txt



