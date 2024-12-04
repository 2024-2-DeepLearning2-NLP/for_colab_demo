PYTHONPATH=. python /home/gaya/group1/OLDS/src/main.py -do_process True -dataset samsum -save_path data/samsum_omission -preprocessing_num_workers 32
PYTHONPATH=. python ./src/main.py -do_process True -dataset data/dialogsum/dialogsum -save_path data/dialogsum_omission -preprocessing_num_workers 32
PYTHONPATH=. python /home/gaya/group1/OLDS/src/main.py -do_process True -dataset data/qmsum -save_path data/qmsum_omission -preprocessing_num_workers 32
PYTHONPATH=. python /home/gaya/group1/OLDS/src/main.py -do_process True -dataset data/tweetsumm/tweetsumm -save_path data/tweetsumm_omission -preprocessing_num_workers 32
PYTHONPATH=. python ./src/main.py -do_process True -dataset data/emailsum/emailsum -save_path data/emailsum_omission -preprocessing_num_workers 32