# tweetsum pair train (bert-base) -> 일단 파일 위치 바꿔서 시도해 봄
PYTHONPATH=.. accelerate launch /home/gaya/group1/OLDS/src/main.py \
    -do_train True \
    -model bert-base-uncased \
    -log_file logs/tweetsum.bert_base.pair.train.log \
    -output_dir models/tweetsum_bert_base_pair \
    -dataset /home/gaya/group1/OLDS/data/tweetsumm/tweetsumm/omission.save \
    -max_source_length 512 \
    -per_device_train_batch_size 32 \
    -per_device_eval_batch_size 8 \
    -learning_rate 5e-5 \
    -num_train_epochs 3 \
    -weight_decay 0.01 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 1 \
    -preprocessing_num_workers 16 \
    -mode pair
# ------------------------------------------------------------------
# -> 계속 에러 남 
# -------------------------------------------------------------------
# samsum seq train
PYTHONPATH=.. accelerate launch /home/gaya/group1/OLDS/src/main.py \
    -do_train True \
    -model bert-base-uncased \
    -log_file logs/tweetsum.bert_base.seq.train.log \
    -output_dir models/tweetsum_bert_base_seq \
    -dataset /home/gaya/group1/OLDS/data/tweetsumm/tweetsumm/omission.save \
    -max_source_length 512 \
    -per_device_train_batch_size 4 \
    -per_device_eval_batch_size 8 \
    -learning_rate 5e-5 \
    -num_train_epochs 3 \
    -weight_decay 0.01 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 1 \
    -preprocessing_num_workers 64 \
    -mode seq

# samsum span train
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_train True \
    -model bert-base-uncased \
    -log_file logs/samsum.bert_base.span.train.log \
    -output_dir models/samsum_bert_base_span \
    -dataset data/samsum/omission.save \
    -max_source_length 512 \
    -per_device_train_batch_size 2 \
    -per_device_eval_batch_size 4 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.01 \
    -num_warmup_steps 500 \
    -gradient_accumulation_steps 1 \
    -preprocessing_num_workers 16 \
    -mode span

# samsum pair train
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_train True \
    -model roberta-base \
    -log_file logs/samsum.roberta_base.pair.train.log \
    -output_dir models/samsum_roberta_base_pair \
    -dataset data/samsum/omission.save \
    -max_source_length 512 \
    -per_device_train_batch_size 32 \
    -per_device_eval_batch_size 8 \
    -learning_rate 5e-5 \
    -num_train_epochs 3 \
    -weight_decay 0.01 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 1 \
    -preprocessing_num_workers 16 \
    -mode pair

# samsum seq train
PYTHONPATH=.. accelerate launch /home/gaya/group1/OLDS/src/main.py \
    -do_train True \
    -model roberta-base \
    -log_file logs/tweetsum.roberta_base.seq.train.log \
    -output_dir models/tweetsum_roberta_base_seq \
    -dataset   /home/gaya/group1/OLDS/data/tweetsumm/tweetsumm/omission.save  \
    -max_source_length 512 \
    -per_device_train_batch_size 4 \
    -per_device_eval_batch_size 4 \
    -learning_rate 5e-5 \
    -num_train_epochs 3 \
    -weight_decay 0.01 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 1 \
    -preprocessing_num_workers 64 \
    -mode seq

# samsum span train
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_train True \
    -model roberta-base \
    -log_file logs/samsum.roberta_base.span.train.log \
    -output_dir models/samsum_roberta_base_span \
    -dataset data/samsum/omission.save \
    -max_source_length 512 \
    -per_device_train_batch_size 2 \
    -per_device_eval_batch_size 4 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.01 \
    -num_warmup_steps 500 \
    -gradient_accumulation_steps 1 \
    -preprocessing_num_workers 16 \
    -mode span
