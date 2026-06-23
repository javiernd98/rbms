python -m rbms.scripts.train_rbm -d data/neuronas_120_3_stacked.h5 --model_type BBRBM \
--num_hiddens 120 --num_updates 1000000 --gibbs_steps 10 --learning_rate 0.006 --L2 0.00001\
--batch_size 512 --num_chains 512 --train_size 0.8 --n_save 500 --filename rbms/output/RBM_Stack_3_1M_lr_0.006.h5