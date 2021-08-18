# history
:<<"END"
game='dmc_quadruped_walk'
try=1
CUDA_VISIBLE_DEVICES=0 \
	python3 dreamer.py --logdir ./logdir/$game/dreamer/$try --task $game

game='dmc_hopper_hop'
try=1
CUDA_VISIBLE_DEVICES=2 \
	python3 dreamer.py --logdir ./logdir/$game/dreamer/$try --task $game

game='dmlab_explore_object_rewards_few'
try=1
CUDA_VISIBLE_DEVICES=0 \
    python3 dreamer.py --logdir ./logdir/$game/dreamer/$try --task $game \
    --expl epsilon_greedy --horizon 10 --kl_scale 0.1 --action_dist onehot \
    --expl_amount 0.4 --expl_min 0.1 --expl_decay 200000 --time_limit 1000000 \
    --clip_rewards tanh --action_repeat 4

game='dmlab_rooms_watermaze'
try=1
CUDA_VISIBLE_DEVICES=0 \
    python3 dreamer.py --logdir ./logdir/$game/dreamer/$try --task $game \
    --expl epsilon_greedy --horizon 10 --kl_scale 0.1 --action_dist onehot \
    --expl_amount 0.4 --expl_min 0.1 --expl_decay 200000 --time_limit 1000000 \
    --clip_rewards tanh --action_repeat 4

game='dmc_quadruped_walk'
try=2
CUDA_VISIBLE_DEVICES=5 \
	python3 dreamer.py --logdir ./logdir/$game/dreamer/$try --task $game \
    --precision 32

game='dmlab_explore_object_rewards_few'
try=2
CUDA_VISIBLE_DEVICES=0 \
    python3 dreamer.py --logdir ./logdir/$game/dreamer/$try --task $game \
    --expl epsilon_greedy --horizon 10 --kl_scale 0.1 --action_dist onehot \
    --expl_amount 0.4 --expl_min 0.1 --expl_decay 200000 --time_limit 1000000 \
    --clip_rewards tanh --action_repeat 4 \
    --precision 32

game='dmlab_rooms_watermaze'
try=2
CUDA_VISIBLE_DEVICES=0 \
    python3 dreamer.py --logdir ./logdir/$game/dreamer/$try --task $game \
    --expl epsilon_greedy --horizon 10 --kl_scale 0.1 --action_dist onehot \
    --expl_amount 0.4 --expl_min 0.1 --expl_decay 200000 --time_limit 1000000 \
    --clip_rewards tanh --action_repeat 4 \
    --precision 32

game='dmc_quadruped_walk'
try=3
CUDA_VISIBLE_DEVICES=0 \
	python3 dreamer.py --logdir ./logdir/$game/dreamer/$try --task $game \
    --precision 32

game='dmc_quadruped_walk'
detail='trxl_prefill5000_pretrain100'
CUDA_VISIBLE_DEVICES=1 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm False --gate plus \
    --n_layer 2 --n_head 10 --mem_len 4  \
    --precision 32


game='dmc_quadruped_walk'
detail='trxl_prefill1e5_pretrain1e4'
CUDA_VISIBLE_DEVICES=2 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm False --gate plus \
    --n_layer 2 --n_head 10 --mem_len 4  \
    --prefill 100000 --pretrain 10000 \
    --precision 32

game='dmc_quadruped_walk'
detail='trxl_prefill1e4_pretrain1e4_bs100_bl5'
CUDA_VISIBLE_DEVICES=3 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm False --gate plus \
    --n_layer 2 --n_head 10 --mem_len 2 \
    --prefill 10000 --pretrain 10000 --batch_size 100 --batch_length 5\
    --precision 32


game='dmc_quadruped_walk'
detail='trxl_prefill5000_pretrain100_bs50_bl5'
CUDA_VISIBLE_DEVICES=5 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm False --gate plus \
    --n_layer 2 --n_head 10 --mem_len 2 \
    --prefill 5000 --pretrain 100 --batch_size 50 --batch_length 5\
    --precision 32


game='dmc_quadruped_walk'
detail='trxl_prefill5000_pretrain1e4'
CUDA_VISIBLE_DEVICES=4 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm False --gate plus \
    --n_layer 2 --n_head 10 --mem_len 4  \
    --pretrain 10000 \
    --precision 32

game='dmc_quadruped_walk'
detail='trxl_prefill5000_pretrain1e5'
CUDA_VISIBLE_DEVICES=5 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm False --gate plus \
    --n_layer 2 --n_head 10 --mem_len 4  \
    --pretrain 100000 \
    --precision 32

END


game='dmc_quadruped_walk'
detail='trxl_prefill5000_pretrain100_bs50_bl5_mlr6e-5'
CUDA_VISIBLE_DEVICES=2 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm False --gate plus \
    --n_layer 2 --n_head 10 --mem_len 2 --model_lr 6e-5 \
    --prefill 5000 --pretrain 100 --batch_size 50 --batch_length 5\
    --precision 32


game='dmc_quadruped_walk'
detail='trxl_prefill5000_pretrain100_bs50_bl5_mlr6e-6'
CUDA_VISIBLE_DEVICES=3 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm False --gate plus \
    --n_layer 2 --n_head 10 --mem_len 2 --model_lr 6e-6 \
    --prefill 5000 --pretrain 100 --batch_size 50 --batch_length 5\
    --precision 32


game='dmc_quadruped_walk'
detail='trxli_prefill5000_pretrain100_bs50_bl5_mlr6e-5'
CUDA_VISIBLE_DEVICES=5 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm True --gate plus \
    --n_layer 2 --n_head 10 --mem_len 2 --model_lr 6e-5 \
    --prefill 5000 --pretrain 100 --batch_size 50 --batch_length 5\
    --precision 32


game='dmc_quadruped_walk'
detail='trxli_prefill5000_pretrain300_bs50_bl5_mlr6e-4_train300'
CUDA_VISIBLE_DEVICES=1 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm True --gate plus \
    --n_layer 1 --n_head 10 --mem_len 4 --model_lr 6e-4 --train_steps 300 \
    --prefill 5000 --pretrain 300 --batch_size 50 --batch_length 5 \
    --precision 32


game='dmc_quadruped_walk'
detail='gtrxl_prefill5000_pretrain300_bs50_bl5_mlr6e-4_train300'
CUDA_VISIBLE_DEVICES=2 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm True --gate gru \
    --n_layer 1 --n_head 10 --mem_len 4 --model_lr 6e-4 --train_steps 300 \
    --prefill 5000 --pretrain 300 --batch_size 50 --batch_length 5 \
    --precision 32


game='dmc_quadruped_walk'
detail='trxli_prefill5000_pretrain300_bs50_bl5_mlr6e-4_alr8e-4_vlr8e-5_train300'
CUDA_VISIBLE_DEVICES=1 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm True --gate plus \
    --n_layer 1 --n_head 10 --mem_len 4 --model_lr 6e-4 --train_steps 300 --actor_lr 8e-4 \
    --prefill 5000 --pretrain 300 --batch_size 50 --batch_length 5 \
    --precision 32

game='dmc_quadruped_walk'
detail='trxli_prefill5000_pretrain300_bs50_bl5_mlr6e-4_alr8e-6_vlr8e-6_train300'
CUDA_VISIBLE_DEVICES=4 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm True --gate plus \
    --n_layer 1 --n_head 10 --mem_len 4 --model_lr 6e-4 --train_steps 300 --actor_lr 8e-6 --value_lr 8e-6 \
    --prefill 5000 --pretrain 300 --batch_size 50 --batch_length 5 \
    --precision 32


game='dmc_quadruped_walk'
try='prec32_tr300'
CUDA_VISIBLE_DEVICES=1 \
	python3 dreamer.py --logdir ./logdir/$game/dreamer/$try --task $game \
    --precision 32 --train_steps 300

game='dmc_quadruped_walk'
detail='prec32_prefill5000_pretrain100_bs50_bl50_mlr6e-4_alr8e-5_vlr8e-5_train100'
CUDA_VISIBLE_DEVICES=2 \
	python3 dreamer.py --logdir ./logdir/$game/dreamer/$detail --task $game \
    --rssm_model gru --model_lr 6e-4 --train_steps 100 --actor_lr 8e-5 --value_lr 8e-5 \
    --prefill 5000 --pretrain 100 --batch_size 50 --batch_length 50 \
    --precision 32


game='dmc_quadruped_walk'
detail='trxli_prefill5000_pretrain100_bs50_bl50_mlr6e-5_alr8e-5_vlr8e-5_train100'
CUDA_VISIBLE_DEVICES=1 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm True --gate plus \
    --n_layer 1 --n_head 10 --mem_len 4 --model_lr 6e-5 --train_steps 100 --actor_lr 8e-5 --value_lr 8e-5 \
    --prefill 5000 --pretrain 100 --batch_size 50 --batch_length 50 \
    --precision 32

game='dmc_quadruped_walk'
detail='trxli_prefill5000_pretrain100_bs50_bl50_mlr6e-5_alr8e-6_vlr8e-6_train100'
CUDA_VISIBLE_DEVICES=2 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm True --gate plus \
    --n_layer 1 --n_head 10 --mem_len 4 --model_lr 6e-5 --train_steps 100 --actor_lr 8e-6 --value_lr 8e-6 \
    --prefill 5000 --pretrain 100 --batch_size 50 --batch_length 50 \
    --precision 32 

game='dmc_quadruped_walk'
detail='trxli_prefill5000_pretrain100_bs50_bl50_mlr6e-5_alr8e-5_vlr8e-5_train100_expl0.3'
CUDA_VISIBLE_DEVICES=4 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm True --gate plus \
    --n_layer 1 --n_head 10 --mem_len 4 --model_lr 6e-5 --train_steps 100 --actor_lr 8e-5 --value_lr 8e-5 \
    --prefill 5000 --pretrain 100 --batch_size 50 --batch_length 50 \
    --precision 32


game='dmc_quadruped_walk'
detail='gtrxl_prefill5000_pretrain100_bs50_bl50_mlr6e-5_alr8e-5_vlr8e-5_train100_expl0.3'
CUDA_VISIBLE_DEVICES=3 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm True --gate gru \
    --n_layer 1 --n_head 10 --mem_len 4 --model_lr 6e-5 --train_steps 100 --actor_lr 8e-5 --value_lr 8e-5 \
    --prefill 5000 --pretrain 100 --batch_size 50 --batch_length 50 \
    --precision 32



game='dmc_quadruped_walk'
detail='prec32_prefill5000_pretrain100_bs50_bl50_mlr6e-4_alr8e-5_vlr8e-5_train100_expl0.3'
CUDA_VISIBLE_DEVICES=1 \
	python3 dreamer.py --logdir ./logdir/$game/dreamer/$detail --task $game \
    --rssm_model gru --model_lr 6e-4 --train_steps 100 --actor_lr 8e-5 --value_lr 8e-5 \
    --prefill 5000 --pretrain 100 --batch_size 50 --batch_length 50 \
    --precision 32


game='dmc_quadruped_walk'
detail='prec32_prefill5000_pretrain100_bs50_bl50_mlr6e-4_alr8e-5_vlr8e-5_train100_expl0.1'
CUDA_VISIBLE_DEVICES=2 \
	python3 dreamer.py --logdir ./logdir/$game/dreamer/$detail --task $game \
    --rssm_model gru --model_lr 6e-4 --train_steps 100 --actor_lr 8e-5 --value_lr 8e-5 \
    --prefill 5000 --pretrain 100 --batch_size 50 --batch_length 50 \
    --precision 32 --expl_amount 0.1


game='dmc_quadruped_walk'
detail='tmp'
CUDA_VISIBLE_DEVICES=2 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm True --gate gru \
    --n_layer 1 --n_head 10 --mem_len 4 --model_lr 6e-5 --train_steps 100 --actor_lr 8e-5 --value_lr 8e-5 \
    --prefill 5000 --pretrain 100 --batch_size 50 --batch_length 50 \
    --precision 32





game='dmc_quadruped_walk'
detail='trxl_prefill5000_pretrain100_bs50_bl50_mlr6e-5_alr8e-5_vlr8e-5_train100_expl0.3'
CUDA_VISIBLE_DEVICES=3 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm False --gate plus \
    --n_layer 1 --n_head 10 --mem_len 4 --model_lr 6e-5 --train_steps 100 --actor_lr 8e-5 --value_lr 8e-5 \
    --prefill 5000 --pretrain 100 --batch_size 50 --batch_length 50 \
    --precision 32

game='dmc_quadruped_walk'
detail='trxli_prefill5000_pretrain100_bs50_bl50_mlr6e-5_alr8e-5_vlr8e-5_train100_expl0.3'
CUDA_VISIBLE_DEVICES=4 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm True --gate plus \
    --n_layer 1 --n_head 10 --mem_len 4 --model_lr 6e-5 --train_steps 100 --actor_lr 8e-5 --value_lr 8e-5 \
    --prefill 5000 --pretrain 100 --batch_size 50 --batch_length 50 \
    --precision 32

game='dmc_quadruped_walk'
detail='gtrxl_prefill5000_pretrain100_bs50_bl50_mlr6e-5_alr8e-5_vlr8e-5_train100_expl0.3'
CUDA_VISIBLE_DEVICES=6 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm True --gate gru \
    --n_layer 1 --n_head 10 --mem_len 4 --model_lr 6e-5 --train_steps 100 --actor_lr 8e-5 --value_lr 8e-5 \
    --prefill 5000 --pretrain 100 --batch_size 50 --batch_length 50 \
    --precision 32



