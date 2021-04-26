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

END

game='dmc_quadruped_walk'
detail='trxl'
CUDA_VISIBLE_DEVICES=0 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm False --gate plus \
    --n_layer 2 --n_head 10 --mem_len 64

game='dmc_quadruped_walk'
detail='trxl-i'
CUDA_VISIBLE_DEVICES=1 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm True --gate plus --model_lr 8e-5 \
    --n_layer 2 --n_head 10 --mem_len 64

game='dmc_quadruped_walk'
detail='trxl-gate-gru'
CUDA_VISIBLE_DEVICES=2 \
	python3 dreamer.py --logdir ./logdir/$game/transdreamer/$detail --task $game \
    --rssm_model trxl --pre_lnorm True --gate gru --model_lr 8e-5 \
    --n_layer 2 --n_head 10 --mem_len 64


