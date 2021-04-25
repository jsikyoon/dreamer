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

END

game='dmlab_rooms_watermaze'
try=1
CUDA_VISIBLE_DEVICES=0 \
    python3 dreamer.py --logdir ./logdir/$game/dreamer/$try --task $game \
    --expl epsilon_greedy --horizon 10 --kl_scale 0.1 --action_dist onehot \
    --expl_amount 0.4 --expl_min 0.1 --expl_decay 200000 --time_limit 1000000 \
    --clip_rewards tanh --action_repeat 4


