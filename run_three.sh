room='ThreeLargeRooms'
iters=30000
seeds=10
tr='optimistic'
dr=0.2
dkd=0.9
srcenv='FourSmallRooms_11'
eps=0.7
temperature=8.79
exp='epsilon-trex'

echo "Running Epsilon-Greedy"
python q_learning.py --env-name "$room" --num-iters "$iters" --num-seeds "$seeds"
echo "Running Softmax"
python q_learning_softmax.py --env-name "$room" --num-iters "$iters" --num-seeds "$seeds"
echo "Running Pursuit"
python q_learning_pursuit.py --env-name "$room" --num-iters "$iters" --num-seeds "$seeds"
echo "Running MBIE-EB"
python q_learning_count_based.py --env-name "$room" --num-iters "$iters" --num-seeds "$seeds"
echo "Running TReX"
python run_trex_curr.py --num-seeds "$seeds" --src-env "$srcenv" --tgt-env "$room" --transfer "$tr" --discount-r "$dr" --discount-kd "$dkd" --env-name "$room" --num-iters "$iters" --epsilon "$eps" --temp "$temperature" --explore "$exp"