#!/bin/bash

activate

mkdir -p offline_table
cd offline_table

python ../detect_policies.py --dataset table-dummy-v1 --num_policies 6 --num_state_clusters 6 --vi_iterations 50
timeout 30m python ../td3_bc.py --dataset table-dummy-v1

gnuplot << 'EOF'
set terminal pdf size 10cm,9cm
set output "offline_table.pdf"
set size square

set obj 1 rect from 0.45,0.45 to 0.55,0.55 fc rgb "#008800"
plot [0:1] [0:1] "<(grep OPT out-$(cat best_qvalue.txt).txt)" using 2:3:($8/100):($9/100):4 with vectors palette title "Learned policy (ours)"
EOF

gnuplot << 'EOF'
set terminal pdf size 10cm,9cm
set output "td3_table.pdf"
set size square

set obj 1 rect from 0.45,0.45 to 0.55,0.55 fc rgb "#008800"
plot [0:1] [0:1] "td3_policy_latest_table-dummy-v1.txt" using 1:2:($3/50):($4/50) with vectors title "Learned policy (TD3)"
EOF

cd ..
