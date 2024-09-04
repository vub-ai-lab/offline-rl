#!/bin/bash

activate

mkdir -p offline_brayton
cd offline_brayton

python ../detect_policies.py --dataset ../brayton_dataset.mat --num_policies 10 --num_state_clusters 6 --vi_iterations 50
timeout 30m python ../td3_bc.py --dataset ../brayton_dataset.mat

gnuplot << 'EOF'
set terminal pdf size 10cm,9cm
set output "offline_brayton_P.pdf"
set size square

set xlabel "Speed (x1K RPM)"
set ylabel "Temp (x1K Kelvin)"

plot [3:12] [0.2:2] "<(grep OPT out-$(cat best_qvalue.txt).txt)" using 2:3:7 with points palette title "Learned policy (P)"
EOF

gnuplot << 'EOF'
set terminal pdf size 10cm,9cm
set output "offline_brayton_I.pdf"
set size square

set xlabel "Speed (x1K RPM)"
set ylabel "Temp (x1K Kelvin)"

plot [3:12] [0.2:2] "<(grep OPT out-$(cat best_qvalue.txt).txt)" using 2:3:8 with points palette title "Learned policy (I)"
EOF

gnuplot << 'EOF'
set terminal pdf size 10cm,9cm
set output "offline_brayton_states.pdf"
set size square

set xlabel "Speed (x1K RPM)"
set ylabel "Temp (x1K Kelvin)"

plot [3:12] [0.2:2] "<(grep OPT out-$(cat best_qvalue.txt).txt)" using 2:3:5 with points palette title "State partitioning"
EOF

gnuplot << 'EOF'
set terminal pdf size 10cm,9cm
set output "offline_brayton_sequences.pdf"
set size square

set xlabel "Speed (x1K RPM)"
set ylabel "Temp (x1K Kelvin)"

plot [3:12] [0.2:2] "<(grep S out-$(cat best_qvalue.txt).txt)" using 2:3:0 with points palette title "Index"
EOF

gnuplot << 'EOF'
set terminal pdf size 10cm,9cm
set output "offline_brayton_TD3_P.pdf"
set size square

set xlabel "Speed (x1K RPM)"
set ylabel "Temp (x1K Kelvin)"

plot [3:12] [0.2:2] "td3_policy_latest_brayton_dataset.mat.txt" using ($1*2.256+7.09):($2*0.376+0.901):4 with points palette title "TD3 Learned policy (I)"
EOF

gnuplot << 'EOF'
set terminal pdf size 10cm,9cm
set output "offline_brayton_TD3_I.pdf"
set size square

set xlabel "Speed (x1K RPM)"
set ylabel "Temp (x1K Kelvin)"

plot [3:12] [0.2:2] "td3_policy_latest_brayton_dataset.mat.txt" using ($1*2.256+7.09):($2*0.376+0.901):5 with points palette title "TD3 Learned policy (I)"
EOF

cd ..
