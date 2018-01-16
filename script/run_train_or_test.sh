#!/usr/bin/bash

dat_dir="/home/jangwee/workspace/wp_py/GG-modularization/offline_process/merged_output"
train_file=${dat_dir}"/train.dat"
test_file=${dat_dir}"/test.dat"

clf="EnsembleClassifier.py"

function train()
{
    cat $train_file | python $clf train > result/train.log 2>&1 &
}

function test()
{
    cat $test_file | python $clf test > result/test.log 2>&1 &
}

#train
test


