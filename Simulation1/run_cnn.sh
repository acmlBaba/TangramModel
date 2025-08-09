#!/bin/sh

source ////home/acml/.pyenv/versions/anaconda3-5.1.0/etc/profile.d/conda.sh

# ループ回数を指定
loop_count=20
trial_count=10

for ((i=17; i<=loop_count; i++))
do
    cp '../cnn2.h5' './cnn2sender.h5'
    cp '../cnn2.h5' './cnn2receiver.h5'    
    for ((j=1; j<=trial_count; j++))
    do
        if [ $j != 1 ]; then
            conda activate keras
            python relearn.py  $i $j
        fi
        
        conda activate keras
        python Extraction2.py $i $j

        conda activate keras
        python img2img.py  $i $j

        conda activate keras
        python caption_generation.py  $i $j
    
        conda activate base
        python3 img_generation.py $i $j

        conda activate keras
        python matching.py  $i $j
    
        conda activate base
        python3 confirmation.py  $i $j
    done
done
