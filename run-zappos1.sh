for j in $(seq 0 9)
    do
    for i in $(seq 0 3)
        do python ghiaseddin/scripts/train.py --dataset zappos1 --attribute $i  --epochs 25 --attribute_split $j
    done
done
