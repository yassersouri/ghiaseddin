for i in $(seq 0 9)
    do python ghiaseddin/scripts/baseline.py --dataset lfw --attribute $i  --epochs 40
done
