for i in $(seq 0 10)
    do python ghiaseddin/scripts/small_train.py --dataset pubfig --extractor vgg --attribute $i  --epochs 5
done
