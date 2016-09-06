for i in $(seq 0 5)
    do python ghiaseddin/scripts/small_train.py --dataset osr --extractor vgg --attribute $i  --epochs 2
done
