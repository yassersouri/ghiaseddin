for i in $(seq 0 3)
    do python ghiaseddin/scripts/train.py --dataset zappos2 --extractor vgg --attribute $i  --epochs 25
done
