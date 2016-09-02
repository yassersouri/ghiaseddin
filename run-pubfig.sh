for i in $(seq 0 10)
    do python ghiaseddin/scripts/train.py --dataset pubfig --extractor vgg --attribute $i  --epochs 40
done
