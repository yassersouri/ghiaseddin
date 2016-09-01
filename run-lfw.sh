for i in $(seq 0 9)
    do python ghiaseddin/scripts/train.py --dataset lfw --extractor vgg --attribute $i  --epochs 40
done
