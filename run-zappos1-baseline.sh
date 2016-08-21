for i in $(seq 0 3)
	do
	for j in $(seq 0 9)
		do python ghiaseddin/scripts/baseline.py --dataset zappos1 --attribute $i  --epochs 25 --attribute_split $j
	done
done
