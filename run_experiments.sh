#!/bin/bash
set -e
mkdir -p logs
PY=${PY:-python}

run() {
    name=$1
    shift
    echo "=== $name"
    $PY -u train.py --name "$name" "$@" > "logs/$name.log" 2>&1
    grep -E "^(TEST|early)" "logs/$name.log"
}

run cnn_noaug --model cnn --aug none
run cnn_aug --model cnn --aug basic
run cnn_aug_strong --model cnn --aug strong

AUG=$($PY -c "
import json
r = {a: json.load(open(f'results/runs/{n}/metrics.json'))['val']['f1_macro']
     for a, n in [('none', 'cnn_noaug'), ('basic', 'cnn_aug'), ('strong', 'cnn_aug_strong')]}
print(max(r, key=r.get))")
echo "AUG=$AUG"

for mode in fc layer4 layer3_4 full; do
    run resnet_$mode --model resnet18 --mode $mode --aug $AUG --bs 32 --opt adam --lr 1e-3 --lr_backbone 1e-4
done

run cnn_leaky_relu --model cnn --aug $AUG --act leaky_relu
run cnn_gelu --model cnn --aug $AUG --act gelu
run cnn_bs16 --model cnn --aug $AUG --bs 16
run cnn_bs64 --model cnn --aug $AUG --bs 64
run cnn_lr3e-4 --model cnn --aug $AUG --lr 3e-4
run cnn_lr3e-3 --model cnn --aug $AUG --lr 3e-3
run cnn_adamw --model cnn --aug $AUG --opt adamw --wd 1e-2
run cnn_sgd --model cnn --aug $AUG --opt sgd --lr 1e-2 --wd 5e-4
run cnn_drop0.2 --model cnn --aug $AUG --dropout 0.2
run cnn_drop0.7 --model cnn --aug $AUG --dropout 0.7
run cnn_5blocks --model cnn --aug $AUG --channels 32,64,128,256,512
run cnn_best --model cnn --aug $AUG --channels 32,64,128,256,512 --act gelu --dropout 0.2

for mode in fc layer4 layer3_4 full; do
    run resnet_${mode}_aug --model resnet18 --mode $mode --aug basic --bs 32 --opt adam --lr 1e-3 --lr_backbone 1e-4
done

run cnn_aug_flip --model cnn --aug flip

$PY -u train.py --name cnn_5blocks_s1_patience50 --model cnn --aug none --channels 32,64,128,256,512 --seed 1 --patience 50 --out results/extra
$PY -u train.py --name cnn_5blocks_s2_patience50 --model cnn --aug none --channels 32,64,128,256,512 --seed 2 --patience 50 --out results/extra
