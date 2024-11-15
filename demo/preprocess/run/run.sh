## source demo/preprocess/run/run.sh

conda deactivate 
conda deactivate 
conda activate RTE_demo
# make_kanaCSV.pyの出力の1行目を取得
output=$(python demo/preprocess/make_kanaCSV.py | head -n 1)

conda activate espnet
# 取得した1行目をB.pyに引数として渡す
python demo/preprocess/modules/tts.py "${output}.csv"
conda activate RTE_demo

