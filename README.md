
## Purpose

Simulate a depth anything using the M5Stack Module-LLM(ax630c) simulator pulsar2 run.<br>
pulsar2 is a software environment released by axera-tech.<br>

https://github.com/AXERA-TECH/pulsar2-docs-en<br>
https://github.com/nnn112358/M5_LLM_Module_Report/blob/main/anysome_Module_LLM.md<br>
## How to

Enter Docker in pulsar2

```
# sudo docker run -it --net host --rm -v $PWD:/data pulsar2:3.3
```
```
# pip install -r requirements.txt
```


To run a simulation with a quantized axmodel
```
# python pulsar2_run_preprocess_axsim.py

# pulsar2 run --model model/yolo11n_640x640_base.axmodel --input_dir sim_inputs --output_dir sim_outputs --list list.txt

# python pulsar2_run_postprosess_step1.py \
  --model model/yolo11n_640x640_base.axmodel \
  --output-dir ./sim_outputs/0 \
  --num-outputs 3 \
  --bin1 ./sim_outputs/0/_model_23_Concat_output_0.bin \
  --shape1 1 80 80 144 \
  --bin2 ./sim_outputs/0/_model_23_Concat_1_output_0.bin \
  --shape2 1 40 40 144 \
  --bin3 ./sim_outputs/0/_model_23_Concat_2_output_0.bin \
  --shape3 1 20 20 144
```

```
# python pulsar2_run_postprosess_step2.py

Detected: person: 0.74 at bbox [56.7180984740844, 112.75218599172513, 348.06386195841696, 303.03316560825033]
Detected: person: 0.74 at bbox [490.03135985927656, 114.09351218416704, 288.4036620034385, 417.0365206730969]
Detected: person: 0.70 at bbox [5.157686893344362, 72.41141470651273, 255.76884955678906, 291.09160648293977]
Detected: person: 0.55 at bbox [358.5937908478081, 70.39745047553937, 219.00944176624762, 456.35924720133494]
Detected: cup: 0.77 at bbox [265.5600729811175, 367.5716531209058, 49.73386666003307, 67.31903508739606]
Detected: cup: 0.60 at bbox [55.12659170684742, 417.19051503329223, 66.95872977140311, 89.81017130911141]
Detected: cup: 0.60 at bbox [206.73806979040023, 331.7994628377665, 39.5168295215268, 56.61743976066856]
Detected: cup: 0.55 at bbox [139.46003231610405, 301.33651476077085, 34.981523982993394, 45.21633185525121]
Detected: cup: 0.50 at bbox [87.9819399768678, 287.6912431998754, 31.014739771975286, 35.3502619795392]
Detected: bowl: 0.84 at bbox [1.0042724562208605, 403.4177083459919, 50.98300059557005, 51.009406323562075]
Detected: bowl: 0.81 at bbox [103.8810495758662, 369.9760869148585, 115.12678531241363, 45.13749664559498]
Detected: dining table: 0.70 at bbox [0.2316432501811505, 298.77209693371475, 424.13916019424505, 233.22790306628525]
```

To perform a simulation with onnx before quantization

```
# python pulsar2_run_preprocess_onnx.py
 pulsar2 run --model model/yolo11n_640x640_base.onnx --input_dir sim_inputs --output_dir sim_outputs --list list.txt


# python pulsar2_run_postprosess_step1.py \
  --model model/yolo11n_640x640_base.axmodel \
  --output-dir ./sim_outputs/0 \
  --num-outputs 3 \
  --bin1 ./sim_outputs/0/_model_23_Concat_output_0.bin \
  --shape1 1 144 80 80 \
  --bin2 ./sim_outputs/0/_model_23_Concat_1_output_0.bin \
  --shape2 1 144 40 40 \
  --bin3 ./sim_outputs/0/_model_23_Concat_2_output_0.bin \
  --shape3 1 144 20 20```

 python pulsar2_run_postprosess_step2.py 


root@Thinkpad-T14:/data# python pulsar2_run_postprosess_step2_onnx.py
===========================

Detected: person: 0.74115 at bbox [30.31488, 111.136, 377.68982, 305.41906]
Detected: person: 0.71921 at bbox [4.30747, 71.6025, 258.42159, 271.95329]
Detected: person: 0.64091 at bbox [348.71344, 112.3502, 429.20841, 419.6498]
Detected: person: 0.57399 at bbox [363.86258, 74.38624, 209.79074, 388.00895]
Detected: cup: 0.81572 at bbox [265.7564, 367.51141, 49.19618, 67.53213]
Detected: cup: 0.69568 at bbox [55.33813, 417.20182, 66.58464, 89.85252]
Detected: cup: 0.63931 at bbox [88.26941, 287.61952, 30.47482, 35.70231]
Detected: cup: 0.61472 at bbox [206.14111, 332.40309, 39.61181, 56.09976]
Detected: cup: 0.57656 at bbox [139.65174, 301.51069, 34.82096, 45.09185]
Detected: bowl: 0.84523 at bbox [103.54893, 369.82272, 114.96173, 45.48041]
Detected: bowl: 0.83803 at bbox [0.17939, 402.96795, 51.86171, 51.8919]
Detected: dining table: 0.72058 at bbox [0.4651, 297.65119, 424.11724, 234.34881]
===========================



## Result

![image](https://github.com/user-attachments/assets/72efdf9c-7c70-44a3-9615-3248f091be30)


## pulsar2 document 

```
# pulsar2 run -h
usage: main.py run [-h] [--config] [--model] [--input_dir] [--output_dir] [--list] [--random_input ]
                   [--batch_size] [--enable_perlayer_output ] [--dump_with_stride ] [--group_index] [--mode]
                   [--target_hardware]

optional arguments:
  -h, --help            show this help message and exit
  --config              config file path, supported formats: json / yaml / toml / prototxt. type: string.
                        required: false. default:.
  --model               run model path, support ONNX, QuantAxModel and CompiledAxmodel. type: string. required:
                        true.
  --input_dir           model input data in this directory. type: string. required: true. default:.
  --output_dir          model output data directory. type: string. required: true. default:.
  --list                list file path. type: string. required: true. default:.
  --random_input []     random input data. type: bool. required: false. default: false.
  --batch_size          batch size to be used in dynamic inference mode, only work for CompiledAxModel. type: int.
                        required: false. defalult: 0.
  --enable_perlayer_output []
                        enable dump perlayer output. type: bool. required: false. default: false.
  --dump_with_stride []
  --group_index
  --mode                run mode, only work for QuantAxModel. type: enum. required: false. default: Reference.
                        option: Reference, NPUBackend.
  --target_hardware     target hardware, only work for QuantAxModel. type: enum. required: false. default: AX650.
                        option: AX650, AX620E, M76H.

optionalパラメータ:
- `--model`: 実行するモデルのパス（ONNX、QuantAxModel、CompiledAxmodelをサポート）
- `--input_dir`: モデルの入力データのディレクトリ
- `--output_dir`: モデル出力データの保存先ディレクトリ
- `--list`: リストファイルのパス

オプションパラメータ:
- `--config`: 設定ファイルのパス（json/yaml/toml/prototxtをサポート）
- `--random_input`: ランダムな入力データを使用（デフォルト: false）
- `--batch_size`: 動的推論モードで使用するバッチサイズ（CompiledAxModelのみ有効）
- `--enable_perlayer_output`: レイヤーごとの出力ダンプを有効化（デフォルト: false）
- `--mode`: 実行モード（QuantAxModelのみ有効）
  - オプション: Reference、NPUBackend
  - デフォルト: Reference
- `--target_hardware`: ターゲットハードウェア（QuantAxModelのみ有効）
  - オプション: AX650、AX620E、M76H
  - デフォルト: AX650

```



