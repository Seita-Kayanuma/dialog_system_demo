# ResponseTimingEstimator_demo
## シナリオ
```
scinario/data_kana-kanji.csv
```

## 音声
### tts
espnetを用いて作成した音声

### user
macbookで録音した音声

## モデル
### silentモデル
発話末から一定の無音区間で応答
```
python silent/demo_system.py
```
### 発話タイミング推定モデル
発話/非発話を推定し、初めて発話と推定された時応答
```
python　timing/demo_system2.py
```