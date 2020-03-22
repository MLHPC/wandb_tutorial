# Weights & Biases の使い方

<!-- TOC -->

- [Weights & Biases の使い方](#weights--biases-%E3%81%AE%E4%BD%BF%E3%81%84%E6%96%B9)
  - [まずは動かしてみる](#%E3%81%BE%E3%81%9A%E3%81%AF%E5%8B%95%E3%81%8B%E3%81%97%E3%81%A6%E3%81%BF%E3%82%8B)
  - [学習の監視・可視化](#%E5%AD%A6%E7%BF%92%E3%81%AE%E7%9B%A3%E8%A6%96%E3%83%BB%E5%8F%AF%E8%A6%96%E5%8C%96)
    - [最も基本的な使い方](#%E6%9C%80%E3%82%82%E5%9F%BA%E6%9C%AC%E7%9A%84%E3%81%AA%E4%BD%BF%E3%81%84%E6%96%B9)
    - [wandb.init()](#wandbinit)
    - [wandb.config](#wandbconfig)
      - [代入による設定](#%E4%BB%A3%E5%85%A5%E3%81%AB%E3%82%88%E3%82%8B%E8%A8%AD%E5%AE%9A)
      - [更新](#%E6%9B%B4%E6%96%B0)
      - [yaml から設定を読む](#yaml-%E3%81%8B%E3%82%89%E8%A8%AD%E5%AE%9A%E3%82%92%E8%AA%AD%E3%82%80)
    - [wandb.log()](#wandblog)
      - [基本](#%E5%9F%BA%E6%9C%AC)
      - [ステップを指定して記録](#%E3%82%B9%E3%83%86%E3%83%83%E3%83%97%E3%82%92%E6%8C%87%E5%AE%9A%E3%81%97%E3%81%A6%E8%A8%98%E9%8C%B2)
      - [Summary Metrics](#summary-metrics)
      - [ウェブ UI で見るプロットの x 軸を変える](#%E3%82%A6%E3%82%A7%E3%83%96-ui-%E3%81%A7%E8%A6%8B%E3%82%8B%E3%83%97%E3%83%AD%E3%83%83%E3%83%88%E3%81%AE-x-%E8%BB%B8%E3%82%92%E5%A4%89%E3%81%88%E3%82%8B)
      - [ステップ数が多くなるとサンプリングされる](#%E3%82%B9%E3%83%86%E3%83%83%E3%83%97%E6%95%B0%E3%81%8C%E5%A4%9A%E3%81%8F%E3%81%AA%E3%82%8B%E3%81%A8%E3%82%B5%E3%83%B3%E3%83%97%E3%83%AA%E3%83%B3%E3%82%B0%E3%81%95%E3%82%8C%E3%82%8B)
      - [その他記録できること（一部）](#%E3%81%9D%E3%81%AE%E4%BB%96%E8%A8%98%E9%8C%B2%E3%81%A7%E3%81%8D%E3%82%8B%E3%81%93%E3%81%A8%E4%B8%80%E9%83%A8)
  - [ログインについての詳細](#%E3%83%AD%E3%82%B0%E3%82%A4%E3%83%B3%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6%E3%81%AE%E8%A9%B3%E7%B4%B0)
  - [API 経由で W&B を使う方法](#api-%E7%B5%8C%E7%94%B1%E3%81%A7-wb-%E3%82%92%E4%BD%BF%E3%81%86%E6%96%B9%E6%B3%95)
  - [W&B Sweeps でハイパラサーチ](#wb-sweeps-%E3%81%A7%E3%83%8F%E3%82%A4%E3%83%91%E3%83%A9%E3%82%B5%E3%83%BC%E3%83%81)

<!-- /TOC -->

## まずは動かしてみる

PyTorch を使った MNIST によるサンプルコードが `./sample` 以下に同梱。

1. W&B のアカウント作る

2. プロジェクトを作成する。名前は `sample-pytorch-mnist` にする(サンプルコードの中で指定している)

3. クライアントのインストールとログインをする。ブラウザが開くので API キーをコピペする。ログインの詳細は後述。

```
pip3 install wandb
wandb login
```

4. Run 🚀

```
pip3 install -r requirements.txt
python3 sample/main.py
```

5. ウェブ UI のダッシュボードで経過を確認 🔍

## 学習の監視・可視化

### 最も基本的な使い方

`wandb.log({ 'loss': 0.2 })` などすると、リアルタイムで記録が送信され、ウェブ UI で確認できる。
また、記録に関する情報が `./wandb` ディレクトリに諸々が保存されていく。

```python
import wandb

default_hyperparams = {
    'some_hyperparam1': val1,
    'some_hyperparam2': val2
}

wandb.init(
  config=default_hyperparams,
  project="project-name",
  name="name-of-this-run"
)

# ...some ML code

wandb.log({ 'loss': loss })
```

### wandb.init()

よく使いそうな引数は以下の通り

- project: プロジェクトの名前(str)
- name: 実行(run と呼ばれる)ごとに名前をつけられる。name とは別にユニークな ID が割り振られるので重複してもよい。与えなかった場合適当な名前が自動で割り振られる(str)
- notes: その実行に関する備考などを書いておくと、ウェブ UI に表示される(str)
- config: その実行に関する設定。これもウェブ UI に表示される。ハイパーパラメータなどを記録しておくと良い(dict-like)
- id: 自分で ID を指定することもできる。(str)

その他は以下に  
https://docs.wandb.com/library/init

### wandb.config

`wantdb.init()` の `config` 引数に渡す以外にも `wandb.config` でも設定できる。

#### 代入による設定

```python
wandb.config.epochs = 4
wandb.config.batch_size = 32
```

#### 更新

```python
wandb.config.update({"epochs": 8, "batch_size": 64})
```

#### yaml から設定を読む

デフォルトでは `config-defaults.yaml` に書いておくと wandb が勝手に読んでくれる。

```yaml
epochs:
  desc: Number of epochs to train over
  value: 100
batch_size:
  desc: Size of each mini-batch
  value: 32
```

dict を使った設定との共存もできる。

```python
hyperparameter_defaults = dict(
    dropout = 0.5,
    batch_size = 100,
    learning_rate = 0.001,
)

config_dictionary = dict(
    yaml=my_yaml_file,
    params=hyperparameter_defaults,
)

wandb.init(config=config_dictionary)
```

コマンドライン引数から渡すとか他の使い方は以下に  
https://docs.wandb.com/library/config

### wandb.log()

#### 基本

`wandb` は `history` (多分 `dict` の `list`)を持っており、`wandb.log()` が呼ばれるたびに引数に渡した `dict` がこれに `append` されていく。

```python
wandb.log({ 'accuracy': 0.9, 'epoch': 5 })
```

#### ステップを指定して記録

一つのステップの中で数カ所に分けて `wandb.log()` を呼びたい場合は `step` を明示的に指定する。

```python
wandb.log({ 'accuracy': 0.9 }, step=10)
wandb.log({ 'epoch': 5 }, step=10)
```

または `commit=False` を渡す。

```python
# まだ記録されない
wandb.log({ 'accuracy': 0.9 }, commit=False)
# ここで { 'accuracy': 0.9, 'loss': 0.2 } が記録される
wandb.log({ 'loss': 0.2 })
```

#### Summary Metrics

`wandb.log()` で記録したメトリクスの最後の値がそれぞれ自動で保存され、ウェブ UI ダッシュボードの Summary 欄に表示される。また、以下のように明示的に保存することもできる。

```python
# loss: 0.1 が Summary に記録される
wandb.log({ 'loss': 0.3 })
wandb.log({ 'loss': 0.2 })
wandb.log({ 'loss': 0.1 })

# 明示的に保存
wandb.run.summary["test_accuracy"] = test_accuracy
```

#### ウェブ UI で見るプロットの x 軸を変える

ウェブ UI でプロットを見るとき、x 軸を自由に設定できる。例えばバッチを x 軸にして経過を見たいときはバッチを記録に入れておくなどした上でダッシュボードで x 軸をそのキーで指定する。

```python
wandb.log({ 'batch': 5, ... })
```

#### ステップ数が多くなるとサンプリングされる

ウェブ UI で見れるプロットはデータが 1000 を超えると 1000 個だけランダムにサンプリングされるので注意。見るたびに微妙にプロットが違うということが起きうる。

#### その他記録できること（一部）

matplotlib.pyplot オブジェクトを渡すと[ploty](https://plot.ly/) に変換して記録するらしい（要検証）

```python
plt.plot( ... )
wandb.log( { 'chart': plt } ] )
```

- 画像
- 動画
- 音声
- テキスト/ひょう/HTML
- 点群データ

など。詳しくは以下を参照  
https://docs.wandb.com/library/log

## ログインについての詳細

`wandb` は W&B のプログラマブルなクライアントという感じなのでログインが必要。アカウントを持っていない場合は先にサインアップする。

```
wandb login
```

ブラウザが開いて API キーが出るのでそれをコピーして入力する。
ブラウザが使えない環境の場合、
https://app.wandb.ai/authorize
に行くと API キーが払い出されるので、これを入力する。ここで入力された API キーは `~/.netrc` に保存される。

```
machine api.wandb.ai
  login user
  password XXXXXXXXXXXXXXXXXXXXXXXXXXX(API key)
```

もしくは環境変数 `WANDB_API_KEY` に API キーをセットするとそれを読んでくれる。

## API 経由で W&B を使う方法

記録されたメトリクスを取得してきてスクリプトでなにかやるとかに使える。  
https://docs.wandb.com/library/api/examples

## W&B Sweeps でハイパラサーチ

ハイパラサーチがやれるツールらしい。
サーチを管理するための Sweep サーバがあり、ここに学習を行うマシン(複数可能)が学習の結果を報告し、Sweep サーバはそれを受けて学習のスケジューリングとか割り当てを行うっぽい。

大まかなフローは

1. yaml に探索範囲を記述し、それを Sweep サーバに送る
2. Sweep ID が返ってくるので、学習用のマシンでこれを引数に渡して Sweep エージェント起動
3. 多分勝手に学習とか始めてくれる。多分。

https://docs.wandb.com/sweeps
