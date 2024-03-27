# mmdetection-balloon-docker

### 依存環境

- Python 3.11.6
- CUDA 12.1.1
- torch==2.1.1
- torchvision==0.16.1

## 環境構築(Docker)

### CLI 上で学習を行う場合

1\. コンテナの作成と実行

```
docker compose up -d
```

2\. コンテナのシェルを起動する

```
docker compose exec -it balloon /bin/bash
```

3\. シェルを使って学習を実行する

例）

```
root@xxxxxxxxxx:/workspace# python mmdetection/tools/train.py <config file path>
```

4\. シェルから抜ける

```
exit
```

### Dev Containers 上で学習を行う場合

1\. コンテナの作成と実行

```
docker compose up
```

2\. リモートエクスプローラーの「開発コンテナー」を選択し、起動したコンテナにアタッチする

3\. VSCode 上でターミナルを表示し、学習を行う

```
root@xxxxxxxxxx:/workspace# python mmdetection/tools/train.py <config file path>
```

### コンテナの停止

```
docker compose stop
```

再起動する際は以下のコマンドを実行する。

```
docker compose start
```

### コンテナの削除

```
docker compose down
```

## 学習

```
python mmdetection/tools/train.py <config file path>
```
