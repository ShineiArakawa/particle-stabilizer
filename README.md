# particle-stabilizer

## ビルド
CMakeを用いてビルドを行う。以下にプラットフォーム毎のビルド方法を示す。

### Windows
Windowsでのビルドには、Visual Studioを用いる。Visual StudioからCMake Projectとして本リポジトリを開き、ソリューションのビルドを行う。

### Linux
Linux環境でのビルドには、CMakeとgccコンパイラを用いる。(Clangコンパイラでのビルドには追加のオプションが必要)
以下のコマンドを実行する。
```bash
mkdir build && cd build
cmake -G Ninja ..
ninja
```

## 実行
ビルドが完了すると実行可能プログラムである`particle_stabilizer_main`が生成される。また、`NVIDIA CUDA Toolkit`が見つかった場合のみ、共有ライブラリとして`libparticle_stabilizer_cuda`が生成される。以下のコマンドでプログラムを実行する。
```bash
cd build/src/App

./particle_stabilizer_main \
    --box-min 0.0 0.0 0.0 \
    --box-min 10.0 20.0 30.0 \
    --out-dat-path {OUTPUT_DAT_PATH} \
    {INPUT_DAT_PATH}
```

詳細なオプションを表示するためには以下のコマンドを実行する。
```bash
./particle_stabilizer_main -h
```
