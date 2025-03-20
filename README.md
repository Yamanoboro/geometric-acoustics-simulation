# 幾何音響シミュレーション（多角形版）

多角形の部屋内での音響粒子の反射をシミュレーションするStreamlitアプリケーションです。

## 機能

- 多角形の形状（辺の数と半径）を設定可能
- 音源の位置を自由に設定
- 音響粒子数、反射回数などのパラメータをカスタマイズ
- シミュレーション結果をGIFアニメーションとして保存
- 反射データをテキストファイルとしてダウンロード

## インストール方法

このアプリケーションはStreamlitを使用しています。以下の手順でインストールしてください：

```bash
# リポジトリをクローン
git clone https://github.com/Yamanoboro/geometric-acoustics-simulation.git
cd geometric-acoustics-simulation

# 必要なライブラリをインストール
pip install -r requirements.txt

# アプリケーションを実行
streamlit run geometric_acoustics_sim_polygon_streamlit.py
```

## Renderへのデプロイ方法

1. GitHubアカウントを使用してRenderにサインアップ
2. 新しいWebサービスを作成し、このリポジトリを接続
3. 以下の設定を行う:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run geometric_acoustics_sim_polygon_streamlit.py --server.port=$PORT --server.address=0.0.0.0`

## パラメータの説明

- **多角形の辺の数**: 部屋の形状を決定する多角形の辺の数（3以上）
- **部屋の半径**: 中心から壁面までの距離（メートル）
- **音源位置 X/Y**: 音源の座標（メートル）
- **粒子数**: シミュレーションする音響粒子の数
- **最大反射回数**: 各粒子が反射できる最大回数
- **粒子サイズ**: 表示上の粒子の大きさ
- **グリッド間隔**: 表示グリッドの間隔（1mまたは5m）

## 注意事項

- 粒子数や反射回数を増やすと計算時間が長くなります
- 粒子の衝突検出は近似計算を使用しています
- ウェブブラウザで実行する場合、ローカル実行より処理速度が遅くなる可能性があります

## ライセンス

MITライセンス

## 作者

Yamanoboro

## 幾何音響シミュレーション オンラインデモ

幾何音響シミュレーションをインストールせずに試すには、以下のリンクからStreamlitアプリケーションにアクセスできます：

- [幾何音響シミュレーション - Streamlitデモ](https://geometric-acoustics-simulation-6oovcfdw3x76gwzweuhyka.streamlit.app/)

※サーバーの状態によってはロード時間がかかる場合があります。初回アクセス時に数十秒お待ちいただくことがあります。
