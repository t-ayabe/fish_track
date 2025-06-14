# ベースイメージ
FROM python:3.10

# 作業ディレクトリ作成
WORKDIR /app

# 必要ファイルをコピー
COPY requirements.txt ./
COPY your_source_code_dir ./  

# パッケージインストール
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# デフォルトコマンド（例）
CMD ["python", "main.py"]
