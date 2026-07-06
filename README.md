# WCF v2

翻譯 API 服務，提供 NLLB 本地翻譯與外部 LLM 翻譯。

## LLM API Key 設定

LLM 端點需要 API Key 才能呼叫外部服務。Key **不可**提交到 git。

1. 複製範本並填入 Key：

```bash
cp docker/llm_secrets.yaml.example docker/llm_secrets.yaml
# 編輯 docker/llm_secrets.yaml，填入 api_key
```

`docker/llm_secrets.yaml` 格式：

```yaml
providers:
  openai:
    api_key: sk-your-api-key-here
```

2. 啟動時會自動載入：先讀 `llm_config.yaml`，再合併同目錄下的 `llm_secrets.yaml`。

3. 也可用環境變數覆寫（優先於 secrets 檔）：

```bash
export LLM_API_KEY=sk-your-api-key-here
```

## Docker Compose

### Build

```bash
docker-compose -p traslation_api -f docker/compose.yml build
```

### Start service

啟動前請確認 `docker/llm_secrets.yaml` 已建立（見上方 LLM API Key 設定）。

```bash
docker-compose -p traslation_api -f docker/compose.yml up -d
```

### Stop service

```bash
docker-compose -p traslation_api -f docker/compose.yml stop
```

### Check

```bash
docker-compose -p traslation_api -f docker/compose.yml ps
```

### Log

```bash
docker-compose -p traslation_api -f docker/compose.yml logs
```
