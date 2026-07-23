@echo off
REM ============================================================
REM  无垠AI 部署脚本 (Windows Batch)
REM  步骤: 合并LoRA → 转换GGUF → 创建Ollama模型 → 测试
REM ============================================================
setlocal enabledelayedexpansion

echo.
echo ===============================================
echo   无垠AI - Wuyin AI Deployment Script
echo ===============================================
echo.

REM --- 配置 ---
set BASE_MODEL=Qwen2.5-1.5B
set BASE_MODEL_DIR=E:\models\%BASE_MODEL%
set LORA_ADAPTER=E:\models\wuyin-lora-adapter
set MERGED_OUTPUT=E:\models\wuyin-merged
set GGUF_OUTPUT=E:\claude code\wuyin-ai\wuyin-qwen2.5-1.5b-merged.Q4_K_M.gguf
set OLLAMA_MODEL_NAME=wuyin-ai

REM ============================================
REM Step 1: Merge LoRA adapter into base model
REM ============================================
echo [1/5] Merging LoRA adapter into base model...
echo.

if not exist "%BASE_MODEL_DIR%" (
    echo [ERROR] Base model not found: %BASE_MODEL_DIR%
    echo Please download Qwen2.5-1.5B to %BASE_MODEL_DIR% first.
    exit /b 1
)

if not exist "%LORA_ADAPTER%" (
    echo [ERROR] LoRA adapter not found: %LORA_ADAPTER%
    echo Please ensure the wuyin-lora-adapter is at %LORA_ADAPTER%
    exit /b 1
)

echo Merging %LORA_ADAPTER% into %BASE_MODEL_DIR% ...
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; from peft import PeftModel; import torch; base = AutoModelForCausalLM.from_pretrained('%BASE_MODEL_DIR%', torch_dtype=torch.float16, trust_remote_code=True); model = PeftModel.from_pretrained(base, '%LORA_ADAPTER%'); merged = model.merge_and_unload(); merged.save_pretrained('%MERGED_OUTPUT%', safe_serialization=True); tokenizer = AutoTokenizer.from_pretrained('%BASE_MODEL_DIR%', trust_remote_code=True); tokenizer.save_pretrained('%MERGED_OUTPUT%'); print('[OK] Merge complete:', '%MERGED_OUTPUT%')"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] LoRA merge failed!
    exit /b 1
)

echo [OK] LoRA merge complete.
echo.

REM ============================================
REM Step 2: Convert merged model to GGUF
REM ============================================
echo [2/5] Converting merged model to GGUF format (Q4_K_M)...
echo.

if not exist "%MERGED_OUTPUT%" (
    echo [ERROR] Merged model not found: %MERGED_OUTPUT%
    exit /b 1
)

python llama.cpp\convert_hf_to_gguf.py "%MERGED_OUTPUT%" --outtype q4_k_m --outfile "%GGUF_OUTPUT%"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] GGUF conversion failed!
    echo Make sure llama.cpp is cloned and convert_hf_to_gguf.py is available.
    exit /b 1
)

echo [OK] GGUF conversion complete: %GGUF_OUTPUT%
echo.

REM ============================================
REM Step 3: Create Ollama model from Modelfile
REM ============================================
echo [3/5] Creating Ollama model: %OLLAMA_MODEL_NAME% ...
echo.

if not exist "E:\claude code\wuyin-ai\Modelfile" (
    echo [ERROR] Modelfile not found!
    exit /b 1
)

pushd "E:\claude code\wuyin-ai"
ollama create %OLLAMA_MODEL_NAME% -f Modelfile
popd

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Ollama create failed!
    echo Make sure Ollama is installed and running.
    exit /b 1
)

echo [OK] Ollama model created.
echo.

REM ============================================
REM Step 4: Start Ollama serve (if not running)
REM ============================================
echo [4/5] Ensuring Ollama serve is running...
echo.

REM Check if Ollama is already running
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Starting Ollama serve...
    start "" ollama serve
    echo Waiting for Ollama to start...
    timeout /t 5 /nobreak >nul
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Ollama serve failed to start!
        exit /b 1
    )
)

echo [OK] Ollama serve is running.
echo.

REM ============================================
REM Step 5: Test with a sample query
REM ============================================
echo [5/5] Testing with sample query...
echo.

echo Query: "无垠AI，请简单介绍一下你自己。"
echo.

curl -s http://localhost:11434/api/generate -d "{\"model\":\"%OLLAMA_MODEL_NAME%\",\"prompt\":\"无垠AI，请简单介绍一下你自己。\",\"stream\":false}" | python -c "import sys,json; d=json.load(sys.stdin); print('Response:', d.get('response','(no response)'))"

if %ERRORLEVEL% neq 0 (
    echo [WARNING] Test query failed, but model may still work.
    echo Try manually: ollama run %OLLAMA_MODEL_NAME%
) else (
    echo [OK] Test query successful!
)

echo.
echo ===============================================
echo   无垠AI deployment complete!
echo   Model name: %OLLAMA_MODEL_NAME%
echo   Test with:  ollama run %OLLAMA_MODEL_NAME%
echo   API:        http://localhost:11434/v1/chat/completions
echo ===============================================

endlocal
