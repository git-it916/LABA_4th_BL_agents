import transformers
import torch
import gc
import os
import warnings

# 환경 변수 설정 (메모리 단편화 방지)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 전역 파이프라인 캐시
_pipeline_cache = None

def prepare_pipeline_obj():
    """
    파이프라인을 한 번만 생성하고 재사용합니다.

    Returns:
        pipeline: Hugging Face 텍스트 생성 파이프라인

    Raises:
        RuntimeError: CUDA를 사용할 수 없거나 모델 로드 실패 시
    """
    global _pipeline_cache

    if _pipeline_cache is not None:
        print("[알림] 기존 파이프라인 재사용")
        return _pipeline_cache

    print("[알림] 새 파이프라인 생성 중...")

    # GPU 사용 가능 여부 확인
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        warnings.warn(
            "CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다. 속도가 매우 느릴 수 있습니다.",
            category=UserWarning
        )
        print("[경고] GPU를 사용할 수 없습니다. CPU 모드로 진행합니다.")
    else:
        print(f"[알림] GPU 감지: {torch.cuda.get_device_name(0)}")
        print(f"[알림] 총 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 메모리 완전 정리 (GPU 사용 시에만)
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"[알림] 정리 전 GPU 메모리: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    # 모델 설정
    device_map = "auto" if use_cuda else "cpu"
    model_dtype = torch.float16 if use_cuda else torch.float32

    try:
        # 양자화 없이 float16만 사용 (GPU) 또는 float32 (CPU)
        _pipeline_cache = transformers.pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            model_kwargs={
                "dtype": model_dtype,  # torch_dtype -> dtype (deprecated 경고 수정)
                "low_cpu_mem_usage": True,
            },
            device_map=device_map,
        )

        if use_cuda:
            print(f"[알림] 모델 로드 완료. GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        else:
            print("[알림] 모델 로드 완료 (CPU 모드)")

    except OSError as e:
        if "gated repo" in str(e).lower():
            print("\n[오류] Llama 3 모델 접근 권한이 필요합니다.")
            print("해결 방법:")
            print("1. https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct 에서 접근 권한 요청")
            print("2. Hugging Face 토큰 설정:")
            print("   from huggingface_hub import login")
            print("   login()")
            raise RuntimeError("Llama 3 모델 접근 권한이 필요합니다. 위 안내를 참고하세요.") from e
        else:
            raise
    except Exception as e:
        print(f"\n[오류] 모델 로드 중 예외 발생: {e}")
        raise

    return _pipeline_cache


def cleanup_pipeline():
    """
    파이프라인 메모리를 해제합니다.
    """
    global _pipeline_cache

    if _pipeline_cache is not None:
        print("[알림] 파이프라인 메모리 해제 중...")
        del _pipeline_cache
        _pipeline_cache = None
        gc.collect()

        # GPU 사용 시에만 CUDA 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[알림] 파이프라인 메모리 해제 완료")


def chat_with_llama3(pipeline_obj, system_prompt, user_prompt):
    """
    Llama 3 모델과 채팅하여 텍스트를 생성합니다.

    Args:
        pipeline_obj: Hugging Face 파이프라인 객체
        system_prompt (str): 시스템 프롬프트
        user_prompt (str): 사용자 프롬프트

    Returns:
        str: 생성된 텍스트
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    prompt = pipeline_obj.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # EOS 토큰: 모델의 기본 EOS 토큰만 사용
    # "]", "}" 등을 EOS로 설정하면 JSON 배열이 완성되기 전에 중단될 수 있음
    eos_tokens = [
        pipeline_obj.tokenizer.eos_token_id,
        pipeline_obj.tokenizer.convert_tokens_to_ids("<|eot_id|>")  # Llama 3의 공식 종료 토큰
    ]

    # torch.no_grad()는 pipeline 내부에서 이미 처리되므로 제거 가능
    # 하지만 명시적으로 사용해도 문제 없음
    outputs = pipeline_obj(
        prompt,
        max_new_tokens=12288,
        do_sample=True,
        temperature=0.2,
        top_p=0.80,
        return_full_text=False,
        eos_token_id=eos_tokens,
        pad_token_id=pipeline_obj.tokenizer.eos_token_id
    )

    # GPU 사용 시에만 CUDA 캐시 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return outputs[0]["generated_text"].strip()


def call_gemini_api(system_prompt, user_prompt):
    """
    Google Gemini API를 사용하여 텍스트를 생성합니다.

    공식 문서: https://ai.google.dev/gemini-api/docs/quickstart?lang=python

    Args:
        system_prompt (str): 시스템 프롬프트
        user_prompt (str): 사용자 프롬프트

    Returns:
        str: 생성된 텍스트

    Raises:
        ValueError: API 키가 없을 경우
        RuntimeError: API 호출 실패 시
    """
    import os
    from google import genai

    # .env 파일에서 환경 변수 로드
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # dotenv가 없으면 환경 변수 직접 로드
        import pathlib
        env_path = pathlib.Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value

    # API 키 로드
    api_key = os.getenv('GOOGLE_API_KEY')

    if not api_key:
        raise ValueError(
            "Gemini API key not found.\n"
            "Create a .env file with: GOOGLE_API_KEY=your_key_here\n"
            "Get your key at: https://aistudio.google.com/app/apikey"
        )

    # Gemini 클라이언트 초기화
    client = genai.Client(api_key=api_key)

    # 프롬프트 결합 (Gemini는 system + user를 하나로 받음)
    full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

    # 사용할 모델 지정 (Gemini 2.5 Pro - 가장 강력한 최신 모델)
    model_name = 'gemini-2.5-pro'

    print(f"[Gemini] API 호출 중... (모델: {model_name})")

    try:
        # 콘텐츠 생성 요청
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt
        )

        # 응답 텍스트 추출
        response_text = response.text

        print(f"[Gemini] 응답 수신 완료 (길이: {len(response_text)} 문자)")

        return response_text.strip()

    except Exception as e:
        raise RuntimeError(f"Gemini API 호출 중 오류 발생: {e}")