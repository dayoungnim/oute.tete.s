from loguru import logger
from tqdm import tqdm
from packaging import version
import time
# 요기부터
from typing import Optional, Callable, Any
import numpy as np
from outetts.version.playback import ModelOutput
# 요기까지


from .info import GenerationType
from .config import GenerationConfig

try:
    from llama_cpp import Llama, llama_token_is_eog
    from llama_cpp import __version__ as llama_cpp_version
    _GGUF_AVAILABLE = True
except:
    llama_cpp_version = "0.0.0"
    _GGUF_AVAILABLE = False
    raise ImportError(
        "llama.cpp Python bindings not found. This is required for GGUF model support.\n\n"
        "To install, please follow our installation guide:\n"
        "https://github.com/edwko/OuteTTS?tab=readme-ov-file#installation\n\n"
    )

CURRENT_VERSION = version.parse(llama_cpp_version)
VERSION_0_3_7 = version.parse("0.3.7")


class GGUFModel:
    def __init__(
            self,
            model_path: str,
            n_gpu_layers: int = 0,
            max_seq_length: int = 4096,
            additional_model_config: dict = {},
            # 요기부터
            get_audio_fn: Optional[Callable[[Any], np.ndarray]] = None,
            # 요기까지
    ) -> None:

        # 요기부터
        self.get_audio = get_audio_fn
        # 요기까지
        if not _GGUF_AVAILABLE:
            raise ImportError(
                "llama_cpp python module not found."
            )

        additional_model_config["n_ctx"] = max_seq_length
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            last_n_tokens_size=64,
            **additional_model_config
        )

    def is_eog(self):
        if CURRENT_VERSION >= VERSION_0_3_7:
            return self.model._model.vocab
        else:
            return self.model._model.model

    def generate(self, input_ids: list[int], config: GenerationConfig):
        if config.generation_type == GenerationType.STREAM:
            return self._generate_stream(input_ids, config)
        return self._generate_stream(input_ids, config)

    # def _generate_stream(self, input_ids: list[int], config: GenerationConfig):
    #     input_size = len(input_ids)
    #     gen = self.model.generate(
    #         input_ids,
    #         temp=config.sampler_config.temperature,
    #         repeat_penalty=config.sampler_config.repetition_penalty,
    #         top_k=config.sampler_config.top_k,
    #         top_p=config.sampler_config.top_p,
    #         min_p=config.sampler_config.min_p,
    #         mirostat_eta=config.sampler_config.mirostat_eta,
    #         mirostat_tau=config.sampler_config.mirostat_tau,
    #         **config.additional_gen_config,
    #     )
    #     print(type(gen))

    #     logger.info(f"Generating tokens with config: {config}")
    #     logger.info(f"✨ ✨ ✨ ✨ ✨ ✨ ✨ gen: {gen}")

    #     start_time = None  # 첫 토큰 시간
    #     prev_time = None   # 이전 토큰 시간
    #     token_count = 0

    #     gen_start_time = time.perf_counter()
    #     for token in gen:
    #         current_time = time.perf_counter()

    #         if start_time is None:
    #             first_token_delay = current_time - gen_start_time
    #             start_time = current_time
    #             logger.info(f"⏱ 첫 토큰 생성까지 걸린 시간: {first_token_delay:.4f}초")
    #         else:
    #             delta = current_time - prev_time
    #             logger.info(
    #                 f"⏱ 토큰 {token_count} → {token_count + 1} 생성 시간차: {delta:.4f}초")

    #         logger.info(f"⭐️Token: {token}⭐️")

    #         yield token

    #         prev_time = current_time
    #         token_count += 1
    #         input_size += 1

    #         if (llama_token_is_eog(self.is_eog(), token) or
    #                 input_size >= config.max_length):
    #             break

    #     total_time = time.perf_counter() - start_time if start_time else 0
    #     logger.info(f"🏁 전체 토큰 생성 시간: {total_time:.4f}초")

    def _generate_stream(self, input_ids: list[int], config: GenerationConfig):
        input_size = len(input_ids)
        gen = self.model.generate(
            input_ids,
            temp=config.sampler_config.temperature,
            repeat_penalty=config.sampler_config.repetition_penalty,
            top_k=config.sampler_config.top_k,
            top_p=config.sampler_config.top_p,
            min_p=config.sampler_config.min_p,
            mirostat_eta=config.sampler_config.mirostat_eta,
            mirostat_tau=config.sampler_config.mirostat_tau,
            **config.additional_gen_config,
        )

        start_time = None  # 첫 토큰 시간
        first_audio_time = None  # 첫 오디오 생성 시간
        token_count = 0
        new_tokens = []

        gen_start_time = time.perf_counter()
        for token in gen:
            current_time = time.perf_counter()

            if start_time is None:
                first_token_delay = current_time - gen_start_time
                start_time = current_time
                logger.info(f"⏱ 첫 토큰 생성까지 걸린 시간: {first_token_delay:.4f}초")

            if self.get_audio:
                new_tokens.append(token)
                audio_pieces = self.get_audio(new_tokens)

                if audio_pieces is not None and first_audio_time is None:
                    first_audio_time = time.perf_counter()
                    audio_delay = first_audio_time - gen_start_time
                    logger.info(f"🔊 첫 audio_pieces 생성까지 걸린 시간: {audio_delay:.4f}초")

                if audio_pieces is None:
                    continue

            token_count += 1
            input_size += 1

            if (llama_token_is_eog(self.is_eog(), token) or
                    input_size >= config.max_length):
                break

        total_time = time.perf_counter() - start_time if start_time else 0
        logger.info(f"🏁 전체 토큰 생성 시간: {total_time:.4f}초")
        return new_tokens



    # def _generate(self, input_ids: list[int], config: GenerationConfig) -> list:
    #     new_tokens = []
    #     # 요기부터
    #     new_tokens2 = []
    #     audio_chunk_index = 1
    #     # 요기까지
    #     for token in self._generate_stream(input_ids, config):
    #         # # 요기부터
    #         if self.get_audio:
    #             new_tokens2.append(token)
    #             logger.info(f"new_tokens2: {new_tokens2}")

    #             audio_start_time = time.perf_counter()

    #             audio_pieces = self.get_audio(new_tokens2)

    #             audio_end_time = time.perf_counter()
    #             elapsed_audio_time = audio_end_time - audio_start_time
    #             logger.info(f"🕒 get_audio() with {len(new_tokens2)} tokens took {elapsed_audio_time:.4f} seconds")

    #             if audio_pieces is None:
    #                 logger.warning("Audio is empty, skipping save.")
    #                 continue

    #             model_output_itf = ModelOutput(audio_pieces, 24000)
    #             filename = f"chunked_output_{audio_chunk_index}_tok-{token}.wav"
    #             model_output_itf.save(filename)

    #             audio_chunk_index += 1

    #             # # 요기까지
    #             new_tokens.append(token)
    #     logger.info(f"Generated {len(new_tokens)} tokens: {new_tokens}")
    #     return new_tokens

    # def _generate(self, input_ids: list[int], config: GenerationConfig, chunk_size: int = 40) -> list:
    #     new_tokens = []
    #     temp_tokens = []  # chunk_size만큼 모을 임시 토큰 리스트
    #     audio_chunk_index = 1

    #     for token in self._generate_stream(input_ids, config):
    #         if self.get_audio:
    #             temp_tokens.append(token)

    #             # chunk_size만큼 쌓이면 처리
    #             if len(temp_tokens) == chunk_size:
    #                 logger.info(f"🎧 Generating audio from {chunk_size} tokens: {temp_tokens}")

    #                 audio_start_time = time.perf_counter()
    #                 audio_pieces = self.get_audio(temp_tokens)
    #                 audio_end_time = time.perf_counter()

    #                 elapsed_audio_time = audio_end_time - audio_start_time
    #                 logger.info(f"🕒 get_audio() with {chunk_size} tokens took {elapsed_audio_time:.4f} seconds")

    #                 if audio_pieces is None:
    #                     logger.warning("Audio is empty, skipping save.")
    #                 else:
    #                     model_output_itf = ModelOutput(audio_pieces, 24000)
    #                     filename = f"chunked_output_{audio_chunk_index}_tok-{token}.wav"
    #                     model_output_itf.save(filename)
    #                     audio_chunk_index += 1

    #                 temp_tokens = []  # 다음 chunk 준비

    #         new_tokens.append(token)

    #     # 남은 토큰 처리
    #     if self.get_audio and temp_tokens:
    #         logger.info(f"🎧 Generating audio from final tokens (len={len(temp_tokens)}): {temp_tokens}")
    #         audio_pieces = self.get_audio(temp_tokens)
    #         if audio_pieces:
    #             model_output_itf = ModelOutput(audio_pieces, 24000)
    #             filename = f"chunked_output_{audio_chunk_index}_final.wav"
    #             model_output_itf.save(filename)

    #     logger.info(f"Generated {len(new_tokens)} tokens: {new_tokens}")
    #     return new_tokens




    # def _generate(self, input_ids: list[int], config: GenerationConfig) -> list:
    #     new_tokens = []
    #     new_tokens2 = []
    #     audio_chunk_index = 1

    #     first_audio_start_time = None  # 최초 시작 시간 저장용
    #     first_audio_ready_time = None  # 최초 audio_pieces 생성 시간 저장용

    #     for token in self._generate_stream(input_ids, config):
    #         if self.get_audio:
    #             new_tokens2.append(token)
    #             logger.info(f"new_tokens2: {new_tokens2}")

    #             if first_audio_start_time is None:
    #                 first_audio_start_time = time.perf_counter()
    #                 logger.info(f"⏱ 첫 audio_pieces 생성 시작 시간: {first_audio_start_time:.4f}초")

    #             audio_start_time = time.perf_counter()
    #             audio_pieces = self.get_audio(new_tokens2)
    #             audio_end_time = time.perf_counter()

    #             elapsed_audio_time = audio_end_time - audio_start_time
    #             logger.info(f"🕒 get_audio() with {len(new_tokens2)} tokens took {elapsed_audio_time:.4f} seconds")

    #             if audio_pieces is None:
    #                 logger.warning("Audio is empty, skipping save.")
    #                 continue

    #             # 첫 번째 audio_pieces가 생성된 순간 기록
    #             if first_audio_ready_time is None:
    #                 first_audio_ready_time = time.perf_counter()
    #                 logger.info(f"⏱ 첫 audio_pieces 생성 완료 시간: {first_audio_ready_time:.4f}초")
    #                 time_to_first_audio = first_audio_ready_time - first_audio_start_time
    #                 logger.info(f"⏱️ Time until first audio_pieces was ready: {time_to_first_audio:.4f} seconds")

    #             model_output_itf = ModelOutput(audio_pieces, 24000)
    #             filename = f"chunked_output_{audio_chunk_index}_tok-{token}.wav"
    #             model_output_itf.save(filename)

    #             audio_chunk_index += 1

    #             new_tokens.append(token)

    #     logger.info(f"Generated {len(new_tokens)} tokens: {new_tokens}")
    #     return new_tokens
