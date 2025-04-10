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
        return self._generate(input_ids, config)

    def _generate_stream(self, input_ids: list[int], config: GenerationConfig):
        start_time = time.time()
        first_token_time = None
        input_size = len(input_ids)
        # 원래 요기부터
        # gen = tqdm(self.model.generate(
        #     input_ids,
        #     temp=config.sampler_config.temperature,
        #     repeat_penalty=config.sampler_config.repetition_penalty,
        #     top_k=config.sampler_config.top_k,
        #     top_p=config.sampler_config.top_p,
        #     min_p=config.sampler_config.min_p,
        #     mirostat_eta=config.sampler_config.mirostat_eta,
        #     mirostat_tau=config.sampler_config.mirostat_tau,
        #     **config.additional_gen_config,
        # ))
        # 원래 요기까지
        # 요기부터
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
        print(type(gen))

        if hasattr(gen, '__iter__') and not isinstance(gen, str):
            print("✅ This is a generator, yields tokens one-by-one.")
        else:
            print("❌ This is not a generator, it's a complete output.")

        # 요기까지
        logger.info(f"Generating tokens with config: {config}")
        logger.info(f"✨ ✨ ✨ ✨ ✨ ✨ ✨ gen: {gen}")
        for token in gen:
            logger.info(f"⭐️Token: {token}⭐️")

            if first_token_time is None:
                first_token_time = time.time() - start_time
                logger.info(f"First token generation time: {first_token_time:.2f}s")
            
            yield token
            input_size += 1
            if (llama_token_is_eog(self.is_eog(), token) or 
                input_size >= config.max_length):
                break
            # 원래 요기부터
            # gen.set_postfix({
            #     "tokens": input_size,
            #     "max tokens": config.max_length,
            #     "first token time": f"{first_token_time:.2f}s",
            #     "total time": f"{time.time() - start_time:.2f}s"
            # })
            # 원래 요기까지
        total_time = time.time() - start_time
        logger.info(f"Total token generation time: {total_time:.2f}s")

    def _generate(self, input_ids: list[int], config: GenerationConfig) -> list:
        start_time = time.time()
        new_tokens = []
        # 요기부터
        first_saved_audio_time = None
        new_tokens2 = []
        audio_chunk_index = 1
        # 요기까지
        for token in self._generate_stream(input_ids, config):
            # # 요기부터
            if self.get_audio:
                new_tokens2.append(token)
                logger.info(f"new_tokens2: {new_tokens2}")

                audio_pieces = self.get_audio(new_tokens2)
                if audio_pieces is None:
                    logger.warning("Audio is empty, skipping save.")
                    continue

                current_audio_time = time.time() - start_time
                if first_saved_audio_time is None:
                    first_saved_audio_time = current_audio_time
                    logger.info(
                        f"🎧 First audio chunk generated at: {first_saved_audio_time:.2f}s "
                        f"(chunk {audio_chunk_index}, tokens: {len(new_tokens2)}, last_token: {token})"
                    )
                else:
                    logger.info(
                        f"🎵 Audio chunk {audio_chunk_index} generated at: {current_audio_time:.2f}s "
                        f"(tokens: {len(new_tokens2)}, last_token: {token})"
                    )

                
                model_output_itf = ModelOutput(audio_pieces, 24000)
                filename = f"chunked_output_{audio_chunk_index}_tok-{token}.wav"
                model_output_itf.save(filename)

                audio_chunk_index += 1
            # # 요기까지
            logger.info(f"type of token: {type(token)}")
            new_tokens.append(token)
        total_time = time.time() - start_time
        logger.info(f"Generated {len(new_tokens)} tokens: {new_tokens}, type: {type(new_tokens)}")
        logger.info(f"Total generation time: {total_time:.2f}s")
        return new_tokens