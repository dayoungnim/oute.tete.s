import outetts
import time

# Initialize the interface
interface = outetts.Interface(
    config=outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_1B,
        # For llama.cpp backend
        backend=outetts.Backend.LLAMACPP,
        quantization=outetts.LlamaCppQuantization.FP16
        # For transformers backend
        # backend=outetts.Backend.HF,
    )
)

# interface = outetts.Interface(
#     config=outetts.ModelConfig.auto_config(
#         model=outetts.Models.VERSION_1_0_SIZE_1B,
#         # For transformers backend
#         backend=outetts.Backend.HF,
#     )
# )

# Load the default speaker profile
# speaker = interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")

# Or create your own speaker profiles in seconds and reuse them instantly
speaker = interface.create_speaker("/home/dychoi/chralesent.wav")
interface.save_speaker(speaker, "speaker.json")
speaker = interface.load_speaker("speaker.json")

# Start measuring time
start_time = time.time()

# Generate speech
output = interface.generate(
    config=outetts.GenerationConfig(
        text="""안녕하세요 상담원 김정숙입니다, 무엇을 도와드릴까요?""",
        generation_type=outetts.GenerationType.CHUNKED,
        speaker=speaker,
        sampler_config=outetts.SamplerConfig(
            temperature=0.4
        ),
    )
)

# Calculate total generation time
total_time = time.time() - start_time
print(f"Total TTS generation time: {total_time:.2f} seconds")

# Save to file
output.save("output.wav")
