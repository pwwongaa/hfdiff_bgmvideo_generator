import os
import subprocess
import argparse
import requests
from io import BytesIO
from PIL import Image
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

#? Hugging Face Token
# HF_TOKEN = "...Your HF token..."
# export HF_TOKEN=your_token
HF_TOKEN = os.getenv("HF_TOKEN")

# 🎵 Music Generation
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def generate_music(prompt: str, output_path="output_music.wav") -> str:
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    model.set_generation_params(duration=3600)
    wav = model.generate([prompt])
    audio_write(output_path.replace(".wav", ""), wav[0].cpu(), model.sample_rate, strategy="loudness")
    return output_path

def generate_image_cloud(prompt: str) -> Image.Image:
    print(f"🎨 Sending prompt to Hugging Face API: {prompt}")
    url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "height": 1080,
            "width": 1920
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"❌ Image generation HTTP error: {e}")

    if response.status_code != 200:
        print(f"❌ Failed: {response.status_code}\n{response.text}")
        raise RuntimeError("❌ Image generation failed.")

    try:
        return Image.open(BytesIO(response.content))
    except Exception as e:
        raise RuntimeError("❌ Image decoding failed.") from e

def combine_to_video(image_path: str, audio_path: str, output_path=f"output_video.mp4"):
    print(f"🎮 Combining image + audio into video...")
    command = [
        "ffmpeg", "-loop", "1",
        "-i", image_path,
        "-i", audio_path,
        "-c:v", "libx264", "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p", "-shortest",
        "-vf", "scale=1920:1080",
        output_path
    ]
    subprocess.run(command)
    print(f"✅ Video generated: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="輸入主題")
    args = parser.parse_args()

    # 圖片生成
    image = generate_image_cloud(args.prompt)
    image_path = f"cover1080p.png"
    image.save(image_path)

    # 音樂生成
    music_path = generate_music(args.prompt)

    # 合成影片
    combine_to_video(image_path, music_path)

if __name__ == "__main__":
    main()


#? loop audio tracks
def loop_audio_to_duration(input_audio: str, output_audio: str, duration_sec: int = 600):
    print(f"🔁 Looping audio to {duration_sec} seconds...")
    command = [
        "ffmpeg",
        "-stream_loop", "-1",  # 無限重複
        "-i", input_audio,
        "-t", str(duration_sec),  # 指定輸出長度
        "-c", "copy",
        output_audio
    ]
    subprocess.run(command)
    print(f"✅ Audio looped to: {output_audio}")
