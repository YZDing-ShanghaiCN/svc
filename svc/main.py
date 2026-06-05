from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_SONG_DIR = "wav/test02"
DEFAULT_VOICE_GAIN = 1.8
DEFAULT_BACKGROUND_GAIN = 0.85
DEFAULT_DEMUCS_DEVICE = "cpu"

SOURCE_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}

IGNORED_SOURCE_NAMES = {
    "wav_voice.wav",
    "wav_bgmusic.wav",
    "my_voice.wav",
    "output.wav",
}


class PipelineError(RuntimeError):
    """A user-facing pipeline error."""


def svc_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def resolve_path_or_none(value: str | None, base: Path) -> Path | None:
    if value is None:
        return None
    return resolve_path(value, base)


def resolve_input_arg(value: str | None, svc_dir: Path, song_dir: Path) -> Path:
    if value is None:
        return find_input_media(song_dir)

    raw = Path(value).expanduser()
    if raw.is_absolute():
        input_path = raw.resolve()
    else:
        svc_candidate = (svc_dir / raw).resolve()
        song_candidate = (song_dir / raw).resolve()
        if raw.parent == Path(".") and song_candidate.exists() and not svc_candidate.exists():
            input_path = song_candidate
        else:
            input_path = svc_candidate

    ensure_source_file(input_path, "input media")
    return input_path


def ensure_source_file(path: Path, label: str) -> None:
    if not path.exists():
        raise PipelineError(f"Missing {label}: {path}")
    if not path.is_file():
        raise PipelineError(f"{label} is not a file: {path}")
    if path.suffix.lower() not in SOURCE_EXTENSIONS:
        supported = ", ".join(sorted(SOURCE_EXTENSIONS))
        raise PipelineError(f"{label} must use a supported extension ({supported}): {path}")


def ensure_wav_file(path: Path, label: str) -> None:
    if not path.exists():
        raise PipelineError(f"Missing {label}: {path}")
    if not path.is_file():
        raise PipelineError(f"{label} is not a file: {path}")
    if path.suffix.lower() != ".wav":
        raise PipelineError(f"{label} must be a .wav file: {path}")


def find_input_media(song_dir: Path) -> Path:
    if not song_dir.exists():
        raise PipelineError(f"Song directory does not exist: {song_dir}")
    if not song_dir.is_dir():
        raise PipelineError(f"Song directory is not a directory: {song_dir}")

    candidates = [
        child
        for child in sorted(song_dir.iterdir(), key=lambda path: path.name.lower())
        if child.is_file()
        and child.suffix.lower() in SOURCE_EXTENSIONS
        and child.name.lower() not in IGNORED_SOURCE_NAMES
    ]

    input_named = [path for path in candidates if path.stem.lower() == "input"]
    if input_named:
        wav_input = [path for path in input_named if path.suffix.lower() == ".wav"]
        return wav_input[0] if wav_input else input_named[0]
    stems = {path.stem.lower() for path in candidates}
    if len(stems) == 1:
        return preferred_source_candidate(candidates)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        ignored = ", ".join(sorted(IGNORED_SOURCE_NAMES))
        supported = ", ".join(sorted(SOURCE_EXTENSIONS))
        raise PipelineError(
            f"No source media found in {song_dir}. Supported extensions: {supported}. "
            f"Ignored generated files: {ignored}"
        )

    names = "\n  - ".join(str(path) for path in candidates)
    raise PipelineError(
        f"Multiple source media files found in {song_dir}. Use --input.\n"
        f"Candidates:\n  - {names}"
    )


def preferred_source_candidate(candidates: list[Path]) -> Path:
    priority = {
        ".mp4": 0,
        ".m4a": 1,
        ".wav": 2,
        ".mp3": 3,
    }
    return sorted(
        candidates,
        key=lambda path: (priority.get(path.suffix.lower(), 10), path.name.lower()),
    )[0]


def known_venv_pythons(svc_dir: Path) -> list[Path]:
    executable = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    roots = [
        svc_dir.parent / "myvenv",
        svc_dir / "myvenv",
        svc_dir.parent / ".venv",
        svc_dir / ".venv",
        svc_dir.parent / "venv",
        svc_dir / "venv",
    ]
    return [(root / executable).resolve() for root in roots]


def unique_existing_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path).lower() if os.name == "nt" else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            result.append(path)
    return result


def demucs_python_candidates(svc_dir: Path, explicit: str | None) -> list[Path]:
    if explicit:
        return [resolve_path(explicit, svc_dir)]

    candidates = [Path(sys.executable).resolve(), *known_venv_pythons(svc_dir)]
    return unique_existing_paths(candidates)


def default_inference_python(svc_dir: Path, explicit: str | None) -> Path:
    if explicit:
        return resolve_path(explicit, svc_dir)

    current = Path(sys.executable).resolve()
    project_venvs = unique_existing_paths(known_venv_pythons(svc_dir))
    for python_path in project_venvs:
        try:
            if current.samefile(python_path):
                return current
        except OSError:
            pass
    if project_venvs:
        return project_venvs[0]
    return current


def python_can_import(python_path: Path, module: str, cwd: Path) -> tuple[bool, str]:
    if not python_path.exists():
        return False, "file does not exist"

    command = [str(python_path), "-c", f"import {module}"]
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        return False, str(exc)
    return result.returncode == 0, (result.stderr or result.stdout).strip()


def choose_demucs_python(svc_dir: Path, explicit: str | None) -> Path:
    attempts: list[str] = []
    for python_path in demucs_python_candidates(svc_dir, explicit):
        ok, detail = python_can_import(python_path, "demucs", svc_dir)
        if ok:
            return python_path
        attempts.append(f"{python_path}: {detail or 'cannot import demucs'}")
        if explicit:
            break

    searched = "\n  - ".join(attempts) if attempts else "no Python candidates found"
    raise PipelineError(
        "Could not find a Python interpreter with demucs installed. "
        "Use --demucs-python to point at a venv Python.\n"
        f"Tried:\n  - {searched}"
    )


def display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def is_current(outputs: Iterable[Path], inputs: Iterable[Path]) -> bool:
    output_paths = list(outputs)
    input_paths = list(inputs)
    if not output_paths or not input_paths:
        return False
    if not all(path.exists() for path in output_paths):
        return False
    newest_input = max(path.stat().st_mtime for path in input_paths)
    oldest_output = min(path.stat().st_mtime for path in output_paths)
    return oldest_output >= newest_input


def audio_container_label(path: Path) -> str:
    try:
        with path.open("rb") as file:
            header = file.read(16)
    except OSError as exc:
        raise PipelineError(f"Could not read source input header: {path} ({exc})") from exc

    if len(header) >= 12 and header[:4] in {b"RIFF", b"RF64"} and header[8:12] == b"WAVE":
        return "wav"
    if len(header) >= 8 and header[4:8] == b"ftyp":
        return "mp4/m4a"
    if header.startswith(b"ID3") or header[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
        return "mp3"
    return "unknown"


def ffmpeg_candidates(svc_dir: Path, explicit: str | None) -> list[Path]:
    if explicit:
        return [resolve_path(explicit, svc_dir)]

    executable = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidates = [
        svc_dir.parent / "ffmpeg" / "bin" / executable,
        svc_dir / "ffmpeg" / "bin" / executable,
        svc_dir.parent / "myvenv" / "Scripts" / executable,
        svc_dir / "myvenv" / "Scripts" / executable,
    ]
    from_path = shutil.which("ffmpeg")
    if from_path:
        candidates.append(Path(from_path))
    imageio_ffmpeg_path = imageio_ffmpeg_candidate()
    if imageio_ffmpeg_path:
        candidates.append(imageio_ffmpeg_path)
    return unique_existing_paths(path.resolve() for path in candidates)


def imageio_ffmpeg_candidate() -> Path | None:
    try:
        import imageio_ffmpeg
    except ImportError:
        return None

    try:
        ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None
    return ffmpeg_path if ffmpeg_path.exists() else None


def choose_ffmpeg(svc_dir: Path, explicit: str | None) -> Path | None:
    candidates = ffmpeg_candidates(svc_dir, explicit)
    if candidates:
        return candidates[0]
    if explicit:
        raise PipelineError(f"FFmpeg executable does not exist: {resolve_path(explicit, svc_dir)}")
    return None


def prepare_demucs_input(
    svc_dir: Path,
    temp_dir: Path,
    input_path: Path,
    explicit_ffmpeg: str | None,
) -> Path:
    safe_input_path = temp_dir / "_demucs_source.wav"
    label = audio_container_label(input_path)

    if label == "wav":
        shutil.copy2(input_path, safe_input_path)
        print_path("[separation] temp source", safe_input_path)
        return safe_input_path

    ffmpeg_path = choose_ffmpeg(svc_dir, explicit_ffmpeg)
    if ffmpeg_path is None:
        suggested_output = input_path.with_name("input.wav")
        raise PipelineError(
            f"Source media container looks like {label}: {input_path}\n"
            "This pipeline needs ffmpeg to decode MP4/M4A/MP3-style inputs into a temporary PCM WAV for Demucs.\n"
            "The final mixed output is still written as WAV by Python.\n"
            "Install ffmpeg, install imageio-ffmpeg in this venv, or convert manually, for example:\n"
            f'  ffmpeg -y -i "{input_path}" -vn -acodec pcm_s16le "{suggested_output}"'
        )

    command = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(safe_input_path),
    ]
    print(f"[separation] source container: {label}; converting with ffmpeg.")
    print(f"[separation] ffmpeg command: {display_command(command)}")
    result = subprocess.run(command, cwd=str(svc_dir))
    if result.returncode != 0:
        raise PipelineError(f"FFmpeg conversion failed with exit code {result.returncode}")
    ensure_wav_file(safe_input_path, "temporary PCM wav")
    print_path("[separation] temp source", safe_input_path)
    return safe_input_path


def run_demucs(
    svc_dir: Path,
    song_dir: Path,
    input_path: Path,
    voice_path: Path,
    bgmusic_path: Path,
    keep_temp: bool,
    explicit_python: str | None,
    explicit_ffmpeg: str | None,
    demucs_device: str,
) -> None:
    temp_dir = song_dir / "_demucs"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    demucs_python = choose_demucs_python(svc_dir, explicit_python)
    demucs_input_path = prepare_demucs_input(
        svc_dir=svc_dir,
        temp_dir=temp_dir,
        input_path=input_path,
        explicit_ffmpeg=explicit_ffmpeg,
    )
    command = [
        str(demucs_python),
        "-m",
        "demucs",
        "-n",
        "htdemucs",
        "--two-stems=vocals",
        "-d",
        demucs_device,
        "-o",
        str(temp_dir),
        str(demucs_input_path),
    ]

    try:
        print(f"[separation] command: {display_command(command)}")
        
        # Keep Demucs model cache under svc/pretrain for this project.
        env = os.environ.copy()
        env["TORCH_HOME"] = str(svc_dir / "pretrain")
        
        result = subprocess.run(command, cwd=str(svc_dir), env=env)
        if result.returncode != 0:
            raise PipelineError(f"Demucs failed with exit code {result.returncode}")

        stem_dir = temp_dir / "htdemucs" / demucs_input_path.stem
        vocals = stem_dir / "vocals.wav"
        no_vocals = stem_dir / "no_vocals.wav"
        if not vocals.exists() or not no_vocals.exists():
            raise PipelineError(
                "Demucs finished, but expected stems were not found:\n"
                f"  vocals: {vocals}\n"
                f"  no_vocals: {no_vocals}"
            )

        voice_path.parent.mkdir(parents=True, exist_ok=True)
        bgmusic_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(vocals, voice_path)
        os.replace(no_vocals, bgmusic_path)
        print_path("[separation] vocal stem", voice_path)
        print_path("[separation] backing stem", bgmusic_path)
    finally:
        if not keep_temp and temp_dir.exists():
            shutil.rmtree(temp_dir)
            print_path("[separation] cleaned temp", temp_dir)
        elif keep_temp and temp_dir.exists():
            print_path("[separation] kept temp", temp_dir)


def run_conversion(
    svc_dir: Path,
    inference_python: Path,
    model_path: Path,
    config_path: Path,
    voice_input_path: Path,
    voice_output_path: Path,
    transpose: int,
    speaker: str,
    f0_predictor: str,
    clip: float,
    wav_format: str,
    device: str,
) -> None:
    ensure_wav_file(voice_input_path, "voice input wav")
    if not inference_python.exists():
        raise PipelineError(f"Inference Python does not exist: {inference_python}")
    if not model_path.exists():
        raise PipelineError(f"Model path does not exist: {model_path}")
    if not config_path.exists():
        raise PipelineError(f"Config path does not exist: {config_path}")
    voice_output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(inference_python),
        str(svc_dir / "inference_main.py"),
        "-m",
        str(model_path),
        "-c",
        str(config_path),
        "-ip",
        str(voice_input_path),
        "-op",
        str(voice_output_path),
        "-t",
        str(transpose),
        "-s",
        speaker,
        "-f0p",
        f0_predictor,
        "-cl",
        str(clip),
        "-wf",
        wav_format,
        "-d",
        device,
    ]

    print(f"[conversion] cwd: {svc_dir}")
    print(f"[conversion] command: {display_command(command)}")
    result = subprocess.run(command, cwd=str(svc_dir))
    if result.returncode != 0:
        raise PipelineError(f"inference_main.py failed with exit code {result.returncode}")
    ensure_wav_file(voice_output_path, "converted voice wav")
    print_path("[conversion] converted voice", voice_output_path)


def import_audio_stack():
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise PipelineError(
            "Mixing requires numpy and soundfile in the Python running svc/main.py. "
            "Run this script with the project venv Python or install the missing package."
        ) from exc
    return np, sf


def resample_voice(voice, source_sr: int, target_sr: int):
    if source_sr == target_sr:
        return voice

    try:
        import librosa

        channels = [
            librosa.resample(voice[:, channel], orig_sr=source_sr, target_sr=target_sr)
            for channel in range(voice.shape[1])
        ]
        import numpy as np

        return np.stack(channels, axis=1).astype("float32", copy=False)
    except ImportError:
        pass

    try:
        import torch
        import torchaudio

        tensor = torch.from_numpy(voice.T)
        resampled = torchaudio.functional.resample(
            tensor, orig_freq=source_sr, new_freq=target_sr
        )
        return resampled.numpy().T.astype("float32", copy=False)
    except ImportError as exc:
        raise PipelineError(
            "Voice and backing sample rates differ, and neither librosa nor torchaudio "
            "is available for resampling."
        ) from exc


def match_channels(np, voice, target_channels: int):
    voice_channels = voice.shape[1]
    if voice_channels == target_channels:
        return voice
    if voice_channels == 1:
        return np.repeat(voice, target_channels, axis=1)

    mono_voice = voice.mean(axis=1, keepdims=True)
    if target_channels == 1:
        return mono_voice
    return np.repeat(mono_voice, target_channels, axis=1)


def align_length(np, voice, target_length: int):
    voice_length = voice.shape[0]
    if voice_length == target_length:
        return voice
    if voice_length > target_length:
        return voice[:target_length, :]

    pad_width = ((0, target_length - voice_length), (0, 0))
    return np.pad(voice, pad_width, mode="constant")


def mix_audio(
    bgmusic_path: Path,
    voice_path: Path,
    output_path: Path,
    background_gain: float,
    voice_gain: float,
) -> None:
    ensure_wav_file(bgmusic_path, "backing wav")
    ensure_wav_file(voice_path, "converted voice wav")
    if background_gain < 0:
        raise PipelineError(f"Background gain must be non-negative: {background_gain}")
    if voice_gain < 0:
        raise PipelineError(f"Voice gain must be non-negative: {voice_gain}")

    np, sf = import_audio_stack()

    backing, backing_sr = sf.read(bgmusic_path, dtype="float32", always_2d=True)
    voice, voice_sr = sf.read(voice_path, dtype="float32", always_2d=True)

    if backing.size == 0:
        raise PipelineError(f"Backing wav is empty: {bgmusic_path}")
    if voice.size == 0:
        raise PipelineError(f"Converted voice wav is empty: {voice_path}")

    voice = resample_voice(voice, voice_sr, backing_sr)
    voice = match_channels(np, voice, backing.shape[1])
    voice = align_length(np, voice, backing.shape[0])

    backing = backing * background_gain
    voice = voice * voice_gain
    mixed = backing + voice
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, mixed, backing_sr, format="WAV", subtype="PCM_16")
    print_path("[mix] output", output_path)
    print(f"[mix] gains: background={background_gain:.3f}, voice={voice_gain:.3f}")
    print(f"[mix] sample_rate: {backing_sr}, channels: {backing.shape[1]}, peak: {min(peak, 0.99):.6f}")


def print_path(label: str, path: Path) -> None:
    print(f"{label}: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end SVC audio pipeline: separate, convert, and mix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--song-dir", "--song_dir", default=DEFAULT_SONG_DIR, help="Song directory, resolved from svc root when relative.")
    parser.add_argument("--input", default=None, help="Source song media. If omitted, auto-detects input.* or a single non-generated media file.")
    parser.add_argument("--output", default=None, help="Final mixed wav path. Defaults to song_dir/output.wav.")

    parser.add_argument("--model-path", "--model_path", default="logs/44k_denoise/G_163200.pth", help="Model path passed to inference_main.py -m.")
    parser.add_argument("--config-path", "--config_path", default="configs/config.json", help="Config path passed to inference_main.py -c.")
    parser.add_argument("--speaker", default="myvoice_denoise_mono", help="Speaker passed to inference_main.py -s.")
    parser.add_argument("--transpose", type=int, default=-2, help="Transpose semitones passed to inference_main.py -t.")
    parser.add_argument("--f0-predictor", "--f0_predictor", default="rmvpe", help="F0 predictor passed to inference_main.py -f0p.")
    parser.add_argument("--clip", type=float, default=15, help="Clip seconds passed to inference_main.py -cl.")
    parser.add_argument("--device", default="cpu", help="Device passed to inference_main.py -d.")
    parser.add_argument("--wav-format", "--wav_format", default="wav", help="Voice output format passed to inference_main.py -wf.")

    parser.add_argument("--voice-input", "--voice_input", default=None, help="Vocal stem path passed to inference_main.py -ip. Defaults to song_dir/wav_voice.wav.")
    parser.add_argument("--voice-output", "--voice_output", default=None, help="Converted voice path passed to inference_main.py -op. Defaults to song_dir/my_voice.wav.")
    parser.add_argument("--background-output", "--background_output", default=None, help="Backing stem path. Defaults to song_dir/wav_bgmusic.wav.")
    parser.add_argument("--voice-gain", "--voice_gain", type=float, default=DEFAULT_VOICE_GAIN, help="Converted voice volume multiplier for final mixing.")
    parser.add_argument("--background-gain", "--background_gain", type=float, default=DEFAULT_BACKGROUND_GAIN, help="Backing track volume multiplier for final mixing.")
    parser.add_argument("--demucs-python", "--demucs_python", default=None, help="Explicit Python executable for python -m demucs.")
    parser.add_argument("--demucs-device", "--demucs_device", default=DEFAULT_DEMUCS_DEVICE, help="Device passed to Demucs separation.")
    parser.add_argument("--inference-python", "--inference_python", default=None, help="Explicit Python executable for inference_main.py.")
    parser.add_argument("--ffmpeg", default=None, help="Explicit ffmpeg executable for decoding MP4/M4A/MP3-style inputs to temporary PCM WAV.")

    parser.add_argument("--force", action="store_true", help="Regenerate steps even when outputs look current.")
    parser.add_argument("--keep-temp", "--keep_temp", action="store_true", help="Keep song_dir/_demucs after separation.")
    parser.add_argument("--skip-separation", "--skip_separation", action="store_true", help="Use existing vocal and backing stems.")
    parser.add_argument("--skip-conversion", "--skip_conversion", action="store_true", help="Use existing converted voice wav.")
    return parser


def run(args: argparse.Namespace) -> None:
    svc_dir = svc_root()
    song_dir = resolve_path(args.song_dir, svc_dir)
    input_path = (
        resolve_input_arg(args.input, svc_dir, song_dir)
        if args.input or not args.skip_separation
        else None
    )

    voice_input_path = resolve_path_or_none(args.voice_input, svc_dir) or (song_dir / "wav_voice.wav")
    bgmusic_path = resolve_path_or_none(args.background_output, svc_dir) or (song_dir / "wav_bgmusic.wav")
    voice_output_path = resolve_path_or_none(args.voice_output, svc_dir) or (song_dir / "my_voice.wav")
    final_output_path = resolve_path_or_none(args.output, svc_dir) or (song_dir / "output.wav")

    model_path = resolve_path(args.model_path, svc_dir)
    config_path = resolve_path(args.config_path, svc_dir)
    inference_python = default_inference_python(svc_dir, args.inference_python)

    print_path("[setup] svc root", svc_dir)
    print_path("[setup] song dir", song_dir)
    if input_path is not None:
        print_path("[setup] source input", input_path)
    else:
        print("[setup] source input: skipped")

    if args.skip_separation:
        ensure_wav_file(voice_input_path, "vocal stem")
        ensure_wav_file(bgmusic_path, "backing stem")
        print_path("[separation] skipped vocal stem", voice_input_path)
        print_path("[separation] skipped backing stem", bgmusic_path)
    elif input_path is not None and not args.force and is_current([voice_input_path, bgmusic_path], [input_path]):
        print("[separation] outputs are current; use --force to regenerate.")
        print_path("[separation] vocal stem", voice_input_path)
        print_path("[separation] backing stem", bgmusic_path)
    else:
        if input_path is None:
            raise PipelineError("Separation requires a source input wav.")
        run_demucs(
            svc_dir=svc_dir,
            song_dir=song_dir,
            input_path=input_path,
            voice_path=voice_input_path,
            bgmusic_path=bgmusic_path,
            keep_temp=args.keep_temp,
            explicit_python=args.demucs_python,
            explicit_ffmpeg=args.ffmpeg,
            demucs_device=args.demucs_device,
        )

    if args.skip_conversion:
        ensure_wav_file(voice_output_path, "converted voice wav")
        print_path("[conversion] skipped converted voice", voice_output_path)
    elif not args.force and is_current([voice_output_path], [voice_input_path]):
        print("[conversion] output is current; use --force to regenerate.")
        print_path("[conversion] converted voice", voice_output_path)
    else:
        run_conversion(
            svc_dir=svc_dir,
            inference_python=inference_python,
            model_path=model_path,
            config_path=config_path,
            voice_input_path=voice_input_path,
            voice_output_path=voice_output_path,
            transpose=args.transpose,
            speaker=args.speaker,
            f0_predictor=args.f0_predictor,
            clip=args.clip,
            wav_format=args.wav_format,
            device=args.device,
        )

    mix_audio(
        bgmusic_path,
        voice_output_path,
        final_output_path,
        background_gain=args.background_gain,
        voice_gain=args.voice_gain,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ERROR: Interrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
