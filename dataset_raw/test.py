from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_WAV_SLI_DIR = ROOT_DIR / "wav_sli"
SRC_WAV_PRE_DIR = ROOT_DIR / "wav_pre"
DEST_MYVOICE_DIR = ROOT_DIR / "myvoice"
OUT_INDEX_JSON = DEST_MYVOICE_DIR / "song_segments_index.json"


SLICED_WAV_RE = re.compile(r"^(?P<song>\d+)_(?P<segment>\d+)\.wav$", re.IGNORECASE)
PRE_WAV_RE = re.compile(r"^样本(?P<song>\d+)\s*(?P<name>.+?)\.wav$", re.IGNORECASE)


def build_song_name_map(pre_dir: Path) -> dict[int, str]:
	song_name_map: dict[int, str] = {}
	if not pre_dir.exists():
		return song_name_map

	for wav_path in pre_dir.iterdir():
		if not (wav_path.is_file() and wav_path.suffix.lower() == ".wav"):
			continue
		match = PRE_WAV_RE.match(wav_path.name)
		if not match:
			continue
		song_no = int(match.group("song"))
		song_name = match.group("name").strip()
		if song_name:
			song_name_map[song_no] = song_name
	return song_name_map


def group_sliced_wavs(src_dir: Path) -> dict[int, list[tuple[int, Path]]]:
	grouped: dict[int, list[tuple[int, Path]]] = {}
	for wav_path in src_dir.iterdir():
		if not (wav_path.is_file() and wav_path.suffix.lower() == ".wav"):
			continue
		match = SLICED_WAV_RE.match(wav_path.name)
		if not match:
			continue
		song_no = int(match.group("song"))
		seg_no = int(match.group("segment"))
		grouped.setdefault(song_no, []).append((seg_no, wav_path))

	for song_no, items in grouped.items():
		items.sort(key=lambda x: x[0])
		seg_nos = [seg for seg, _ in items]
		if len(seg_nos) != len(set(seg_nos)):
			raise RuntimeError(f"歌曲 {song_no:02d} 存在重复片段序号: {seg_nos}")

	return dict(sorted(grouped.items(), key=lambda x: x[0]))


def main() -> None:
	if not SRC_WAV_SLI_DIR.exists():
		raise FileNotFoundError(f"找不到目录: {SRC_WAV_SLI_DIR}")

	DEST_MYVOICE_DIR.mkdir(parents=True, exist_ok=True)

	grouped = group_sliced_wavs(SRC_WAV_SLI_DIR)
	if not grouped:
		raise RuntimeError(f"在 {SRC_WAV_SLI_DIR} 未找到形如 01_01.wav 的切片文件")

	song_name_map = build_song_name_map(SRC_WAV_PRE_DIR)

	total_segments = sum(len(items) for items in grouped.values())
	number_width = max(4, len(str(total_segments)))

	# 复制并重命名：0001.wav, 0002.wav ...
	current_index = 1
	songs_index: list[dict[str, object]] = []

	for song_no, seg_items in grouped.items():
		start_index = current_index
		for _, wav_path in seg_items:
			new_name = f"{current_index:0{number_width}d}.wav"
			dest_path = DEST_MYVOICE_DIR / new_name
			shutil.copy2(wav_path, dest_path)
			current_index += 1
		end_index = current_index - 1

		song_code = f"{song_no:02d}"
		song_name = song_name_map.get(song_no, f"样本{song_code}")
		songs_index.append(
			{
				"song_no": song_no,
				"song_name": song_name,
				"segment_count": len(seg_items),
				"start_index": start_index,
				"end_index": end_index,
			}
		)

	index_obj = {
		"created_at": datetime.now(timezone.utc).isoformat(),
		"source_wav_sli_dir": str(SRC_WAV_SLI_DIR),
		"dest_myvoice_dir": str(DEST_MYVOICE_DIR),
		"number_width": number_width,
		"total_segments": total_segments,
		"songs": songs_index,
	}

	OUT_INDEX_JSON.write_text(
		json.dumps(index_obj, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)

	print(f"完成: 共复制 {total_segments} 段到 {DEST_MYVOICE_DIR}")
	print(f"索引: {OUT_INDEX_JSON}")


if __name__ == "__main__":
	main()
