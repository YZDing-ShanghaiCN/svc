Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ScriptDir "..\myvenv\Scripts\python.exe"
$Demucs = Join-Path $ScriptDir "..\myvenv\Scripts\demucs.exe"
$SongDir = "wav/test"

Push-Location $ScriptDir
try {

& $Demucs --two-stems=vocals -o "$SongDir/_demucs" "$SongDir/input.wav"

Move-Item -Force "$SongDir/_demucs/htdemucs/input/vocals.wav" "$SongDir/wav_voice.wav"
Move-Item -Force "$SongDir/_demucs/htdemucs/input/no_vocals.wav" "$SongDir/wav_bgmusic.wav"

Remove-Item -Recurse -Force "$SongDir/_demucs"

& $Python inference_main.py `
  -m logs/44k_denoise/G_163200.pth `
  -c configs/config.json `
  -ip "$SongDir/wav_voice.wav" `
  -op "$SongDir/my_voice.wav" `
  -t -2 `
  -s myvoice_denoise_mono `
  -f0p rmvpe `
  -cl 15 `
  -wf wav `
  -d cpu
}
finally {
  Pop-Location
}
