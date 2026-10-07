"""Generate photoreal B-roll clips via DashScope Wan text-to-video.

Uses DASHSCOPE_API_KEY already on this machine. Clips are anonymous
synthesised people (not corpus participants, not author likenesses),
no brand logos, no watermark. 16:9, then upscaled to 1920x1080 in
make_video.py if needed.

Default model: wan2.2-t2v-plus (5 s, 1080p). Falls back to
wan2.1-t2v-turbo 720p if the plus model is rejected.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

from dashscope import VideoSynthesis

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "ieeevr" / "video_assets" / "broll"
NEG = (
    "logo, brand name, trademark, watermark, subtitle, celebrity, cartoon, "
    "anime, deformed hands, extra fingers, low quality, flicker, "
    "VR motion controller, Oculus Touch, tracking ring, dual wands, gamepad, "
    "Xbox controller, smartwatch face, Apple Watch, Fitbit, glowing watch screen, "
    "sports sweatband, fitness tracker with a display, "
    "blood, gore, corpse, graphic wound, gun pointed at camera, "
    "copyrighted game, Call of Duty, Resident Evil, identifiable IP"
)
# Old generic VR-gameplay clips are kept on disk but no longer generated.
CLIPS = [
    {
        "id": "lab_affect",
        "prompt": (
            "Photoreal scientific documentary, 16:9. Seated adult in a psychology "
            "lab wearing a generic black VR headset with a cable. Left wrist wears "
            "a research physiology band: small black rectangular pod, metal skin "
            "electrodes on the inner wrist, tiny green PPG light, NO watch screen. "
            "Right hand rests on a small lab joystick: short stick on a square box, "
            "NOT a VR game controller, NO tracking ring. Two monitors: left shows "
            "EDA and BVP waveforms; right shows a 360-degree equirectangular room "
            "with a yellow gaze crosshair. Slow push-in. No logos."
        ),
    },
    {
        "id": "eye_cameras",
        "prompt": (
            "Photoreal documentary close-up, 16:9. A researcher places a generic "
            "black VR headset onto a seated participant. Underside of the headset "
            "shows small infrared eye-tracking cameras and IR LEDs facing the eyes. "
            "A nearby monitor shows a live grayscale eye image and a gaze point on "
            "a 360-degree panoramic landscape. Hands adjust the strap. No brand "
            "logos, no readable UI words."
        ),
    },
    {
        "id": "e4_wrist",
        "prompt": (
            "Photoreal macro shot, 16:9. Inner wrist of a VR participant. A black "
            "research wearable like an Empatica E4: rectangular black electronics "
            "pod on the top of the wrist, two round metal electrodermal electrodes "
            "touching skin, a tiny green optical pulse LED glowing. NO watch face, "
            "NO numbers on a screen, NOT Apple Watch, NOT Fitbit. Out-of-focus "
            "black VR headset at the top of frame. Shallow depth of field."
        ),
    },
    {
        "id": "joystick_rating",
        "prompt": (
            "Photoreal medium close-up, 16:9. Right hand of a person in a VR "
            "headset holding a laboratory emotion-rating joystick: a short black "
            "stick on a small square black base, thumb moving the stick slightly "
            "as if rating valence and arousal. NOT a VR motion controller, NO "
            "tracking ring, only one stick. Background monitor: 360-degree forest "
            "and a simple circular affect circumplex with a moving dot. Documentary."
        ),
    },
    {
        "id": "gaze_heatmap",
        "prompt": (
            "Photoreal over-the-shoulder, 16:9. A lab monitor fills most of the "
            "frame: a 360-degree equirectangular video of a city street with a "
            "semi-transparent red-yellow gaze heatmap and a moving gaze cursor. "
            "In the foreground, slightly blurred, a seated person wears a VR "
            "headset and turns their head. Scientific eye-tracking visualization. "
            "No logos, no readable brand UI."
        ),
    },
    {
        "id": "physio_scope",
        "prompt": (
            "Photoreal close-up of a lab monitor, 16:9. Screen shows three "
            "scrolling scientific waveforms: a slow electrodermal skin-conductance "
            "trace, a pulsatile blood-volume pulse, and a heart-rate line, in "
            "green, red, and blue. A small inset in the corner shows a 360 VR "
            "scene. A wrist with a black electrode band is visible at the bottom "
            "of frame. No brand logos, waveforms look like research software."
        ),
    },
    {
        "id": "pupil_ir",
        "prompt": (
            "Photoreal extreme close-up, 16:9, infrared eye-tracking look. A human "
            "eye filling the frame, dark iris, a bright corneal IR glint from an "
            "eye tracker, pupil slightly dilating, the person looking left then "
            "right. Faint reflection of a VR headset rim. Scientific, not horror, "
            "not cinematic makeup. No logos."
        ),
    },
    {
        "id": "head_look",
        "prompt": (
            "Photoreal documentary, 16:9. Seated participant in a generic black VR "
            "headset slowly turns head left, then up, then right, as if inspecting "
            "a 360-degree virtual scene. Black physiology band on the left wrist "
            "with no watch screen. Right hand on a small square-base lab joystick. "
            "Research lab, window light. Head kinematics and looking behavior. "
            "No VR wands, no tracking rings, no logos."
        ),
    },
    {
        "id": "two_ratings",
        "prompt": (
            "Photoreal research lab, 16:9. Two seated adults side by side in "
            "matching generic VR headsets, both watching the same 360 forest on "
            "monitors behind them. Each holds a small square-base lab joystick, "
            "NOT VR game controllers. The left person's joystick is pushed "
            "forward (high arousal); the right person's joystick is pulled back "
            "and left (low valence). Black electrode wristbands, no watch faces. "
            "Same stimulus, different emotion ratings. Slow dolly. No logos."
        ),
    },
    {
        "id": "desktop_rating",
        "prompt": (
            "Photoreal documentary, 16:9. Desktop affective-computing setup, NO "
            "VR headset. A person watches an abstract colorful film on a monitor. "
            "Left wrist: black research EDA/PPG band with metal electrodes, no "
            "watch screen. Right hand on a small lab joystick rating emotion "
            "continuously. Scientific, not gaming. No logos, no copyrighted movie."
        ),
    },
    {
        "id": "therapy_sensor",
        "prompt": (
            "Photoreal documentary, 16:9. A woman sits in a quiet consulting room "
            "wearing a generic black VR headset for exposure therapy. A black "
            "physiological wristband with visible metal electrodes is on her "
            "wrist. One small lab joystick in her right hand for rating distress, "
            "NOT two VR controllers. Soft daylight, plant, calm. Slow lateral "
            "move. No hospital logos."
        ),
    },
    {
        "id": "researcher_dash",
        "prompt": (
            "Photoreal over-the-shoulder of a researcher, 16:9. Dual monitors: "
            "left has EDA and pulse waveforms plus a binary emotion classifier "
            "output as abstract colored bars; right has 360 video with a gaze "
            "heatmap. Behind the desk, a participant wears a VR headset. The "
            "researcher watches the screens, not the participant. Lab lighting. "
            "No institution names, no logos."
        ),
    },
    {
        "id": "pov_horror",
        "prompt": (
            "Photoreal first-person 360 VR horror stimulus, 16:9, as seen "
            "through a headset. Dark abandoned hospital corridor, flickering "
            "fluorescent lights, fog on the floor, a door slowly opening at "
            "the far end. Atmospheric scare, NO blood, NO gore, NO monsters "
            "in close-up, NO jump-scare face. Cinematic but scientific "
            "stimulus clip. Slow forward drift. No logos, no subtitles."
        ),
    },
    {
        "id": "pov_combat",
        "prompt": (
            "Photoreal first-person VR combat-training stimulus, 16:9, as "
            "seen through a headset. Sci-fi arena at dusk, two armored "
            "training robots sparring with staffs in the middle distance, "
            "sparks, dust, the camera weaves slightly as a player would. "
            "Game-like, PG-13, NO blood, NO firearms, NO gore, NOT a "
            "copyrighted title. No HUD text, no logos."
        ),
    },
    {
        "id": "pov_coaster",
        "prompt": (
            "Photoreal first-person 360 VR high-arousal stimulus, 16:9. "
            "Front seat of a wooden roller coaster cresting a drop, wind, "
            "sunset sky, track plunging ahead. Exciting, not horror. No "
            "faces of riders in close-up, no logos, no park names."
        ),
    },
    {
        "id": "hmd_horror",
        "prompt": (
            "Photoreal psychology lab, 16:9. A seated adult in a generic "
            "black VR headset flinches and turns the head as if startled. "
            "Behind them a large monitor shows a dark 360-degree haunted "
            "corridor with fog. Left wrist: black research EDA band with "
            "metal electrodes, green PPG light, NO watch screen. Right "
            "hand on a small square-base lab joystick, NOT a VR wand. "
            "Documentary lighting. No logos."
        ),
    },
    {
        "id": "hmd_combat",
        "prompt": (
            "Photoreal psychology lab, 16:9. Seated participant in a generic "
            "black VR headset, head turning quickly left-right as if in a "
            "fight. Monitor behind shows a 360-degree sci-fi arena with two "
            "robots sparring. Black physiology wristband, one lab joystick "
            "on a square base, NO motion controllers. High arousal, "
            "scientific, not a gaming stream. No logos."
        ),
    },
    {
        "id": "stim_monitor",
        "prompt": (
            "Photoreal close-up of a lab monitor, 16:9. The screen is split: "
            "left a 360 equirectangular dark forest horror path with a "
            "yellow gaze crosshair; right a 360 combat arena. Waveforms of "
            "EDA and heart rate overlay the bottom. A blurred VR headset "
            "in the foreground. Scientific stimulus dashboard. No brand UI, "
            "no readable English words."
        ),
    },
    {
        "id": "clip_octants",
        "prompt": (
            "Photoreal documentary, 16:9. A researcher stands before a "
            "wall of eight TV monitors, each playing a different 360 VR "
            "emotion clip: dark horror corridor, robot arena fight, "
            "roller coaster, calm forest, city street, ocean, fire, "
            "night alley. The researcher points at the horror screen then "
            "the fight screen. Psychology lab. No institution logos."
        ),
    },
    {
        "id": "two_stimuli",
        "prompt": (
            "Photoreal research lab, 16:9. Two seated adults in matching "
            "generic VR headsets. Left person's monitor shows a dark "
            "horror corridor; right person's monitor shows a combat arena. "
            "Both wear electrode wristbands and hold square-base lab "
            "joysticks, NOT VR wands. Same evaluation protocol, different "
            "360 stimuli. Slow dolly. No logos."
        ),
    },
]


def _submit(clip: dict, model: str, size: str):
    print(f"submit {clip['id']}  {model}  {size}", flush=True)
    return VideoSynthesis.async_call(
        model=model,
        prompt=clip["prompt"],
        negative_prompt=NEG,
        size=size,
        watermark=False,
        prompt_extend=False,
        seed=2468 + (sum(ord(c) for c in clip["id"]) % 1000),
    )


def _video_url(result) -> str | None:
    out = getattr(result, "output", None)
    if out is None:
        return None
    if isinstance(out, dict):
        return out.get("video_url") or (out.get("results") or {}).get("video_url")
    return getattr(out, "video_url", None)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def _ok(rsp) -> bool:
    code = getattr(rsp, "status_code", None)
    return code in (200, 201, None) or str(code) in ("200", "201")


def main() -> None:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set")
    OUT.mkdir(parents=True, exist_ok=True)
    # China mainland: wan2.2-t2v-plus works; turbo id is wanx2.1-t2v-turbo.
    # Concurrent cap is about 2 tasks; QPS about 2.
    models = [
        ("wan2.2-t2v-plus", "1920*1080"),
        ("wanx2.1-t2v-plus", "1280*720"),
        ("wanx2.1-t2v-turbo", "1280*720"),
    ]
    pending = []
    for clip in CLIPS:
        dest = OUT / f"{clip['id']}.mp4"
        if dest.exists() and dest.stat().st_size > 50_000:
            print(f"skip existing {dest.name}", flush=True)
            continue
        pending.append((clip, dest))

    meta = []
    i = 0
    while i < len(pending):
        batch = pending[i : i + 2]
        jobs = []
        for clip, dest in batch:
            rsp = None
            used = models[0]
            for model, size in models:
                time.sleep(0.6)
                rsp = _submit(clip, model, size)
                if _ok(rsp):
                    used = (model, size)
                    break
                print(
                    f"  reject {model}: {getattr(rsp, 'status_code', None)} "
                    f"{getattr(rsp, 'code', '')} {getattr(rsp, 'message', '')}",
                    flush=True,
                )
            if rsp is None or not _ok(rsp):
                meta.append({"id": clip["id"], "ok": False, "status": "submit_failed"})
                continue
            jobs.append((clip, rsp, dest, used))
        for clip, rsp, dest, used in jobs:
            print(f"wait {clip['id']} ...", flush=True)
            t0 = time.time()
            result = VideoSynthesis.wait(rsp)
            elapsed = time.time() - t0
            url = _video_url(result)
            status = getattr(getattr(result, "output", None), "task_status", None)
            if not url:
                raw = getattr(result, "output", None)
                print(f"  FAIL {clip['id']} status={status} output={raw}", flush=True)
                meta.append({"id": clip["id"], "ok": False, "status": str(status), "elapsed_s": round(elapsed, 1)})
                continue
            _download(url, dest)
            print(f"  wrote {dest}  {dest.stat().st_size/1e6:.1f} MB  {elapsed:.0f}s", flush=True)
            meta.append({"id": clip["id"], "ok": True, "path": dest.name, "elapsed_s": round(elapsed, 1), "model": used[0], "size": used[1]})
        i += 2

    (OUT / "manifest.json").write_text(
        json.dumps({"clips": meta}, indent=2) + "\n", encoding="utf-8"
    )
    print("done", json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
