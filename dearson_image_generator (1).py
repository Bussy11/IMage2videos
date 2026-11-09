"""
DEARson Image Generator
-----------------------
Pages:
  1. Image Generator  - text -> image (free, no API key: Pollinations.ai)
  2. Image to Video   - upload (or generate) an image -> MP4 (runs locally, always works)
  3. Text to Audio    - free TTS (Edge voices / gTTS) + voice-cloning buttons

Run:  streamlit run dearson_image_generator.py
"""
import asyncio
import io
import os
import random
import tempfile
from urllib.parse import quote

import requests
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG (must be first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="DEARson Image Generator",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# STYLE: colourful buttons, ALL IN ONE LINE (even on a phone)
# ----------------------------------------------------------------------------
BUTTON_COLORS = {
    "gen": "linear-gradient(135deg,#ff512f,#dd2476)",
    "regen": "linear-gradient(135deg,#f7971e,#ffd200)",
    "clear": "linear-gradient(135deg,#434343,#000000)",
    "mkvid": "linear-gradient(135deg,#11998e,#38ef7d)",
    "newimg": "linear-gradient(135deg,#8e2de2,#4a00e0)",
    "resetvid": "linear-gradient(135deg,#434343,#000000)",
    "speak": "linear-gradient(135deg,#00c6ff,#0072ff)",
    "clone": "linear-gradient(135deg,#f953c6,#b91d73)",
    "speakclone": "linear-gradient(135deg,#fc4a1a,#f7b733)",
    "clearclone": "linear-gradient(135deg,#434343,#000000)",
}

_css_btns = "\n".join(
    f".st-key-{k} button{{background:{v} !important;color:#fff !important;border:none !important;"
    f"font-weight:700 !important;border-radius:14px !important;width:100%;"
    f"padding:.55rem .2rem !important;font-size:.82rem !important;"
    f"box-shadow:0 4px 12px rgba(0,0,0,.25) !important;transition:transform .15s;}}"
    f".st-key-{k} button:hover{{transform:translateY(-2px) scale(1.03);}}"
    f".st-key-{k} button p{{color:#fff !important;font-weight:700;}}"
    for k, v in BUTTON_COLORS.items()
)

st.markdown(
    f"""
<style>
/* keep columns side by side on phones */
div[data-testid="stHorizontalBlock"]{{flex-wrap:nowrap !important;gap:.4rem !important;}}
div[data-testid="stHorizontalBlock"] > div{{min-width:0 !important;flex:1 1 0 !important;}}
.block-container{{padding-top:1.2rem;max-width:760px;}}
.dear-title{{text-align:center;font-size:2rem;font-weight:900;margin:0;
  background:linear-gradient(90deg,#ff512f,#dd2476,#8e2de2,#00c6ff);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
.dear-sub{{text-align:center;opacity:.7;margin-bottom:.8rem;font-size:.9rem;}}
div[data-testid="stImage"] img, video{{border-radius:16px;}}
{_css_btns}
</style>
<h1 class="dear-title">DEARson Image Generator</h1>
<div class="dear-sub">Create images • Turn them into video • Turn text into voice</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# CACHED RESOURCES  (speed: reuse connections, avoid repeat downloads)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "DEARson-Image-Generator/1.0"})
    return s


STYLES = {
    "None": "",
    "Realistic photo": ", ultra realistic photo, 8k, sharp focus, natural lighting",
    "Cinematic": ", cinematic lighting, dramatic, film still, depth of field",
    "Anime": ", anime style, vibrant colors, detailed illustration",
    "3D render": ", 3d render, octane, soft shadows, high detail",
    "Digital art": ", digital painting, concept art, trending on artstation",
    "Logo / flyer": ", clean graphic design, bold typography, flat vector style",
}
SIZES = {
    "Square 1:1": (768, 768),
    "Portrait 9:16": (576, 1024),
    "Landscape 16:9": (1024, 576),
}


@st.cache_data(ttl=3600, max_entries=40, show_spinner=False)
def fetch_image(prompt: str, w: int, h: int, seed: int, model: str, hf_token: str = "") -> bytes:
    """Text -> image bytes. Pollinations (free, no key) first, Hugging Face as backup."""
    last_err = None
    url = f"https://image.pollinations.ai/prompt/{quote(prompt)}"
    params = {"width": w, "height": h, "seed": seed, "model": model, "nologo": "true"}
    for _ in range(3):
        try:
            r = get_session().get(url, params=params, timeout=90)
            if r.ok and r.headers.get("content-type", "").startswith("image"):
                return r.content
            last_err = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
    if hf_token:
        try:
            from huggingface_hub import InferenceClient

            img = InferenceClient(token=hf_token).text_to_image(
                prompt, model="black-forest-labs/FLUX.1-schnell"
            )
            buf = io.BytesIO()
            img.save(buf, "PNG")
            return buf.getvalue()
        except Exception as e:  # noqa: BLE001
            last_err = f"{last_err} | HF: {e}"
    raise RuntimeError(f"Image generation failed: {last_err}")


def shrink(img_bytes: bytes, max_side: int = 1280) -> bytes:
    """Downscale big phone photos so everything runs faster."""
    from PIL import Image

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# IMAGE -> VIDEO  (local Ken-Burns engine: no API, no token, never fails)
# ----------------------------------------------------------------------------
VIDEO_SIZES = {"Square 1:1": (640, 640), "Portrait 9:16": (480, 848), "Landscape 16:9": (848, 480)}
EFFECTS = ["Zoom in", "Zoom out", "Pan left", "Pan right", "Pan up", "Pan down", "Zoom + pan"]


@st.cache_data(ttl=1800, max_entries=6, show_spinner=False)
def make_video(img_bytes: bytes, effect: str, seconds: int, fps: int, size: tuple) -> bytes:
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    W, H = size
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    sw, sh = img.size
    aspect = W / H
    if sw / sh > aspect:
        bh, bw = sh, sh * aspect
    else:
        bw, bh = sw, sw / aspect

    n = max(2, seconds * fps)
    path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=8)
    try:
        for i in range(n):
            p = i / (n - 1)
            p = p * p * (3 - 2 * p)  # smooth easing
            z, dx, dy = 1.0, 0.0, 0.0
            if effect == "Zoom in":
                z = 1 + 0.35 * p
            elif effect == "Zoom out":
                z = 1.35 - 0.35 * p
            elif effect in ("Pan left", "Pan right"):
                z = 1.25
                dx = (1 - 2 * p) * (1 if effect == "Pan left" else -1)
            elif effect in ("Pan up", "Pan down"):
                z = 1.25
                dy = (1 - 2 * p) * (1 if effect == "Pan up" else -1)
            else:  # Zoom + pan
                z = 1.05 + 0.3 * p
                dx = 0.6 * (1 - 2 * p)
            cw, ch = bw / z, bh / z
            cx = sw / 2 + dx * max(0, (sw - cw) / 2)
            cy = sh / 2 + dy * max(0, (sh - ch) / 2)
            box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
            frame = img.resize((W, H), Image.BICUBIC, box=box)
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return data


# ---- Real AI motion (people actually move) via free LTX-Video Space on Hugging Face ----
LTX_SPACE = "Lightricks/ltx-video-distilled"
LTX_SIZES = {"Landscape 16:9": (704, 512), "Portrait 9:16": (512, 704), "Square 1:1": (640, 640)}
CINEMA = ("cinematic lighting, natural realistic human movement, lifelike faces and expressions, "
          "smooth stable camera, shallow depth of field, warm film color grading, 35mm film look, high detail")
NEG = "worst quality, blurry, jittery, distorted faces, deformed hands, extra limbs, inconsistent motion, morphing, text, watermark"
DEFAULT_SCENE = "The people in the scene come alive, smiling, breathing and moving naturally"


@st.cache_resource(ttl=3000, show_spinner=False)
def get_client(space: str, token: str):
    from gradio_client import Client

    if token:
        try:
            return Client(space, hf_token=token)
        except TypeError:
            return Client(space, token=token)
    return Client(space)


def ltx_clip(img_bytes: bytes, prompt: str, seconds: float, size: tuple, token: str, space: str) -> bytes:
    """One AI clip from one image. People really move, smile, dance (prompt-driven)."""
    from gradio_client import handle_file

    W, H = size
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.write(img_bytes)
    tmp.close()
    client = get_client(space, token)
    try:
        try:
            res = client.predict(
                prompt=prompt, negative_prompt=NEG, input_image_filepath=handle_file(tmp.name),
                height_ui=H, width_ui=W, mode="image-to-video", duration_ui=float(seconds),
                api_name="/image_to_video",
            )
        except Exception as e1:  # noqa: BLE001
            if any(k in str(e1).lower() for k in ("quota", "gpu", "queue", "exceeded")):
                raise
            res = client.predict(
                prompt, NEG, handle_file(tmp.name), None, H, W, "image-to-video",
                float(seconds), 9, 42, True, 1.0, True, api_name="/image_to_video",
            )
    finally:
        os.remove(tmp.name)
    out = res[0] if isinstance(res, (list, tuple)) else res
    for _ in range(2):
        if isinstance(out, dict):
            out = out.get("video") or out.get("path") or out
    if not isinstance(out, str):
        raise RuntimeError("The video Space returned no video.")
    with open(out, "rb") as f:
        return f.read()


def last_frame(video_bytes: bytes) -> bytes:
    """Last frame of a clip: used so the next scene continues from where this one ended."""
    import imageio.v2 as imageio
    from PIL import Image

    p = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    p.write(video_bytes)
    p.close()
    last = None
    reader = imageio.get_reader(p.name)
    for fr in reader:
        last = fr
    reader.close()
    os.remove(p.name)
    buf = io.BytesIO()
    Image.fromarray(last).save(buf, "JPEG", quality=92)
    return buf.getvalue()


def join_clips(clips: list) -> bytes:
    """Stitch scenes into one video."""
    if len(clips) == 1:
        return clips[0]
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = imageio.get_writer(out_path, fps=30, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
    size = None
    for c in clips:
        p = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        p.write(c)
        p.close()
        reader = imageio.get_reader(p.name)
        for fr in reader:
            if size is None:
                size = (fr.shape[1], fr.shape[0])
            if (fr.shape[1], fr.shape[0]) != size:
                fr = np.asarray(Image.fromarray(fr).resize(size))
            writer.append_data(fr)
        reader.close()
        os.remove(p.name)
    writer.close()
    with open(out_path, "rb") as f:
        data = f.read()
    os.remove(out_path)
    return data


def friendly_error(e: Exception) -> str:
    msg = str(e)
    if any(k in msg.lower() for k in ("quota", "exceeded", "gpu")):
        return ("The free GPU time is used up right now. Paste a free Hugging Face token under "
                "⚙️ Settings (your own daily quota is used), or try again in a few hours.")
    return msg


# ----------------------------------------------------------------------------
# TEXT -> AUDIO
# ----------------------------------------------------------------------------
FALLBACK_VOICES = {
    "English (US) - Jenny (F)": "en-US-JennyNeural",
    "English (US) - Guy (M)": "en-US-GuyNeural",
    "English (UK) - Sonia (F)": "en-GB-SoniaNeural",
    "English (UK) - Ryan (M)": "en-GB-RyanNeural",
    "English (Nigeria) - Ezinne (F)": "en-NG-EzinneNeural",
    "English (Nigeria) - Abeo (M)": "en-NG-AbeoNeural",
    "French - Denise (F)": "fr-FR-DeniseNeural",
    "Spanish - Elvira (F)": "es-ES-ElviraNeural",
}


@st.cache_data(ttl=86400, show_spinner=False)
def get_voices() -> dict:
    try:
        import edge_tts

        voices = asyncio.run(edge_tts.list_voices())
        out = {f"{v['Locale']} - {v['ShortName'].split('-')[-1].replace('Neural','')} ({v['Gender'][0]})": v["ShortName"] for v in voices}
        return dict(sorted(out.items())) or FALLBACK_VOICES
    except Exception:  # noqa: BLE001
        return FALLBACK_VOICES


def tts_edge(text: str, voice: str, rate: int, pitch: int) -> bytes:
    import edge_tts

    path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name

    async def run():
        await edge_tts.Communicate(text, voice, rate=f"{rate:+d}%", pitch=f"{pitch:+d}Hz").save(path)

    asyncio.run(run())
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return data


def tts_gtts(text: str, lang: str = "en") -> bytes:
    from gtts import gTTS

    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    return buf.getvalue()


def clone_speak(text: str, ref_bytes: bytes, lang: str, space: str, api_name: str) -> bytes:
    """Voice cloning through a public Hugging Face Space (XTTS by default)."""
    from gradio_client import Client, handle_file

    ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    ref.write(ref_bytes)
    ref.close()
    client = Client(space)
    result = client.predict(text, lang, handle_file(ref.name), handle_file(ref.name),
                            False, False, False, True, api_name=api_name)
    os.remove(ref.name)
    items = result if isinstance(result, (list, tuple)) else [result]
    for it in items:
        if isinstance(it, dict):
            it = it.get("value") or it.get("path")
        if isinstance(it, str) and it.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
            with open(it, "rb") as f:
                return f.read()
    raise RuntimeError("The cloning Space returned no audio (it may be busy or changed).")


# ----------------------------------------------------------------------------
# NAVIGATION  (only the chosen page runs => faster)
# ----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("gen_img", None)
ss.setdefault("gen_prompt", "")
ss.setdefault("gen_seed", random.randint(1, 10**6))
ss.setdefault("vid_src", None)
ss.setdefault("vid_out", None)
ss.setdefault("clone_ref", None)

page = st.radio(
    "Page",
    ["🎨 Image", "🎬 Image → Video", "🔊 Text → Audio"],
    horizontal=True,
    label_visibility="collapsed",
)

with st.expander("⚙️ Settings (optional Hugging Face token)"):
    hf_token = st.text_input(
        "Hugging Face token (free; strongly recommended for AI video, gives you your own GPU quota)",
        type="password",
        value=os.environ.get("HF_TOKEN", ""),
    )

# ============================================================================
# PAGE 1 - IMAGE GENERATOR
# ============================================================================
if page.startswith("🎨"):
    prompt = st.text_area("Describe the image", value=ss.gen_prompt, height=100,
                          placeholder="A Nigerian barbing salon at sunset, neon signs, cinematic")
    c1, c2, c3 = st.columns(3)
    style = c1.selectbox("Style", list(STYLES))
    size_name = c2.selectbox("Size", list(SIZES))
    model = c3.selectbox("Engine", ["flux", "turbo"])

    b1, b2, b3 = st.columns(3)
    go = b1.button("✨ Generate", key="gen")
    again = b2.button("🔄 New version", key="regen")
    clear = b3.button("🗑 Clear", key="clear")

    if clear:
        ss.gen_img, ss.gen_prompt = None, ""
        st.rerun()

    if (go or again) and prompt.strip():
        ss.gen_prompt = prompt.strip()
        ss.gen_seed = random.randint(1, 10**6) if again or go else ss.gen_seed
        w, h = SIZES[size_name]
        with st.spinner("Painting your image..."):
            try:
                ss.gen_img = fetch_image(ss.gen_prompt + STYLES[style], w, h, ss.gen_seed, model, hf_token)
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    elif (go or again) and not prompt.strip():
        st.warning("Type a description first.")

    if ss.gen_img:
        st.image(ss.gen_img, use_container_width=True)
        st.download_button("⬇️ Download image", ss.gen_img, "dearson_image.png", "image/png")

        st.markdown("**💬 Comment on the result: what should change?**")
        comment = st.text_area("Comment", key="comment_img", height=70, label_visibility="collapsed",
                               placeholder="Make it brighter, add a signboard that says DEARson, remove the people...")
        r1, r2 = st.columns(2)
        if r1.button("🎯 Generate with my comment", key="newimg"):
            if comment.strip():
                w, h = SIZES[size_name]
                new_prompt = f"{ss.gen_prompt}, {comment.strip()}"
                with st.spinner("Applying your comment..."):
                    try:
                        ss.gen_seed = random.randint(1, 10**6)
                        ss.gen_img = fetch_image(new_prompt + STYLES[style], w, h, ss.gen_seed, model, hf_token)
                        ss.gen_prompt = new_prompt
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(str(e))
            else:
                st.warning("Write a comment first.")
        if r2.button("🎬 Send to video", key="mkvid"):
            ss.vid_src = ss.gen_img
            st.success("Sent! Open the 🎬 Image → Video page.")

# ============================================================================
# PAGE 2 - IMAGE TO VIDEO
# ============================================================================
elif page.startswith("🎬"):
    up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    if up is not None:
        try:
            ss.vid_src = shrink(up.getvalue())
        except Exception:  # noqa: BLE001
            st.error("That file couldn't be read as an image. Try another one.")

    if ss.vid_src:
        st.image(ss.vid_src, caption="Source image", use_container_width=True)

        c1, c2, c3 = st.columns(3)
        engine = c1.selectbox("Engine", ["AI human motion", "Simple zoom/pan"])
        vsize = c2.selectbox("Shape", list(LTX_SIZES))
        secs = c3.slider("Secs/scene", 2, 8, 5)
        if engine == "Simple zoom/pan":
            m1, m2 = st.columns(2)
            effect = m1.selectbox("Motion", EFFECTS)
            fps = m2.selectbox("FPS", [12, 18, 24], index=1)

        story = st.text_area(
            "🎬 What should happen? One line = one scene (max 4)",
            key="vid_story", height=140,
            placeholder="The couple smiles and sings, guests clap and cheer\n"
                        "The couple starts to dance happily\n"
                        "The pastor walks in quickly and joins the celebration",
        )
        cinematic = st.checkbox("🎞 Cinematic look", value=True)

        st.markdown("**🖼 Want a different picture first? Describe it:**")
        vcomment = st.text_area("vc", key="vid_comment", height=70, label_visibility="collapsed",
                                placeholder="Same wedding scene but at night with glowing lights")

        v1, v2, v3 = st.columns(3)
        make = v1.button("🎬 Make video", key="mkvid")
        newi = v2.button("🖼 New image", key="newimg")
        rst = v3.button("♻️ Reset", key="resetvid")

        if rst:
            ss.vid_src = ss.vid_out = None
            st.rerun()

        if newi:
            if vcomment.strip():
                with st.spinner("Generating new image from your description..."):
                    try:
                        w, h = SIZES["Square 1:1"]
                        ss.vid_src = fetch_image(vcomment.strip(), w, h, random.randint(1, 10**6), "flux", hf_token)
                        ss.vid_out = None
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(str(e))
            else:
                st.warning("Describe the new image first.")

        if make:
            ss.vid_out = None
            if engine == "Simple zoom/pan":
                with st.spinner("Rendering video..."):
                    try:
                        ss.vid_out = make_video(ss.vid_src, effect, secs, fps, VIDEO_SIZES[vsize])
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Video failed: {e}")
            else:
                scenes = [s.strip() for s in story.splitlines() if s.strip()][:4] or [DEFAULT_SCENE]
                clips, current = [], ss.vid_src
                with st.status("Creating your AI video (about 1 minute per scene)...", expanded=True) as status:
                    try:
                        for i, scene in enumerate(scenes, 1):
                            st.write(f"🎬 Scene {i}/{len(scenes)}: {scene}")
                            prompt = scene + (". " + CINEMA if cinematic else "")
                            clip = ltx_clip(current, prompt, secs, LTX_SIZES[vsize], hf_token, LTX_SPACE)
                            clips.append(clip)
                            if i < len(scenes):
                                current = last_frame(clip)
                        ss.vid_out = join_clips(clips)
                        status.update(label="Video ready!", state="complete")
                    except Exception as e:  # noqa: BLE001
                        status.update(label="Video stopped", state="error")
                        st.error(friendly_error(e))
                        if clips:
                            ss.vid_out = join_clips(clips)
                            st.warning(f"Showing the {len(clips)} scene(s) that finished.")

        if ss.vid_out:
            st.video(ss.vid_out)
            st.download_button("⬇️ Download video", ss.vid_out, "dearson_video.mp4", "video/mp4")
    else:
        st.info("Upload an image, or create one on the 🎨 Image page and tap 'Send to video'.")

# ============================================================================
# PAGE 3 - TEXT TO AUDIO + VOICE CLONING
# ============================================================================
else:
    text = st.text_area("Text to speak", height=130, placeholder="Welcome to Dearson Barbing & Technology...")
    engine = st.radio("Engine", ["Edge voices (best)", "Google TTS (backup)"], horizontal=True)

    if engine.startswith("Edge"):
        voices = get_voices()
        names = list(voices)
        default = next((i for i, n in enumerate(names) if "en-NG" in n or "en-US" in n), 0)
        vname = st.selectbox("Voice", names, index=default)
        s1, s2 = st.columns(2)
        rate = s1.slider("Speed %", -50, 50, 0)
        pitch = s2.slider("Pitch Hz", -30, 30, 0)
    else:
        lang = st.selectbox("Language", ["en", "fr", "es", "de", "pt", "yo", "ha", "sw"])

    a1, _, _ = st.columns(3)
    if a1.button("🔊 Speak", key="speak"):
        if text.strip():
            with st.spinner("Creating audio..."):
                try:
                    audio = (tts_edge(text, voices[vname], rate, pitch) if engine.startswith("Edge")
                             else tts_gtts(text, lang))
                    st.audio(audio, format="audio/mp3")
                    st.download_button("⬇️ Download audio", audio, "dearson_audio.mp3", "audio/mpeg")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Audio failed: {e}")
        else:
            st.warning("Type some text first.")

    st.divider()
    st.subheader("🧬 Voice cloning")
    st.caption("Upload 10–30 seconds of clear speech (WAV/MP3), no background noise.")
    ref = st.file_uploader("Voice sample", type=["wav", "mp3", "m4a", "ogg"], key="refup")
    with st.expander("Cloning engine (advanced)"):
        space = st.text_input("Hugging Face Space", value="coqui/xtts")
        api_name = st.text_input("API name", value="/predict")
        clang = st.selectbox("Cloned voice language", ["en", "fr", "es", "de", "pt", "it"])

    k1, k2, k3 = st.columns(3)
    if k1.button("🧬 Clone voice", key="clone"):
        if ref is not None:
            ss.clone_ref = ref.getvalue()
            st.success("Voice sample saved. Now tap 'Speak cloned'.")
        else:
            st.warning("Upload a voice sample first.")
    if k2.button("🎙 Speak cloned", key="speakclone"):
        if not ss.clone_ref:
            st.warning("Clone a voice first.")
        elif not text.strip():
            st.warning("Type some text first.")
        else:
            with st.spinner("Cloning and speaking (can take a minute)..."):
                try:
                    cloned = clone_speak(text, ss.clone_ref, clang, space, api_name)
                    st.audio(cloned, format="audio/wav")
                    st.download_button("⬇️ Download cloned audio", cloned, "dearson_cloned.wav", "audio/wav")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Cloning failed: {e}. The free Space may be busy; try again shortly.")
    if k3.button("🗑 Clear voice", key="clearclone"):
        ss.clone_ref = None
        st.info("Voice sample removed.")
