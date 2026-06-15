"""
딜레이 효과 GIF 시각화

같은 트랙(seed 고정)에서 여러 delay로 에이전트를 주행시켜 나란히 비교하는
GIF를 만든다. delay=0은 깔끔하게 주행, delay가 커질수록 흔들리다 트랙 이탈.

실행 (프로젝트 루트에서):
    python src/make_gif.py --delays 0 6 10 20
"""
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium_delay import ObsActDelayWrapper
from PIL import Image, ImageDraw, ImageFont

import common

PANEL_W, PANEL_H = 300, 200
HEADER_H = 30


def get_font(size):
    for p in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT = get_font(18)


def rollout(model, delay, max_steps, stride, seed):
    """delay 환경에서 한 에피소드를 주행하며 프레임을 수집한다."""
    env = ObsActDelayWrapper(
        gym.make("CarRacing-v3", render_mode="rgb_array"), act_delay=delay)
    obs, _ = env.reset(seed=seed)
    kept = []          # (frame, cum_reward, done)
    total, done = 0.0, False
    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        total += r
        done = bool(term or trunc)
        if t % stride == 0 or done:
            kept.append((env.render(), total, done))
        if done:
            break
    env.close()
    return kept, total, done


def make_panel(frame, delay, reward, mark_fail):
    img = Image.fromarray(frame).resize((PANEL_W, PANEL_H))
    panel = Image.new("RGB", (PANEL_W, PANEL_H + HEADER_H), (20, 20, 20))
    panel.paste(img, (0, HEADER_H))
    d = ImageDraw.Draw(panel)
    d.rectangle([0, 0, PANEL_W, HEADER_H], fill=(150, 30, 30) if mark_fail else (40, 40, 40))
    status = "  CRASH" if mark_fail else ""
    d.text((6, 6), f"delay={delay:>2}   R={reward:6.0f}{status}",
           font=FONT, fill=(255, 255, 255))
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     type=str, default=str(common.DEFAULT_MODEL))
    parser.add_argument("--delays",    nargs="+", type=int, default=[0, 6, 10, 20])
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--stride",    type=int, default=3)
    parser.add_argument("--fps",       type=int, default=20)
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--out",       type=str,
                        default=str(common.RESULTS_DIR / "delay_compare.gif"))
    args = parser.parse_args()

    model = PPO.load(args.model)

    rollouts = {}   # delay -> (kept, crashed)
    maxlen = 0
    for delay in args.delays:
        kept, total, crashed = rollout(
            model, delay, args.max_steps, args.stride, args.seed)
        rollouts[delay] = (kept, crashed)
        maxlen = max(maxlen, len(kept))
        print(f"delay={delay:>2}: {len(kept):>3} frames | "
              f"final R={total:7.0f} | crashed={crashed}")

    # 길이가 다른 패널을 최대 길이에 맞춰 마지막 프레임으로 정지(freeze) 패딩
    W, H = PANEL_W * len(args.delays), PANEL_H + HEADER_H
    composites = []
    for i in range(maxlen):
        comp = Image.new("RGB", (W, H))
        for j, delay in enumerate(args.delays):
            kept, crashed = rollouts[delay]
            if i < len(kept):
                frame, reward, done = kept[i]
                mark = done
            else:                       # 종료된 패널은 마지막 프레임으로 정지
                frame, reward, _ = kept[-1]
                mark = crashed
            comp.paste(make_panel(frame, delay, reward, mark), (j * PANEL_W, 0))
        composites.append(comp)

    composites[0].save(
        args.out, save_all=True, append_images=composites[1:],
        duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"\nGIF 저장: {args.out}  ({len(composites)} frames, {W}x{H})")


if __name__ == "__main__":
    main()
