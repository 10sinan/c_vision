"""
Stage 2 Demo - Ham vs. EMA ile yumuşatılmış kafa konumu.
Kırmızı: ham, Yeşil: filtrelenmiş. Çıkış: 'q'. '+'/'-' ile alpha ayarı.
"""

import cv2
from head_tracker import HeadTracker
from signal_filter import PoseNormalizer, EMAFilter


def to_px(x_n, y_n, w, h):
    return int(x_n * w / 2 + w / 2), int(h / 2 - y_n * h / 2)


def main():
    tracker = HeadTracker(width=640, height=480)
    tracker.start_camera()
    normalizer = PoseNormalizer(640, 480)
    ema = EMAFilter(alpha=0.8)

    try:
        while True:
            ok, frame, pose = tracker.process_frame()
            if not ok:
                break
            h, w = frame.shape[:2]
            if pose is not None:
                raw = normalizer.normalize(pose)
                smooth = ema.update(raw)
                cv2.circle(frame, to_px(raw[0], raw[1], w, h), 6, (0, 0, 255), -1)
                cv2.circle(frame, to_px(smooth[0], smooth[1], w, h), 9, (0, 255, 0), 2)
                cv2.putText(frame, f"Ham   : {raw[0]:+.3f} {raw[1]:+.3f} {raw[2]:.0f}cm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                cv2.putText(frame, f"EMA   : {smooth[0]:+.3f} {smooth[1]:+.3f} {smooth[2]:.0f}cm", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame, f"alpha={ema.alpha:.2f}  (+/- ile degistir)", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow("Asama 2: Normalizasyon + EMA", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('+'), ord('=')):
                ema.alpha = min(1.0, ema.alpha + 0.05)
            elif key == ord('-'):
                ema.alpha = max(0.05, ema.alpha - 0.05)
    finally:
        tracker.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
