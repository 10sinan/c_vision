"""
Arka plan iş parçacığında kamera + MediaPipe çalıştırır.
Render döngüsü kamerayı beklemez; sadece en son poz bilgisini okur.
"""

import threading
import time

from head_tracker import HeadTracker


class TrackerThread:
    def __init__(self, tracker: HeadTracker):
        self.tracker = tracker
        self._lock = threading.Lock()
        self._pose = None          # en son ham poz
        self._stamp = 0.0          # en son yüz tespit zamanı
        self._running = False
        self._thread = None
        self.failed = False        # kamera akışı koptu mu

    def start(self):
        self.tracker.start_camera()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, _frame, pose = self.tracker.process_frame()
            if not ok:
                self.failed = True
                break
            if pose is not None:
                with self._lock:
                    self._pose = pose
                    self._stamp = time.monotonic()

    def latest(self):
        """:return: (pose | None, yüzün görüldüğü son zamandan beri geçen sn)"""
        with self._lock:
            if self._pose is None:
                return None, float("inf")
            return self._pose, time.monotonic() - self._stamp

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.tracker.release()
