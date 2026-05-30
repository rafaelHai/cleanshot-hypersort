from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .config import expand_path
from .organizer import OrganizeResult, ScreenshotOrganizer

Callback = Callable[[OrganizeResult], None]


class ScreenshotEventHandler(FileSystemEventHandler):
    def __init__(self, organizer: ScreenshotOrganizer, callback: Optional[Callback] = None):
        self.organizer = organizer
        self.callback = callback

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        destination = getattr(event, "dest_path", None)
        if destination:
            self._handle_path(Path(destination))

    def _handle(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._handle_path(Path(event.src_path))

    def _handle_path(self, path: Path) -> None:
        result = self.organizer.organize_file(path)
        if result.status != "ignored" and self.callback:
            self.callback(result)


class ScreenshotWatcher:
    def __init__(self, config: Dict[str, Any], callback: Optional[Callback] = None):
        self.config = config
        self.callback = callback
        self.organizer = ScreenshotOrganizer(config)
        self.observer: Optional[Observer] = None
        self.is_running = False

    def start(self) -> None:
        if self.is_running:
            return

        watch_folder = expand_path(self.config.get("watch_folder", "~/Desktop"))
        watch_folder.mkdir(parents=True, exist_ok=True)

        recursive = bool(self.config.get("recursive_watch", False))
        self.observer = Observer()
        handler = ScreenshotEventHandler(self.organizer, self.callback)
        self.observer.schedule(handler, str(watch_folder), recursive=recursive)
        self.observer.start()
        self.is_running = True

    def stop(self) -> None:
        if not self.observer:
            self.is_running = False
            return

        self.observer.stop()
        self.observer.join(timeout=3)
        self.observer = None
        self.is_running = False

    def restart(self, config: Dict[str, Any]) -> None:
        self.stop()
        self.config = config
        self.organizer = ScreenshotOrganizer(config)
        self.start()
