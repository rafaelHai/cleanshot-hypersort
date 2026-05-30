from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QGraphicsDropShadowEffect,
)

from cleanshot.core.activity import append_activity
from cleanshot.core.config import config_path, expand_path, load_config, save_config
from cleanshot.core.organizer import ScreenshotOrganizer
from cleanshot.core.brain import brain_path, learn_from_output_folders, memory_count, taught_count, teach_image
from cleanshot.core.stats import FolderSummary, collect_stats, human_size
from cleanshot.core.watcher import ScreenshotWatcher
from cleanshot.ui.styles import LIGHT_STYLE


MODE_OPTIONS = [
    ("smart", "Smart mode - learns + recognizes content"),
    ("app", "App mode - folder per app"),
    ("day", "Day mode - folder per day"),
]


class MainWindow(QMainWindow):
    organize_event = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CleanShot")
        self.resize(1240, 820)
        self.setMinimumSize(1080, 720)
        self.setStyleSheet(LIGHT_STYLE)

        self.config: Dict[str, Any] = load_config()
        self.watcher: ScreenshotWatcher | None = None

        self.nav_buttons: List[QPushButton] = []
        self.total_label: QLabel | None = None
        self.size_label: QLabel | None = None
        self.today_label: QLabel | None = None
        self.mode_label: QLabel | None = None
        self.brain_label: QLabel | None = None
        self.status_label: QLabel | None = None
        self.types_layout: QVBoxLayout | None = None
        self.folders_layout: QVBoxLayout | None = None
        self.activity_layout: QVBoxLayout | None = None

        self.watch_input: QLineEdit | None = None
        self.output_input: QLineEdit | None = None
        self.file_types_input: QLineEdit | None = None
        self.keywords_input: QLineEdit | None = None
        self.organization_mode_combo: QComboBox | None = None
        self.copy_mode_combo: QComboBox | None = None
        self.rename_checkbox: QCheckBox | None = None
        self.duplicates_checkbox: QCheckBox | None = None
        self.recursive_checkbox: QCheckBox | None = None
        self.enable_ocr_checkbox: QCheckBox | None = None
        self.confidence_input: QLineEdit | None = None
        self.smart_confidence_input: QLineEdit | None = None
        self.visual_distance_input: QLineEdit | None = None
        self.neuro_threshold_input: QLineEdit | None = None
        self.session_minutes_input: QLineEdit | None = None
        self.unknown_app_input: QLineEdit | None = None
        self.other_folder_input: QLineEdit | None = None
        self.review_folder_input: QLineEdit | None = None
        self.folder_template_input: QLineEdit | None = None
        self.filename_template_input: QLineEdit | None = None
        self.smart_subfolders_checkbox: QCheckBox | None = None
        self.review_low_confidence_checkbox: QCheckBox | None = None
        self.visual_memory_checkbox: QCheckBox | None = None
        self.session_memory_checkbox: QCheckBox | None = None
        self.neuro_learning_checkbox: QCheckBox | None = None
        self.teach_file_input: QLineEdit | None = None
        self.teach_category_input: QLineEdit | None = None
        self.teach_subcategory_input: QLineEdit | None = None

        self.stack = QStackedWidget()
        self._build_ui()

        self.organize_event.connect(self._on_organized)
        self._start_watcher()

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_overview)
        self.refresh_timer.start(2500)
        self.refresh_overview()

    def _build_ui(self) -> None:
        root = QFrame()
        root.setObjectName("Root")
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        sidebar = self._build_sidebar()
        root_layout.addWidget(sidebar)

        content = QFrame()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(32, 30, 32, 30)
        content_layout.setSpacing(0)

        self.stack.addWidget(self._build_overview_page())
        self.stack.addWidget(self._build_settings_page())
        self.stack.addWidget(self._build_teach_page())
        content_layout.addWidget(self.stack)

        root_layout.addWidget(content, 1)
        self.setCentralWidget(root)
        self._set_active_nav(0)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(230)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(22, 24, 22, 22)
        layout.setSpacing(14)

        title = QLabel("CleanShot")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Local smart sorter")
        subtitle.setObjectName("AppSubtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(18)

        overview_btn = self._nav_button("Overview", 0)
        settings_btn = self._nav_button("Settings", 1)
        teach_btn = self._nav_button("Teach", 2)
        layout.addWidget(overview_btn)
        layout.addWidget(settings_btn)
        layout.addWidget(teach_btn)
        layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.status_label = QLabel("Starting")
        self.status_label.setObjectName("StatusPill")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        version = QLabel("v1.0.0")
        version.setObjectName("SmallMutedText")
        version.setAlignment(Qt.AlignCenter)
        layout.addWidget(version)

        return sidebar

    def _nav_button(self, text: str, index: int) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("NavButton")
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: self._set_active_nav(index))
        self.nav_buttons.append(button)
        return button

    def _set_active_nav(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for button_index, button in enumerate(self.nav_buttons):
            button.setProperty("active", button_index == index)
            button.style().unpolish(button)
            button.style().polish(button)

    def _build_overview_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("OverviewScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("TransparentWidget")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(18)

        header = QHBoxLayout()
        header_text = QVBoxLayout()
        title = QLabel("Overview")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Live overview, learning memory, folder shortcuts, and clean activity history.")
        subtitle.setObjectName("PageSubtitle")
        header_text.addWidget(title)
        header_text.addWidget(subtitle)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_overview)
        open_output_btn = QPushButton("Open Output")
        open_output_btn.setObjectName("PrimaryButton")
        open_output_btn.clicked.connect(lambda: self._open_folder(self.config.get("output_folder", "~/Pictures/CleanShot")))

        header.addLayout(header_text, 1)
        header.addWidget(refresh_btn)
        header.addWidget(open_output_btn)
        layout.addLayout(header)

        cards = QGridLayout()
        cards.setHorizontalSpacing(14)
        cards.setVerticalSpacing(14)

        self.total_label = QLabel("0")
        self.size_label = QLabel("0 B")
        self.today_label = QLabel("0")
        self.mode_label = QLabel("Day")
        self.brain_label = QLabel("0")

        cards.addWidget(self._stat_card("Total screenshots", self.total_label), 0, 0)
        cards.addWidget(self._stat_card("Storage used", self.size_label), 0, 1)
        cards.addWidget(self._stat_card("Organized today", self.today_label), 0, 2)
        cards.addWidget(self._stat_card("Current mode", self.mode_label), 0, 3)
        cards.addWidget(self._stat_card("Learned patterns", self.brain_label), 1, 0)
        layout.addLayout(cards)

        lower_grid = QGridLayout()
        lower_grid.setHorizontalSpacing(14)
        lower_grid.setVerticalSpacing(14)

        types_card, self.types_layout = self._section_card("Screenshot types", min_height=250)
        folders_card, self.folders_layout = self._section_card("Folders", min_height=250)
        activity_card, self.activity_layout = self._section_card("Activity log", min_height=360, scrollable=True)

        lower_grid.addWidget(types_card, 0, 0)
        lower_grid.addWidget(folders_card, 0, 1)
        lower_grid.addWidget(activity_card, 1, 0, 1, 2)
        lower_grid.setColumnStretch(0, 1)
        lower_grid.setColumnStretch(1, 1)
        layout.addLayout(lower_grid)
        layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding))

        scroll.setWidget(content)
        outer.addWidget(scroll)
        return page


    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(18)

        title = QLabel("Settings")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Choose how CleanShot sorts screenshots and where it saves them.")
        subtitle.setObjectName("PageSubtitle")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(14)

        paths_card = self._card()
        paths_layout = QVBoxLayout(paths_card)
        paths_layout.setContentsMargins(18, 18, 18, 18)
        paths_layout.setSpacing(14)
        paths_layout.addWidget(self._section_header("Folders"))

        self.watch_input = QLineEdit(self.config.get("watch_folder", "~/Desktop"))
        self.output_input = QLineEdit(self.config.get("output_folder", "~/Pictures/CleanShot"))
        paths_layout.addLayout(self._path_row("Watch folder", self.watch_input))
        paths_layout.addLayout(self._path_row("Output folder", self.output_input))

        mode_card = self._card()
        mode_layout = QVBoxLayout(mode_card)
        mode_layout.setContentsMargins(18, 18, 18, 18)
        mode_layout.setSpacing(14)
        mode_layout.addWidget(self._section_header("Organization mode"))

        self.organization_mode_combo = QComboBox()
        for _mode, label in MODE_OPTIONS:
            self.organization_mode_combo.addItem(label)
        self.organization_mode_combo.setCurrentIndex(self._mode_index(self.config.get("organization_mode", "day")))
        mode_layout.addLayout(self._form_row("Sorting mode", self.organization_mode_combo))

        explanation = QLabel(
            "Smart mode is the default: it combines visual recognition, OCR when enabled, app/window context, short session memory, and your taught examples. "
            "App mode sorts by the active app. Day mode sorts by date. Unsure Smart decisions go to the review folder instead of guessing."
        )
        explanation.setObjectName("MutedText")
        explanation.setWordWrap(True)
        mode_layout.addWidget(explanation)

        options_card = self._card()
        options_layout = QVBoxLayout(options_card)
        options_layout.setContentsMargins(18, 18, 18, 18)
        options_layout.setSpacing(14)
        options_layout.addWidget(self._section_header("Rules"))

        self.file_types_input = QLineEdit(", ".join(self.config.get("file_types", [])))
        self.keywords_input = QLineEdit(", ".join(self.config.get("screenshot_keywords", [])))
        self.unknown_app_input = QLineEdit(self.config.get("unknown_app_folder", "Unknown App"))
        self.other_folder_input = QLineEdit(self.config.get("other_folder", "Other"))
        self.review_folder_input = QLineEdit(self.config.get("review_folder", "_Review"))
        self.confidence_input = QLineEdit("0.75")
        self.smart_confidence_input = QLineEdit(str(self.config.get("smart_confidence_threshold", 0.72)))
        self.visual_distance_input = QLineEdit(str(self.config.get("visual_similarity_distance", 8)))
        self.neuro_threshold_input = QLineEdit(str(self.config.get("neuro_similarity_threshold", 0.86)))
        self.session_minutes_input = QLineEdit(str(self.config.get("session_context_minutes", 4)))
        self.folder_template_input = QLineEdit(self.config.get("folder_template", "{mode_folder}"))
        self.filename_template_input = QLineEdit(self.config.get("filename_template", "screenshot-{timestamp}{extension}"))

        self.copy_mode_combo = QComboBox()
        self.copy_mode_combo.addItems(["Move screenshots", "Copy screenshots"])
        self.copy_mode_combo.setCurrentIndex(1 if self.config.get("copy_instead_of_move", False) else 0)

        options_layout.addLayout(self._form_row("File types", self.file_types_input))
        options_layout.addLayout(self._form_row("Screenshot name keywords", self.keywords_input))
        options_layout.addLayout(self._form_row("Move or copy", self.copy_mode_combo))
        options_layout.addLayout(self._form_row("Unknown app folder", self.unknown_app_input))
        options_layout.addLayout(self._form_row("Fallback folder", self.other_folder_input))
        options_layout.addLayout(self._form_row("Review folder", self.review_folder_input))
        options_layout.addLayout(self._form_row("Smart mode threshold", self.smart_confidence_input))
        options_layout.addLayout(self._form_row("ScreenshotDNA similarity distance", self.visual_distance_input))
        options_layout.addLayout(self._form_row("NeuroVector learning threshold", self.neuro_threshold_input))
        options_layout.addLayout(self._form_row("Session memory minutes", self.session_minutes_input))
        options_layout.addLayout(self._form_row("Folder template", self.folder_template_input))
        options_layout.addLayout(self._form_row("Filename template", self.filename_template_input))

        self.rename_checkbox = QCheckBox("Auto rename screenshots")
        self.duplicates_checkbox = QCheckBox("Detect duplicates")
        self.recursive_checkbox = QCheckBox("Watch subfolders recursively")
        self.enable_ocr_checkbox = QCheckBox("Use OCR if pytesseract is installed")
        self.smart_subfolders_checkbox = QCheckBox("Smart subfolders, e.g. Code/JavaScript")
        self.review_low_confidence_checkbox = QCheckBox("Send unsure decisions to review")
        self.visual_memory_checkbox = QCheckBox("Use ScreenshotDNA visual memory")
        self.session_memory_checkbox = QCheckBox("Use short session memory")
        self.neuro_learning_checkbox = QCheckBox("Use NeuroVector learned recognition")

        self.rename_checkbox.setChecked(bool(self.config.get("auto_rename", True)))
        self.duplicates_checkbox.setChecked(bool(self.config.get("detect_duplicates", True)))
        self.recursive_checkbox.setChecked(bool(self.config.get("recursive_watch", False)))
        self.enable_ocr_checkbox.setChecked(bool(self.config.get("enable_ocr", False)))
        self.smart_subfolders_checkbox.setChecked(bool(self.config.get("smart_subfolders", True)))
        self.review_low_confidence_checkbox.setChecked(bool(self.config.get("review_low_confidence", True)))
        self.visual_memory_checkbox.setChecked(bool(self.config.get("enable_visual_memory", True)))
        self.session_memory_checkbox.setChecked(bool(self.config.get("enable_session_memory", True)))
        self.neuro_learning_checkbox.setChecked(bool(self.config.get("enable_neuro_learning", True)))

        checkbox_grid = QGridLayout()
        checkbox_grid.setHorizontalSpacing(18)
        checkbox_grid.setVerticalSpacing(12)
        checkbox_grid.addWidget(self.rename_checkbox, 0, 0)
        checkbox_grid.addWidget(self.duplicates_checkbox, 0, 1)
        checkbox_grid.addWidget(self.recursive_checkbox, 1, 0)
        checkbox_grid.addWidget(self.enable_ocr_checkbox, 1, 1)
        checkbox_grid.addWidget(self.smart_subfolders_checkbox, 2, 0)
        checkbox_grid.addWidget(self.review_low_confidence_checkbox, 2, 1)
        checkbox_grid.addWidget(self.visual_memory_checkbox, 3, 0)
        checkbox_grid.addWidget(self.session_memory_checkbox, 3, 1)
        checkbox_grid.addWidget(self.neuro_learning_checkbox, 4, 0)
        options_layout.addLayout(checkbox_grid)

        actions_card = self._card()
        actions_layout = QVBoxLayout(actions_card)
        actions_layout.setContentsMargins(18, 18, 18, 18)
        actions_layout.setSpacing(14)
        actions_layout.addWidget(self._section_header("Actions"))

        actions_row = QHBoxLayout()
        save_btn = QPushButton("Save Settings")
        save_btn.setObjectName("PrimaryButton")
        save_btn.clicked.connect(self.save_settings)
        organize_existing_btn = QPushButton("Organize Existing Screenshots")
        organize_existing_btn.clicked.connect(self.organize_existing)
        learn_btn = QPushButton("Learn From Folders")
        learn_btn.clicked.connect(self.learn_from_folders)
        open_review_btn = QPushButton("Open Review")
        open_review_btn.clicked.connect(lambda: self._open_folder(expand_path(self.config.get("output_folder", "~/Pictures/CleanShot")) / self.config.get("review_folder", "_Review")))
        open_config_btn = QPushButton("Open Config Folder")
        open_config_btn.clicked.connect(lambda: self._open_folder(config_path().parent))

        actions_row.addWidget(save_btn)
        actions_row.addWidget(organize_existing_btn)
        actions_row.addWidget(learn_btn)
        actions_row.addWidget(open_review_btn)
        actions_row.addWidget(open_config_btn)
        actions_row.addStretch(1)
        actions_layout.addLayout(actions_row)

        hint = QLabel(
            "Template tokens: {mode_folder}, {smart_folder}, {category}, {subcategory}, {app}, {date}, {year}, {month}, {day}, "
            "{timestamp}, {original_stem}, {extension}, {confidence}."
        )
        hint.setObjectName("MutedText")
        hint.setWordWrap(True)
        actions_layout.addWidget(hint)

        layout.addWidget(paths_card)
        layout.addWidget(mode_card)
        layout.addWidget(options_card)
        layout.addWidget(actions_card)
        layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding))

        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        return page

    def _build_teach_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(18)

        title = QLabel("Teach")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Correct CleanShot by showing it examples. Your choices become local learned patterns in ~/.cleanshot/brain.json.")
        subtitle.setObjectName("PageSubtitle")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(14)

        manual_card = self._card()
        manual_layout = QVBoxLayout(manual_card)
        manual_layout.setContentsMargins(18, 18, 18, 18)
        manual_layout.setSpacing(14)
        manual_layout.addWidget(self._section_header("Teach one screenshot"))

        self.teach_file_input = QLineEdit()
        self.teach_file_input.setPlaceholderText("Choose a screenshot that was sorted wrong")
        self.teach_category_input = QLineEdit()
        self.teach_category_input.setPlaceholderText("Example: Code, Photos, Browser, Documents")
        self.teach_subcategory_input = QLineEdit()
        self.teach_subcategory_input.setPlaceholderText("Optional: JavaScript, Family, GitHub, Receipts")

        file_row = QHBoxLayout()
        file_row.addLayout(self._form_row("Screenshot", self.teach_file_input), 1)
        choose_file_btn = QPushButton("Choose")
        choose_file_btn.clicked.connect(self._choose_teach_file)
        file_row.addWidget(choose_file_btn)
        manual_layout.addLayout(file_row)
        manual_layout.addLayout(self._form_row("Correct folder / category", self.teach_category_input))
        manual_layout.addLayout(self._form_row("Correct subfolder", self.teach_subcategory_input))

        teach_btn = QPushButton("Teach CleanShot")
        teach_btn.setObjectName("PrimaryButton")
        teach_btn.clicked.connect(self.teach_single_screenshot)
        manual_layout.addWidget(teach_btn)

        folder_card = self._card()
        folder_layout = QVBoxLayout(folder_card)
        folder_layout.setContentsMargins(18, 18, 18, 18)
        folder_layout.setSpacing(14)
        folder_layout.addWidget(self._section_header("Teach from folders"))
        folder_help = QLabel(
            "Put screenshots into the folders you want, then click Learn From Folders. "
            "Example: CleanShot/Code/JavaScript/example.png teaches Code/JavaScript. "
            "Example: CleanShot/Photos/Family/example.png teaches Photos/Family."
        )
        folder_help.setObjectName("MutedText")
        folder_help.setWordWrap(True)
        folder_layout.addWidget(folder_help)

        folder_actions = QHBoxLayout()
        open_output_btn = QPushButton("Open Output Folder")
        open_output_btn.clicked.connect(lambda: self._open_folder(self.config.get("output_folder", "~/Pictures/CleanShot")))
        learn_btn = QPushButton("Learn From Folders")
        learn_btn.setObjectName("PrimaryButton")
        learn_btn.clicked.connect(self.learn_from_folders)
        open_brain_btn = QPushButton("Open Brain Folder")
        open_brain_btn.clicked.connect(lambda: self._open_folder(brain_path().parent))
        folder_actions.addWidget(open_output_btn)
        folder_actions.addWidget(learn_btn)
        folder_actions.addWidget(open_brain_btn)
        folder_actions.addStretch(1)
        folder_layout.addLayout(folder_actions)

        tech_card = self._card()
        tech_layout = QVBoxLayout(tech_card)
        tech_layout.setContentsMargins(18, 18, 18, 18)
        tech_layout.setSpacing(10)
        tech_layout.addWidget(self._section_header("How learning works"))
        tech_text = QLabel(
            "CleanShot now uses NeuroVector, a local visual embedding I built for this app. "
            "It stores a compact fingerprint of examples you teach it, then compares new screenshots using weighted similarity. "
            "It also still uses OCR, VisionCore code detection, ScreenshotDNA near-duplicate matching, app context, and session memory. "
            "Nothing is uploaded."
        )
        tech_text.setObjectName("MutedText")
        tech_text.setWordWrap(True)
        tech_layout.addWidget(tech_text)

        layout.addWidget(manual_card)
        layout.addWidget(folder_card)
        layout.addWidget(tech_card)
        layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding))
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        return page

    def _choose_teach_file(self) -> None:
        if not self.teach_file_input:
            return
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Choose screenshot to teach",
            str(expand_path(self.config.get("watch_folder", "~/Desktop"))),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)",
        )
        if chosen:
            self.teach_file_input.setText(chosen)

    def teach_single_screenshot(self) -> None:
        if not self.teach_file_input or not self.teach_category_input or not self.teach_subcategory_input:
            return
        path = self.teach_file_input.text().strip()
        category = self.teach_category_input.text().strip()
        subcategory = self.teach_subcategory_input.text().strip()
        if not path or not category:
            QMessageBox.warning(self, "CleanShot", "Choose a screenshot and type the correct category.")
            return
        ok = teach_image(Path(path), category=category, subcategory=subcategory, config=self.config)
        if not ok:
            QMessageBox.warning(self, "CleanShot", "Could not teach from that file. Make sure it is an image.")
            return
        append_activity("teach", f"Taught CleanShot: {Path(path).name} -> {category}{('/' + subcategory) if subcategory else ''}", path, "")
        self.refresh_overview()
        QMessageBox.information(self, "CleanShot", f"Learned this example as {category}{('/' + subcategory) if subcategory else ''}.")

    def _stat_card(self, label: str, value_label: QLabel) -> QFrame:
        card = self._card()
        card.setMinimumHeight(116)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)

        title = QLabel(label)
        title.setObjectName("CardTitle")
        value_label.setObjectName("CardValue")
        value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(value_label)
        return card

    def _section_card(self, title: str, min_height: int = 0, scrollable: bool = False) -> tuple[QFrame, QVBoxLayout]:
        card = self._card()
        if min_height:
            card.setMinimumHeight(min_height)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(self._section_header(title))

        content_layout = QVBoxLayout()
        content_layout.setSpacing(8)

        if scrollable:
            holder = QWidget()
            holder.setObjectName("TransparentWidget")
            holder.setLayout(content_layout)
            content_layout.setContentsMargins(0, 0, 0, 0)

            scroll = QScrollArea()
            scroll.setObjectName("InnerScroll")
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setWidget(holder)
            layout.addWidget(scroll, 1)
        else:
            layout.addLayout(content_layout, 1)

        return card, content_layout

    def _card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(17, 24, 39, 22))
        card.setGraphicsEffect(shadow)
        return card

    def _section_header(self, title: str) -> QLabel:
        label = QLabel(title)
        label.setObjectName("SectionTitle")
        return label

    def _path_row(self, label: str, line_edit: QLineEdit) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addLayout(self._form_row(label, line_edit), 1)
        choose_btn = QPushButton("Choose")
        choose_btn.clicked.connect(lambda: self._choose_folder(line_edit))
        row.addWidget(choose_btn)
        open_btn = QPushButton("Open")
        open_btn.clicked.connect(lambda: self._open_folder(line_edit.text()))
        row.addWidget(open_btn)
        return row

    def _form_row(self, label: str, widget: QWidget) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(7)
        label_widget = QLabel(label)
        label_widget.setObjectName("MutedText")
        layout.addWidget(label_widget)
        layout.addWidget(widget)
        return layout

    def refresh_overview(self) -> None:
        stats = collect_stats(self.config)
        if self.total_label:
            self.total_label.setText(f"{stats['total_screenshots']:,}")
        if self.size_label:
            self.size_label.setText(str(stats["total_size_label"]))
        if self.today_label:
            self.today_label.setText(f"{stats['organized_today']:,}")
        if self.mode_label:
            self.mode_label.setText(self._mode_display(self.config.get("organization_mode", "day")))
        if self.brain_label:
            self.brain_label.setText(f"{memory_count():,} / {taught_count():,} taught")

        if self.status_label:
            self.status_label.setText("Watching" if self.watcher and self.watcher.is_running else "Stopped")

        self._render_types(stats.get("types", {}))
        self._render_folders(stats.get("folders", []))
        self._render_activity(stats.get("activity", []))

    def _render_types(self, types: Dict[str, int]) -> None:
        if not self.types_layout:
            return
        self._clear_layout(self.types_layout)

        if not types:
            self.types_layout.addWidget(self._empty_label("No screenshots organized yet."))
            self.types_layout.addStretch(1)
            return

        total = max(sum(types.values()), 1)
        for extension, count in types.items():
            row = QVBoxLayout()
            top = QHBoxLayout()
            name = QLabel(extension)
            name.setObjectName("MutedText")
            amount = QLabel(f"{count:,}")
            amount.setObjectName("MutedText")
            amount.setAlignment(Qt.AlignRight)
            top.addWidget(name)
            top.addStretch(1)
            top.addWidget(amount)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(round((count / total) * 100))
            bar.setTextVisible(False)
            row.addLayout(top)
            row.addWidget(bar)
            self.types_layout.addLayout(row)

        self.types_layout.addStretch(1)

    def _render_folders(self, folders: List[FolderSummary]) -> None:
        if not self.folders_layout:
            return
        self._clear_layout(self.folders_layout)

        if not folders:
            self.folders_layout.addWidget(self._empty_label("No output folders yet."))
            self.folders_layout.addStretch(1)
            return

        for folder in folders:
            row = QFrame()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 2, 0, 2)
            row_layout.setSpacing(10)

            text_layout = QVBoxLayout()
            text_layout.setSpacing(2)
            folder_name = QLabel(folder.path.name)
            folder_name.setObjectName("MutedText")
            folder_info = QLabel(f"{folder.count:,} files - {human_size(folder.size)}")
            folder_info.setObjectName("SmallMutedText")
            text_layout.addWidget(folder_name)
            text_layout.addWidget(folder_info)

            open_btn = QPushButton("Open")
            open_btn.setMaximumWidth(78)
            open_btn.clicked.connect(lambda checked=False, path=folder.path: self._open_folder(path))

            row_layout.addLayout(text_layout, 1)
            row_layout.addWidget(open_btn)
            self.folders_layout.addWidget(row)

        self.folders_layout.addStretch(1)

    def _render_activity(self, records: List[Dict[str, Any]]) -> None:
        if not self.activity_layout:
            return
        self._clear_layout(self.activity_layout)

        if not records:
            self.activity_layout.addWidget(self._empty_label("No activity yet. Take a screenshot or run Organize Existing."))
            self.activity_layout.addStretch(1)
            return

        for record in records[:18]:
            row_frame = QFrame()
            row_frame.setObjectName("ActivityRow")
            row_frame.setMinimumHeight(62)
            row_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            row = QHBoxLayout(row_frame)
            row.setContentsMargins(12, 8, 12, 8)
            row.setSpacing(12)

            action_name = str(record.get("action", "activity")).lower()
            time_text = self._format_time(str(record.get("time", "")))
            category = str(record.get("category") or "").strip()
            subcategory = str(record.get("subcategory") or "").strip()
            confidence = record.get("confidence")
            reason = str(record.get("reason") or "").strip()
            destination = str(record.get("destination") or "").strip()
            source = str(record.get("source") or "").strip()
            message_text = str(record.get("message", "Activity"))

            time_label = QLabel(time_text)
            time_label.setObjectName("ActivityTime")
            time_label.setFixedWidth(64)
            time_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

            text_box = QVBoxLayout()
            text_box.setSpacing(3)

            title_row = QHBoxLayout()
            title_row.setSpacing(8)
            message = QLabel(message_text)
            message.setObjectName("ActivityMessage")
            message.setWordWrap(False)
            message.setTextInteractionFlags(Qt.TextSelectableByMouse)
            title_row.addWidget(message, 1)

            if confidence is not None:
                try:
                    pct = round(float(confidence) * 100)
                    conf = QLabel(f"{pct}%")
                    conf.setObjectName("ConfidencePill")
                    conf.setAlignment(Qt.AlignCenter)
                    title_row.addWidget(conf)
                except (TypeError, ValueError):
                    pass

            text_box.addLayout(title_row)

            detail_bits: List[str] = []
            if category:
                label = category if not subcategory else f"{category}/{subcategory}"
                detail_bits.append(label)
            if reason:
                detail_bits.append(reason[:90])
            elif source:
                detail_bits.append(Path(source).name)
            if destination:
                detail_bits.append(f"→ {Path(destination).parent.name}")
            detail = QLabel("  •  ".join(detail_bits) if detail_bits else "Ready")
            detail.setObjectName("ActivityDetail")
            detail.setWordWrap(False)
            text_box.addWidget(detail)

            badge = QLabel(action_name.title())
            badge.setObjectName("ActionBadge")
            badge.setProperty("kind", action_name)
            badge.setAlignment(Qt.AlignCenter)
            badge.setFixedWidth(102)

            row.addWidget(time_label)
            row.addLayout(text_box, 1)
            row.addWidget(badge)
            self.activity_layout.addWidget(row_frame)

            animation = QPropertyAnimation(row_frame, b"windowOpacity", row_frame)
            animation.setDuration(220)
            animation.setStartValue(0.0)
            animation.setEndValue(1.0)
            animation.setEasingCurve(QEasingCurve.OutCubic)
            animation.start(QPropertyAnimation.DeleteWhenStopped)

        self.activity_layout.addStretch(1)

    def _empty_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("MutedText")
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        return label

    def _format_time(self, value: str) -> str:
        try:
            return datetime.fromisoformat(value).strftime("%H:%M")
        except ValueError:
            return "--:--"

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_any_layout(child_layout)

    def _clear_any_layout(self, layout: Any) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_any_layout(child_layout)

    def _choose_folder(self, line_edit: QLineEdit) -> None:
        current = expand_path(line_edit.text())
        chosen = QFileDialog.getExistingDirectory(self, "Choose folder", str(current))
        if chosen:
            line_edit.setText(chosen)

    def _open_folder(self, folder: str | Path) -> None:
        path = expand_path(folder)
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def save_settings(self) -> None:
        if not all([
            self.watch_input,
            self.output_input,
            self.file_types_input,
            self.keywords_input,
            self.organization_mode_combo,
            self.copy_mode_combo,
            self.rename_checkbox,
            self.duplicates_checkbox,
            self.recursive_checkbox,
            self.enable_ocr_checkbox,
            self.confidence_input,
            self.smart_confidence_input,
            self.visual_distance_input,
            self.neuro_threshold_input,
            self.session_minutes_input,
            self.unknown_app_input,
            self.other_folder_input,
            self.review_folder_input,
            self.folder_template_input,
            self.filename_template_input,
            self.smart_subfolders_checkbox,
            self.review_low_confidence_checkbox,
            self.visual_memory_checkbox,
            self.session_memory_checkbox,
            self.neuro_learning_checkbox,
        ]):
            return

        file_types = [item.strip() for item in self.file_types_input.text().split(",") if item.strip()]
        keywords = [item.strip() for item in self.keywords_input.text().split(",") if item.strip()]

        try:
            threshold = float(self.confidence_input.text().strip())
            smart_threshold = float(self.smart_confidence_input.text().strip())
            visual_distance = int(self.visual_distance_input.text().strip())
            neuro_threshold = float(self.neuro_threshold_input.text().strip())
            session_minutes = int(self.session_minutes_input.text().strip())
        except ValueError:
            QMessageBox.warning(self, "CleanShot", "Thresholds must be numbers. ScreenshotDNA distance and session minutes must be whole numbers.")
            return

        updated = {
            **self.config,
            "watch_folder": self.watch_input.text().strip(),
            "output_folder": self.output_input.text().strip(),
            "file_types": file_types,
            "screenshot_keywords": keywords,
            "organization_mode": self._selected_mode(),
            "copy_instead_of_move": self.copy_mode_combo.currentIndex() == 1,
            "auto_rename": self.rename_checkbox.isChecked(),
            "detect_duplicates": self.duplicates_checkbox.isChecked(),
            "recursive_watch": self.recursive_checkbox.isChecked(),
            "enable_ocr": self.enable_ocr_checkbox.isChecked(),
            "smart_subfolders": self.smart_subfolders_checkbox.isChecked(),
            "review_low_confidence": self.review_low_confidence_checkbox.isChecked(),
            "enable_visual_memory": self.visual_memory_checkbox.isChecked(),
            "enable_session_memory": self.session_memory_checkbox.isChecked(),
            "enable_neuro_learning": self.neuro_learning_checkbox.isChecked(),
            "smart_confidence_threshold": smart_threshold,
            "neuro_similarity_threshold": neuro_threshold,
            "visual_similarity_distance": visual_distance,
            "session_context_minutes": session_minutes,
            "unknown_app_folder": self.unknown_app_input.text().strip() or "Unknown App",
            "other_folder": self.other_folder_input.text().strip() or "Other",
            "review_folder": self.review_folder_input.text().strip() or "_Review",
            "folder_template": self.folder_template_input.text().strip() or "{mode_folder}",
            "filename_template": self.filename_template_input.text().strip() or "screenshot-{timestamp}{extension}",
        }

        self.config = save_config(updated)
        self._restart_watcher()
        append_activity("settings", f"Settings saved · {self._mode_display(self.config.get('organization_mode', 'smart'))} mode", mode=str(self.config.get("organization_mode", "smart")))
        self.refresh_overview()
        QMessageBox.information(self, "CleanShot", "Settings saved.")

    def organize_existing(self) -> None:
        organizer = ScreenshotOrganizer(self.config)
        results = organizer.organize_existing()
        organized = len([item for item in results if item.status in {"organized", "duplicate"}])
        failed = len([item for item in results if item.status == "failed"])
        self.refresh_overview()
        QMessageBox.information(
            self,
            "CleanShot",
            f"Finished organizing existing screenshots.\n\nOrganized: {organized}\nFailed: {failed}",
        )

    def learn_from_folders(self) -> None:
        learned = learn_from_output_folders(self.config)
        append_activity("brain", f"Learned {learned} visual pattern(s) from existing folders")
        self.refresh_overview()
        QMessageBox.information(
            self,
            "CleanShot",
            f"CleanShot learned {learned} visual pattern(s).\n\nBrain file:\n{brain_path()}",
        )

    def _start_watcher(self) -> None:
        try:
            self.watcher = ScreenshotWatcher(self.config, callback=lambda result: self.organize_event.emit(result))
            self.watcher.start()
            append_activity("watcher", "Watcher started")
        except Exception as exc:  # noqa: BLE001 - show user-friendly app error
            if self.status_label:
                self.status_label.setText("Stopped")
            QMessageBox.warning(self, "CleanShot", f"Could not start watcher:\n{exc}")

    def _restart_watcher(self) -> None:
        try:
            if self.watcher:
                self.watcher.restart(self.config)
            else:
                self._start_watcher()
            append_activity("watcher", "Watcher restarted")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "CleanShot", f"Could not restart watcher:\n{exc}")

    def _on_organized(self, result: object) -> None:
        self.refresh_overview()

    def _mode_index(self, mode: str) -> int:
        for index, (value, _label) in enumerate(MODE_OPTIONS):
            if value == mode:
                return index
        return 0

    def _selected_mode(self) -> str:
        if not self.organization_mode_combo:
            return "day"
        index = self.organization_mode_combo.currentIndex()
        if 0 <= index < len(MODE_OPTIONS):
            return MODE_OPTIONS[index][0]
        return "day"

    def _mode_display(self, mode: str) -> str:
        names = {"day": "Day", "app": "App", "smart": "Smart"}
        return names.get(str(mode).lower(), "Day")

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt method name
        if self.watcher:
            self.watcher.stop()
        super().closeEvent(event)


def run_app() -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()
