LIGHT_STYLE = """
* {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    font-size: 14px;
    color: #111827;
}

QMainWindow {
    background: #F4F6FA;
}

#Root {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #F8FAFC,
        stop: 0.48 #F4F7FB,
        stop: 1 #EEF4FF);
}

#Sidebar {
    background: rgba(255, 255, 255, 245);
    border-right: 1px solid #E4E9F2;
}

#AppTitle {
    font-size: 24px;
    font-weight: 850;
    color: #0F172A;
    letter-spacing: -0.8px;
}

#AppSubtitle {
    color: #64748B;
    font-size: 12px;
    font-weight: 650;
}

#PageTitle {
    font-size: 29px;
    font-weight: 850;
    color: #0F172A;
    letter-spacing: -1px;
}

#PageSubtitle {
    color: #64748B;
    font-size: 13px;
    font-weight: 560;
}

#NavButton {
    border: none;
    border-radius: 14px;
    padding: 13px 15px;
    text-align: left;
    color: #475569;
    background: transparent;
    font-weight: 720;
}

#NavButton:hover {
    background: #F1F5F9;
    color: #0F172A;
}

#NavButton[active="true"] {
    background: #EEF5FF;
    color: #1D4ED8;
    border-left: 3px solid #3B82F6;
}

#StatusPill {
    border-radius: 999px;
    padding: 8px 12px;
    background: #ECFDF5;
    color: #047857;
    font-size: 12px;
    font-weight: 800;
}

#Card {
    background: rgba(255, 255, 255, 248);
    border: 1px solid #E2E8F0;
    border-radius: 22px;
}

#Card:hover {
    border: 1px solid #CBD5E1;
    background: #FFFFFF;
}

#CardTitle {
    color: #64748B;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.85px;
}

#CardValue {
    color: #0F172A;
    font-size: 30px;
    font-weight: 900;
    letter-spacing: -1px;
}

#SectionTitle {
    font-size: 18px;
    font-weight: 850;
    color: #0F172A;
    letter-spacing: -0.3px;
}

#MutedText {
    color: #64748B;
    font-size: 13px;
    font-weight: 560;
}

#SmallMutedText {
    color: #64748B;
    font-size: 12px;
    font-weight: 620;
}

QPushButton {
    border: 1px solid #CBD5E1;
    background: #FFFFFF;
    border-radius: 12px;
    padding: 10px 14px;
    font-weight: 760;
    color: #0F172A;
}

QPushButton:hover {
    background: #F8FAFC;
    border-color: #94A3B8;
}

QPushButton:pressed {
    background: #E2E8F0;
}

#PrimaryButton {
    background: #2563EB;
    border: 1px solid #2563EB;
    color: #FFFFFF;
}

#PrimaryButton:hover {
    background: #1D4ED8;
    border-color: #1D4ED8;
}

#DangerButton {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    color: #B91C1C;
}

QLineEdit, QComboBox {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 12px;
    padding: 11px 13px;
    color: #0F172A;
    selection-background-color: #DBEAFE;
}

QLineEdit:focus, QComboBox:focus {
    border: 1px solid #2563EB;
    background: #FFFFFF;
}

QCheckBox {
    spacing: 10px;
    color: #334155;
    font-weight: 650;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 6px;
    border: 1px solid #CBD5E1;
    background: #FFFFFF;
}

QCheckBox::indicator:hover {
    border-color: #2563EB;
}

QCheckBox::indicator:checked {
    background: #2563EB;
    border: 1px solid #2563EB;
}

QProgressBar {
    border: none;
    border-radius: 7px;
    background: #E2E8F0;
    height: 10px;
    text-align: center;
}

QProgressBar::chunk {
    border-radius: 7px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #3B82F6,
        stop:1 #14B8A6);
}

QScrollArea, #TransparentWidget {
    border: none;
    background: transparent;
}

QScrollBar:vertical {
    width: 10px;
    background: transparent;
}

QScrollBar::handle:vertical {
    background: #CBD5E1;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #94A3B8;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

#Divider {
    background: #E2E8F0;
    max-height: 1px;
}

#ActivityRow {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
}

#ActivityRow:hover {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
}

#ActivityTime {
    color: #64748B;
    font-size: 12px;
    font-weight: 800;
}

#ActivityMessage {
    color: #1E293B;
    font-size: 13px;
    font-weight: 760;
}

#ActivityDetail {
    color: #64748B;
    font-size: 12px;
    font-weight: 560;
}

#ActionBadge {
    border-radius: 999px;
    padding: 6px 9px;
    font-size: 11px;
    font-weight: 850;
    background: #EFF6FF;
    color: #1D4ED8;
}

#ActionBadge[kind="organized"] {
    background: #ECFDF5;
    color: #047857;
}

#ActionBadge[kind="duplicate"] {
    background: #FFFBEB;
    color: #B45309;
}

#ActionBadge[kind="error"] {
    background: #FEF2F2;
    color: #B91C1C;
}

#ActionBadge[kind="teach"], #ActionBadge[kind="brain"] {
    background: #F5F3FF;
    color: #6D28D9;
}

#ActionBadge[kind="watcher"] {
    background: #F0FDFA;
    color: #0F766E;
}

#ConfidencePill {
    border-radius: 999px;
    padding: 4px 8px;
    font-size: 11px;
    font-weight: 850;
    background: #EEF2FF;
    color: #3730A3;
}
"""
