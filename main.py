"""
Intelligent Currency Converter Agent
Mini Project - AI Course
Team: 4 Members

Agent uses BFS, DFS, UCS, and A* to find optimal currency conversion paths
through a weighted currency graph.
"""

import sys
import heapq
import random
import math
from collections import deque
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QLineEdit, QPushButton, QTextEdit, QGroupBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QProgressBar, QSplitter, QMessageBox, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ─────────────────────────────────────────────
#  CURRENCY GRAPH (weighted edges = fees/cost)
# ─────────────────────────────────────────────
CURRENCY_GRAPH = {
    "PKR": {"USD": 0.0036, "AED": 0.013,  "SAR": 0.013,  "CNY": 0.026},
    "USD": {"PKR": 278.5,  "EUR": 0.92,   "GBP": 0.79,   "AED": 3.67,
            "JPY": 149.5,  "CNY": 7.24,   "SAR": 3.75,   "CAD": 1.36},
    "EUR": {"USD": 1.09,   "GBP": 0.86,   "PKR": 303.8,  "JPY": 162.5,
            "CAD": 1.48,   "AED": 3.99,   "CNY": 7.88},
    "GBP": {"USD": 1.27,   "EUR": 1.16,   "PKR": 353.2,  "JPY": 189.3,
            "AED": 4.65,   "CAD": 1.73},
    "AED": {"USD": 0.272,  "PKR": 75.9,   "EUR": 0.251,  "SAR": 1.02,
            "CNY": 1.97,   "GBP": 0.215},
    "JPY": {"USD": 0.0067, "EUR": 0.0062, "GBP": 0.0053, "CNY": 0.048},
    "CNY": {"USD": 0.138,  "EUR": 0.127,  "PKR": 38.5,   "JPY": 20.7,
            "AED": 0.508},
    "SAR": {"USD": 0.267,  "PKR": 74.3,   "AED": 0.98,   "EUR": 0.245},
    "CAD": {"USD": 0.735,  "EUR": 0.676,  "GBP": 0.578,  "JPY": 109.8},
}

CURRENCY_NAMES = {
    "PKR": "Pakistani Rupee",
    "USD": "US Dollar",
    "EUR": "Euro",
    "GBP": "British Pound",
    "AED": "UAE Dirham",
    "JPY": "Japanese Yen",
    "CNY": "Chinese Yuan",
    "SAR": "Saudi Riyal",
    "CAD": "Canadian Dollar",
}

CURRENCY_FLAGS = {
    "PKR": "🇵🇰", "USD": "🇺🇸", "EUR": "🇪🇺", "GBP": "🇬🇧",
    "AED": "🇦🇪", "JPY": "🇯🇵", "CNY": "🇨🇳", "SAR": "🇸🇦", "CAD": "🇨🇦",
}


# ─────────────────────────────────────────────
#  AI AGENT - SEARCH ALGORITHMS
# ─────────────────────────────────────────────
class CurrencyAgent:
    """Autonomous agent that finds optimal currency conversion paths."""

    def __init__(self, graph):
        self.graph = graph

    def convert_along_path(self, path, amount):
        result = amount
        for i in range(len(path) - 1):
            result *= self.graph[path[i]][path[i + 1]]
        return result

    # ── BFS ──────────────────────────────────
    def bfs(self, source, target):
        if source == target:
            return [source], 0, [source]
        queue = deque([(source, [source])])
        visited = {source}
        nodes_explored = []
        while queue:
            current, path = queue.popleft()
            nodes_explored.append(current)
            for neighbor in self.graph.get(current, {}):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    if neighbor == target:
                        nodes_explored.append(neighbor)
                        return new_path, len(nodes_explored), nodes_explored
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        return None, len(nodes_explored), nodes_explored

    # ── DFS ──────────────────────────────────
    def dfs(self, source, target):
        nodes_explored = []

        def _dfs(current, path, visited):
            nodes_explored.append(current)
            if current == target:
                return path
            visited.add(current)
            for neighbor in self.graph.get(current, {}):
                if neighbor not in visited:
                    result = _dfs(neighbor, path + [neighbor], visited)
                    if result:
                        return result
            return None

        result = _dfs(source, [source], set())
        return result, len(nodes_explored), nodes_explored

    # ── UCS (Uniform Cost Search) ─────────────
    def ucs(self, source, target):
        # Cost = negative log of rate (minimize cost = maximize conversion)
        import math
        pq = [(0, source, [source])]
        visited = {}
        nodes_explored = []
        while pq:
            cost, current, path = heapq.heappop(pq)
            if current in visited:
                continue
            visited[current] = cost
            nodes_explored.append(current)
            if current == target:
                return path, len(nodes_explored), nodes_explored
            for neighbor, rate in self.graph.get(current, {}).items():
                if neighbor not in visited:
                    edge_cost = -math.log(rate) if rate > 0 else float('inf')
                    heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))
        return None, len(nodes_explored), nodes_explored

    # ── A* Search ────────────────────────────
    def astar(self, source, target):
        import math

        def heuristic(node):
            # Heuristic: direct rate exists? 0, else small penalty
            if target in self.graph.get(node, {}):
                return 0
            return 0.5

        pq = [(0 + heuristic(source), 0, source, [source])]
        visited = {}
        nodes_explored = []
        while pq:
            f, g, current, path = heapq.heappop(pq)
            if current in visited:
                continue
            visited[current] = g
            nodes_explored.append(current)
            if current == target:
                return path, len(nodes_explored), nodes_explored
            for neighbor, rate in self.graph.get(current, {}).items():
                if neighbor not in visited:
                    edge_cost = -math.log(rate) if rate > 0 else float('inf')
                    new_g = g + edge_cost
                    heapq.heappush(pq, (new_g + heuristic(neighbor), new_g, neighbor, path + [neighbor]))
        return None, len(nodes_explored), nodes_explored

    def run_all(self, source, target, amount):
        """Run all algorithms and return comparison results."""
        import time
        results = {}
        algos = {
            "BFS":  self.bfs,
            "DFS":  self.dfs,
            "UCS":  self.ucs,
            "A*":   self.astar,
        }
        for name, fn in algos.items():
            start = time.perf_counter()
            path, nodes, explored = fn(source, target)
            elapsed = (time.perf_counter() - start) * 1000
            if path:
                converted = self.convert_along_path(path, amount)
            else:
                converted = None
            results[name] = {
                "path": path,
                "converted": converted,
                "nodes": nodes,
                "time_ms": round(elapsed, 4),
                "explored": explored,
            }
        return results

    # ── Autonomous Decision ───────────────────
    def choose_best_algorithm(self, source, target):
        """
        Autonomous agent decision:
        - Direct edge exists → BFS (fast, fewest hops)
        - No direct edge    → A* (heuristic-guided optimal path)
        """
        if target in self.graph.get(source, {}):
            return "BFS", "Direct conversion available — BFS selected (optimal for single-hop)"
        else:
            return "A*", "No direct route — A* selected (heuristic search for best multi-hop path)"


# ─────────────────────────────────────────────
#  TREND DIALOG (matplotlib popup)
# ─────────────────────────────────────────────
class TrendDialog(QDialog):
    """Popup window showing 7-day exchange rate trend using matplotlib."""

    def __init__(self, from_code, to_code, base_rate, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"📉 7-Day Trend: {from_code} → {to_code}")
        self.setMinimumSize(700, 420)
        self.setStyleSheet("""
            QDialog { background-color: #0f0f1a; color: #e0e0f0; }
            QLabel  { color: #8080ff; font-size: 13px; font-weight: bold; }
            QPushButton {
                background: #2a2a5a; color: white; border-radius: 8px;
                padding: 8px 18px; font-size: 13px;
            }
            QPushButton:hover { background: #3a3a7a; }
        """)
        layout = QVBoxLayout(self)

        # Generate realistic 7-day data with small variance
        random.seed(42)
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        rates = [base_rate * (1 + random.uniform(-0.02, 0.02)) for _ in days]
        rates[-1] = base_rate  # today = actual rate

        # Matplotlib figure with dark theme
        fig = Figure(figsize=(6.5, 3.5), facecolor="#0a0a18")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#0a0a18")

        ax.plot(days, rates, color="#6060ff", linewidth=2.5, marker="o",
                markersize=7, markerfacecolor="#60ff90", markeredgecolor="#6060ff")
        ax.fill_between(days, rates, min(rates) * 0.998,
                        alpha=0.15, color="#6060ff")

        ax.set_title(f"{from_code} → {to_code}  |  Last 7 Days",
                     color="#c0c0ff", fontsize=13, pad=12)
        ax.set_xlabel("Day", color="#8080aa", fontsize=11)
        ax.set_ylabel("Exchange Rate", color="#8080aa", fontsize=11)
        ax.tick_params(colors="#9090bb")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")
        ax.yaxis.label.set_color("#8080aa")
        ax.xaxis.label.set_color("#8080aa")
        fig.tight_layout(pad=1.5)

        canvas = FigureCanvas(fig)
        layout.addWidget(QLabel(f"  Exchange Rate Trend — {from_code} / {to_code}"))
        layout.addWidget(canvas)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)


# ─────────────────────────────────────────────
#  PYQT5 GUI
# ─────────────────────────────────────────────
STYLE = """
QMainWindow {
    background-color: #0f0f1a;
}
QWidget {
    background-color: #0f0f1a;
    color: #e0e0f0;
    font-family: 'Segoe UI', Arial;
}
QGroupBox {
    border: 1.5px solid #2a2a4a;
    border-radius: 10px;
    margin-top: 12px;
    padding: 10px;
    font-weight: bold;
    font-size: 13px;
    color: #8080ff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QComboBox, QLineEdit {
    background-color: #1a1a2e;
    border: 1.5px solid #2a2a5a;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    color: #e0e0ff;
}
QComboBox:focus, QLineEdit:focus {
    border-color: #6060ff;
}
QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #4040cc, stop:1 #7040ff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: bold;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #5050dd, stop:1 #8050ff);
}
QPushButton:pressed {
    background: #3030aa;
}
QPushButton#clearBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #444, stop:1 #666);
}
QTextEdit {
    background-color: #0a0a18;
    border: 1.5px solid #2a2a4a;
    border-radius: 8px;
    padding: 8px;
    font-family: 'Consolas', monospace;
    font-size: 12px;
    color: #c0c0ff;
}
QTableWidget {
    background-color: #0a0a18;
    border: 1.5px solid #2a2a4a;
    border-radius: 8px;
    gridline-color: #1a1a3a;
    font-size: 12px;
}
QHeaderView::section {
    background-color: #1a1a3a;
    color: #8080ff;
    padding: 6px;
    border: none;
    font-weight: bold;
}
QTableWidget::item {
    padding: 4px;
}
QTableWidget::item:selected {
    background-color: #2a2a5a;
}
QTabWidget::pane {
    border: 1.5px solid #2a2a4a;
    border-radius: 8px;
}
QTabBar::tab {
    background: #1a1a2e;
    color: #8080cc;
    padding: 8px 18px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #2a2a5a;
    color: #ffffff;
    font-weight: bold;
}
QLabel#resultLabel {
    font-size: 28px;
    font-weight: bold;
    color: #60ff90;
    padding: 10px;
}
QLabel#subtitleLabel {
    font-size: 12px;
    color: #8080aa;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.agent = CurrencyAgent(CURRENCY_GRAPH)
        self._last_from = None
        self._last_to = None
        self._last_rate = None
        self.setWindowTitle("💱 Intelligent Currency Converter Agent")
        self.setMinimumSize(1000, 720)
        self.setStyleSheet(STYLE)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        # ── Header ──
        header = QLabel("💱  Intelligent Currency Converter Agent")
        header.setFont(QFont("Segoe UI", 20, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #8080ff; padding: 6px;")
        root.addWidget(header)

        sub = QLabel("AI-Powered • Autonomous • Multi-Algorithm Search")
        sub.setAlignment(Qt.AlignCenter)
        sub.setObjectName("subtitleLabel")
        root.addWidget(sub)

        # ── Input Panel ──
        input_group = QGroupBox("⚙️  Agent Input")
        ig_layout = QHBoxLayout(input_group)
        ig_layout.setSpacing(14)

        # From currency
        from_box = QVBoxLayout()
        from_box.addWidget(QLabel("From Currency"))
        self.from_combo = QComboBox()
        for code in CURRENCY_GRAPH:
            self.from_combo.addItem(f"{CURRENCY_FLAGS[code]} {code} — {CURRENCY_NAMES[code]}", code)
        self.from_combo.setCurrentText("🇵🇰 PKR — Pakistani Rupee")
        from_box.addWidget(self.from_combo)
        ig_layout.addLayout(from_box)

        # Amount
        amt_box = QVBoxLayout()
        amt_box.addWidget(QLabel("Amount"))
        self.amount_input = QLineEdit("1000")
        self.amount_input.setPlaceholderText("Enter amount...")
        self.amount_input.returnPressed.connect(self.run_agent)
        amt_box.addWidget(self.amount_input)
        ig_layout.addLayout(amt_box)

        # Arrow
        arrow = QLabel("→")
        arrow.setFont(QFont("Segoe UI", 22, QFont.Bold))
        arrow.setStyleSheet("color: #6060ff;")
        arrow.setAlignment(Qt.AlignCenter)
        ig_layout.addWidget(arrow)

        # To currency
        to_box = QVBoxLayout()
        to_box.addWidget(QLabel("To Currency"))
        self.to_combo = QComboBox()
        for code in CURRENCY_GRAPH:
            self.to_combo.addItem(f"{CURRENCY_FLAGS[code]} {code} — {CURRENCY_NAMES[code]}", code)
        self.to_combo.setCurrentText("🇺🇸 USD — US Dollar")
        to_box.addWidget(self.to_combo)
        ig_layout.addLayout(to_box)

        # Convert button
        self.convert_btn = QPushButton("🚀  Run Agent")
        self.convert_btn.setMinimumHeight(44)
        self.convert_btn.clicked.connect(self.run_agent)
        ig_layout.addWidget(self.convert_btn)

        # Swap button
        self.swap_btn = QPushButton("🔄 Swap")
        self.swap_btn.setObjectName("clearBtn")
        self.swap_btn.setMinimumHeight(44)
        self.swap_btn.setToolTip("Swap FROM and TO currencies")
        self.swap_btn.clicked.connect(self.swap_currencies)
        ig_layout.addWidget(self.swap_btn)

        # Trend button
        self.trend_btn = QPushButton("📉 Trend")
        self.trend_btn.setObjectName("clearBtn")
        self.trend_btn.setMinimumHeight(44)
        self.trend_btn.setToolTip("Show 7-day exchange rate trend")
        self.trend_btn.clicked.connect(self.show_trend)
        ig_layout.addWidget(self.trend_btn)

        # Clear button
        self.clear_btn = QPushButton("🗑  Clear")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.setMinimumHeight(44)
        self.clear_btn.clicked.connect(self.clear_all)
        ig_layout.addWidget(self.clear_btn)

        root.addWidget(input_group)

        # ── Result Display ──
        result_group = QGroupBox("📊  Agent Result")
        rg_layout = QVBoxLayout(result_group)

        self.result_label = QLabel("—")
        self.result_label.setObjectName("resultLabel")
        self.result_label.setAlignment(Qt.AlignCenter)
        rg_layout.addWidget(self.result_label)

        self.path_label = QLabel("Run the agent to see conversion path")
        self.path_label.setAlignment(Qt.AlignCenter)
        self.path_label.setStyleSheet("color: #9090bb; font-size: 13px;")
        rg_layout.addWidget(self.path_label)

        self.comparison_label = QLabel("")
        self.comparison_label.setAlignment(Qt.AlignCenter)
        self.comparison_label.setStyleSheet("color: #ffcc60; font-size: 12px; padding: 4px;")
        rg_layout.addWidget(self.comparison_label)

        root.addWidget(result_group)

        # ── Tabs: Comparison + Log ──
        tabs = QTabWidget()

        # Tab 1: Algorithm Comparison Table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Algorithm", "Path", "Converted Amount", "Nodes Explored", "Time (ms)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        tabs.addTab(self.table, "📈  Algorithm Comparison")

        # Tab 2: Agent Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("Agent execution log will appear here...")
        tabs.addTab(self.log_text, "🧠  Agent Log")

        # Tab 3: Currency Graph
        self.graph_text = QTextEdit()
        self.graph_text.setReadOnly(True)
        self._populate_graph_view()
        tabs.addTab(self.graph_text, "🌐  Currency Graph")

        root.addWidget(tabs)

        # ── Status bar ──
        self.statusBar().setStyleSheet("color: #6060aa; background: #0a0a18; padding: 4px;")
        self.statusBar().showMessage("Ready — Enter amount and currencies, then click Run Agent")

    def _populate_graph_view(self):
        text = "CURRENCY EXCHANGE GRAPH\n"
        text += "=" * 50 + "\n\n"
        for src, neighbors in CURRENCY_GRAPH.items():
            text += f"{CURRENCY_FLAGS[src]} {src} ({CURRENCY_NAMES[src]})\n"
            for dst, rate in neighbors.items():
                text += f"    → {CURRENCY_FLAGS[dst]} {dst}  :  {rate}\n"
            text += "\n"
        self.graph_text.setText(text)

    def run_agent(self):
        from_code = self.from_combo.currentData()
        to_code = self.to_combo.currentData()

        # ── INPUT VALIDATION ──────────────────────────────
        raw = self.amount_input.text().strip()
        try:
            amount = float(raw)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input",
                "⚠️  Please enter a valid numeric amount.\nExample: 1000 or 250.5")
            self.amount_input.setFocus()
            return

        if amount <= 0:
            QMessageBox.warning(self, "Invalid Amount",
                "⚠️  Amount must be greater than 0.")
            self.amount_input.setFocus()
            return

        if from_code == to_code:
            self.result_label.setText(f"{amount:,.4f} {to_code}")
            self.path_label.setText("Same currency — no conversion needed.")
            self.comparison_label.setText("")
            return

        self.statusBar().showMessage("🤖  Agent analysing problem...")
        self.convert_btn.setEnabled(False)
        QApplication.processEvents()

        # ── AUTONOMOUS ALGORITHM DECISION ─────────────────
        chosen_algo, decision_reason = self.agent.choose_best_algorithm(from_code, to_code)

        # ── RUN ALL ALGORITHMS ────────────────────────────
        results = self.agent.run_all(from_code, to_code, amount)

        # Primary result = agent's chosen algorithm
        primary = results.get(chosen_algo) or results.get("A*") or list(results.values())[0]

        # Store for trend button
        self._last_from = from_code
        self._last_to = to_code
        direct_rate = CURRENCY_GRAPH.get(from_code, {}).get(to_code)
        self._last_rate = direct_rate if direct_rate else (
            primary["converted"] / amount if primary["converted"] else 1.0
        )

        if primary["converted"] is not None:
            self.result_label.setText(
                f"{primary['converted']:,.4f} {to_code}  [{chosen_algo}]"
            )
            path_str = "  →  ".join(
                [f"{CURRENCY_FLAGS[c]} {c}" for c in primary["path"]]
            )
            self.path_label.setText(f"Agent Path ({chosen_algo}): {path_str}")

            # ── DIRECT vs OPTIMIZED COMPARISON ───────────
            if direct_rate is not None:
                direct_val = amount * direct_rate
                optimized_val = primary["converted"]
                saved = optimized_val - direct_val
                sign = "+" if saved >= 0 else ""
                self.comparison_label.setText(
                    f"Direct: {direct_val:,.4f} {to_code}  │  "
                    f"Optimized ({chosen_algo}): {optimized_val:,.4f} {to_code}  │  "
                    f"Difference: {sign}{saved:,.4f} {to_code}"
                )
            else:
                self.comparison_label.setText(
                    f"No direct route available — Agent used {chosen_algo} for best path"
                )
        else:
            self.result_label.setText("No path found")
            self.path_label.setText("Agent could not find a conversion route.")
            self.comparison_label.setText("")

        # ── POPULATE TABLE ────────────────────────────────
        self.table.setRowCount(0)
        for algo, res in results.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(algo))
            path_display = " → ".join(res["path"]) if res["path"] else "No path"
            self.table.setItem(row, 1, QTableWidgetItem(path_display))
            amount_display = f"{res['converted']:,.4f} {to_code}" if res["converted"] else "N/A"
            self.table.setItem(row, 2, QTableWidgetItem(amount_display))
            self.table.setItem(row, 3, QTableWidgetItem(str(res["nodes"])))
            self.table.setItem(row, 4, QTableWidgetItem(f"{res['time_ms']} ms"))

            # Highlight chosen algorithm row
            if algo == chosen_algo:
                for col in range(5):
                    item = self.table.item(row, col)
                    if item:
                        item.setBackground(QColor("#1a2a1a"))
                        item.setForeground(QColor("#60ff90"))

        # ── POPULATE LOG ──────────────────────────────────
        log = []
        log.append("=" * 58)
        log.append("  AGENT EXECUTION LOG")
        log.append("=" * 58)
        log.append(f"  Goal      : Convert {amount} {from_code} → {to_code}")
        log.append(f"  Planner   : Multi-Algorithm Search")
        log.append(f"  Decision  : {chosen_algo} chosen based on problem analysis")
        log.append(f"  Reason    : {decision_reason}")
        log.append("")
        for algo, res in results.items():
            marker = " ◀ SELECTED" if algo == chosen_algo else ""
            log.append(f"  [{algo}]{marker}")
            log.append(f"    Explored  : {' → '.join(res['explored'])}")
            log.append(f"    Final Path: {' → '.join(res['path']) if res['path'] else 'None'}")
            if res['converted']:
                log.append(f"    Result    : {res['converted']:,.6f} {to_code}")
            else:
                log.append(f"    Result    : N/A")
            log.append(f"    Nodes     : {res['nodes']}")
            log.append(f"    Time      : {res['time_ms']} ms")
            log.append("")
        log.append(f"  Agent selected {chosen_algo} based on problem analysis")
        log.append("=" * 58)
        self.log_text.setText("\n".join(log))

        self.convert_btn.setEnabled(True)
        self.statusBar().showMessage(
            f"✅  Agent completed — {chosen_algo} selected | {decision_reason}"
        )

    def swap_currencies(self):
        """Swap FROM and TO currency selections."""
        from_idx = self.from_combo.currentIndex()
        to_idx = self.to_combo.currentIndex()
        self.from_combo.setCurrentIndex(to_idx)
        self.to_combo.setCurrentIndex(from_idx)

    def show_trend(self):
        """Show 7-day trend popup for current currency pair."""
        from_code = self.from_combo.currentData()
        to_code = self.to_combo.currentData()
        if from_code == to_code:
            QMessageBox.information(self, "Same Currency",
                "Please select two different currencies to view trend.")
            return
        # Use last known rate or direct rate
        rate = getattr(self, "_last_rate", None)
        if rate is None:
            rate = CURRENCY_GRAPH.get(from_code, {}).get(to_code)
        if rate is None:
            # Approximate via USD bridge
            usd_from = CURRENCY_GRAPH.get(from_code, {}).get("USD", 1.0)
            usd_to = CURRENCY_GRAPH.get("USD", {}).get(to_code, 1.0)
            rate = usd_from * usd_to
        dialog = TrendDialog(from_code, to_code, rate, parent=self)
        dialog.exec_()

    def clear_all(self):
        self.result_label.setText("—")
        self.path_label.setText("Run the agent to see conversion path")
        self.comparison_label.setText("")
        self.table.setRowCount(0)
        self.log_text.clear()
        self.amount_input.setText("1000")
        self._last_rate = None
        self.statusBar().showMessage("Cleared — Ready for new conversion")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
