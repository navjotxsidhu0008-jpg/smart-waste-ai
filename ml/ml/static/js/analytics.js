/**
 * ♻️ AI Smart Waste Management System — Analytics Dashboard Script (static/js/analytics.js)
 *
 * Powers:
 * - Dynamic CSS conic-gradient Doughnut Chart & Recyclability Share calculation
 * - Live refresh of SQLite statistics & recent detection records
 * - One-click CSV export of detection history
 * - Database reset control
 */

document.addEventListener("DOMContentLoaded", () => {
    const refreshBtn = document.getElementById("analytics-refresh-btn");
    const exportBtn = document.getElementById("analytics-export-btn");
    const clearBtn = document.getElementById("analytics-clear-btn");

    const CATEGORY_COLORS = {
        "Plastic": "#3b82f6",
        "Paper": "#0ea5e9",
        "Metal": "#f59e0b",
        "Glass": "#06b6d4",
        "Organic": "#10b981",
        "General Waste": "#64748b"
    };

    async function loadAnalyticsData() {
        try {
            const [statsRes, detRes] = await Promise.all([
                fetch("/api/stats"),
                fetch("/api/detections?limit=50")
            ]);

            if (statsRes.ok) {
                const stats = await statsRes.json();
                updateStatsAndCharts(stats);
            }

            if (detRes.ok) {
                const detData = await detRes.json();
                updateAnalyticsTable(detData.detections || []);
            }
        } catch (err) {
            console.error("Error loading analytics data:", err);
        }
    }

    function updateStatsAndCharts(stats) {
        // Update Top KPI Cards
        setText("kpi-total", stats.total_detections);
        setText("kpi-today", stats.today_detections);
        setText("kpi-top-category", stats.most_frequent_category || "None yet");
        setText("kpi-top-count", stats.most_frequent_count || 0);
        setText("chart-total-badge", `${stats.total_detections} Total`);
        setText("doughnut-total", stats.total_detections);

        const counts = stats.category_counts || {};
        const dist = stats.distribution || {};
        const total = stats.total_detections || 0;

        // Update Horizontal Bars, Legend & Individual Category Cards
        Object.keys(CATEGORY_COLORS).forEach((cat) => {
            const safeKey = cat.replace(/\s+/g, "-");
            const countVal = counts[cat] || 0;
            const pctVal = dist[cat] ? dist[cat].percentage : 0.0;

            setText(`dist-count-${safeKey}`, countVal);
            setText(`dist-pct-${safeKey}`, `${pctVal}%`);
            setText(`legend-val-${safeKey}`, countVal);
            setText(`analytics-card-${safeKey}`, countVal);

            const fillEl = document.getElementById(`dist-fill-${safeKey}`);
            if (fillEl) {
                fillEl.style.width = `${Math.min(100, Math.max(0, pctVal))}%`;
            }
        });

        // Update Doughnut Ring Conic Gradient
        const doughnutEl = document.getElementById("doughnut-ring");
        if (doughnutEl) {
            if (total === 0) {
                doughnutEl.style.background = "conic-gradient(#e2e8f0 0% 100%)";
            } else {
                let cumulativePct = 0;
                const segments = [];
                Object.entries(CATEGORY_COLORS).forEach(([cat, color]) => {
                    const cnt = counts[cat] || 0;
                    if (cnt > 0) {
                        const slicePct = (cnt / total) * 100;
                        const start = cumulativePct.toFixed(2);
                        cumulativePct += slicePct;
                        const end = cumulativePct.toFixed(2);
                        segments.push(`${color} ${start}% ${end}%`);
                    }
                });
                doughnutEl.style.background = `conic-gradient(${segments.join(", ")})`;
            }
        }

        // Compute Recyclable & Compostable Share (All except General Waste)
        const generalCount = counts["General Waste"] || 0;
        const recyclableCount = Math.max(0, total - generalCount);
        const recyclableShare = total > 0 ? ((recyclableCount / total) * 100).toFixed(1) : "0.0";
        setText("recyclable-share-pct", `${recyclableShare}%`);
    }

    function updateAnalyticsTable(detections) {
        const tbody = document.getElementById("analytics-tbody");
        setText("table-record-count", `${detections.length} records shown`);
        if (!tbody) return;

        if (!detections.length) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="5" class="empty-table-msg">
                        No detections found in database. Go to the <a href="/">Live AI Scanner</a> to classify waste items!
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = detections
            .map((item) => {
                const formattedTime = String(item.timestamp).replace("T", " ");
                return `
                    <tr>
                        <td>#${item.id}</td>
                        <td class="td-time">${escapeHtml(formattedTime)}</td>
                        <td><span class="category-pill" data-category="${escapeHtml(item.waste_type)}">${escapeHtml(item.waste_type)}</span></td>
                        <td><strong>${Number(item.confidence).toFixed(1)}%</strong></td>
                        <td>${escapeHtml(item.recommended_bin)}</td>
                    </tr>
                `;
            })
            .join("");
    }

    async function exportDetectionsCsv() {
        try {
            const res = await fetch("/api/detections?limit=200");
            if (!res.ok) return;
            const data = await res.json();
            const rows = data.detections || [];

            if (!rows.length) {
                alert("No detection records available to export yet.");
                return;
            }

            const headers = ["ID", "Timestamp", "Waste_Type", "Confidence_Percent", "Recommended_Bin"];
            const csvLines = [
                headers.join(","),
                ...rows.map((r) =>
                    [
                        r.id,
                        `"${r.timestamp}"`,
                        `"${r.waste_type}"`,
                        r.confidence,
                        `"${r.recommended_bin}"`
                    ].join(",")
                )
            ];

            const blob = new Blob([csvLines.join("\n")], { type: "text/csv;charset=utf-8;" });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = `smart_waste_detections_${new Date().toISOString().slice(0, 10)}.csv`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        } catch (err) {
            console.error("CSV Export failed:", err);
        }
    }

    function setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    if (refreshBtn) {
        refreshBtn.addEventListener("click", loadAnalyticsData);
    }

    if (exportBtn) {
        exportBtn.addEventListener("click", exportDetectionsCsv);
    }

    if (clearBtn) {
        clearBtn.addEventListener("click", async () => {
            if (!confirm("Are you sure you want to delete all detection records from the SQLite database?")) {
                return;
            }
            const res = await fetch("/api/detections", { method: "DELETE" });
            if (res.ok) {
                await loadAnalyticsData();
            }
        });
    }

    // Initial calculation for doughnut chart on page load
    loadAnalyticsData();
});
