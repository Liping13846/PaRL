const queryEl = document.getElementById("query");
const searchBtn = document.getElementById("searchBtn");
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const resultsEl = document.getElementById("results");

function getMode() {
  const checked = document.querySelector('input[name="mode"]:checked');
  return checked ? checked.value : "lite";
}

function setStatus(text, type = "loading") {
  statusEl.className = `status ${type}`;
  statusEl.textContent = text;
  statusEl.classList.remove("hidden");
}

function clearStats() {
  statsEl.innerHTML = "";
  statsEl.classList.add("hidden");
}

function renderStats(stats = {}) {
  const items = [
    ["结果数", stats.result_count ?? "-"],
    ["候选数", stats.candidate_count ?? "-"],
    ["API 调用", stats.api_calls ?? "-"],
    ["OpenAlex", stats.openalex_api_calls ?? "-"],
    ["S2 调用", stats.s2_api_calls ?? "-"],
    ["主检索源", stats.primary_backend ?? "-"],
  ];

  statsEl.innerHTML = items
    .map(
      ([label, value]) =>
        `<div class="stat-item"><span>${label}</span><strong>${value}</strong></div>`
    )
    .join("");
  statsEl.classList.remove("hidden");
}

function tierLabel(tier) {
  if (tier === "highly_relevant") return "高度相关";
  if (tier === "partially_relevant") return "部分相关";
  return "相关";
}

function renderResults(payload, stats = {}) {
  const results = payload.results || payload.full_output?.results || [];
  if (!results.length) {
    const candidateCount = stats.candidate_count;
    const hint = candidateCount
      ? `检索到 ${candidateCount} 篇候选论文，但均未通过相关性筛选。请换种表述再试。`
      : "未检索到论文，请换种表述或尝试英文关键词再试。";
    resultsEl.innerHTML = `<div class="empty">${hint}</div>`;
    return;
  }

  resultsEl.innerHTML = results
    .map((paper, index) => {
      const authors = (paper.authors || []).slice(0, 5).join(", ");
      const year = paper.year || "未知年份";
      const venue = paper.venue || "未知来源";
      const score = paper.relevance_score ?? "-";
      const tier = paper.tier || "partially_relevant";
      const evidence = paper.markdown_evidence || "暂无摘要证据";

      return `
        <article class="paper-card">
          <span class="badge ${tier}">${tierLabel(tier)}</span>
          <span class="badge">${Number(score).toFixed ? Number(score).toFixed(2) : score}</span>
          <h3>${index + 1}. ${escapeHtml(paper.title || "Untitled")}</h3>
          <div class="meta">${escapeHtml(authors)} · ${year} · ${escapeHtml(venue)}</div>
          <p class="evidence">${escapeHtml(evidence)}</p>
        </article>
      `;
    })
    .join("");
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function getConfig() {
  const el = document.getElementById("config");
  return el ? el.value : "dev";
}

async function runSearch() {
  const query = queryEl.value.trim();
  if (!query) {
    setStatus("请输入查询内容。", "error");
    return;
  }

  searchBtn.disabled = true;
  resultsEl.innerHTML = "";
  clearStats();
  setStatus("正在检索论文，请稍候…", "loading");

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        mode: getMode(),
        config: getConfig(),
      }),
    });

    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "搜索失败");
    }

    const payload = body.data;
    const stats = payload.stats || payload.full_output?.stats || {};
    renderStats(stats);
    renderResults(payload, stats);
    setStatus(`搜索完成，模式：${payload.mode || getMode()}`, "loading");
    statusEl.className = "status hidden";
  } catch (error) {
    setStatus(`搜索失败：${error.message}`, "error");
  } finally {
    searchBtn.disabled = false;
  }
}

searchBtn.addEventListener("click", runSearch);

queryEl.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    runSearch();
  }
});
