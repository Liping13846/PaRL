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
  if (type === "loading") {
    statusEl.innerHTML = `<span class="spinner" aria-hidden="true"></span><span>${escapeHtml(text)}</span>`;
  } else {
    statusEl.textContent = text;
  }
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

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderLink(label, url) {
  if (!url) return "";
  return `<a class="paper-link" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${label}</a>`;
}

function renderLinks(links = {}) {
  const items = [
    renderLink("论文主页", links.primary),
    renderLink("arXiv", links.arxiv),
    renderLink("PDF", links.pdf),
    renderLink("HTML 预览", links.ar5iv),
    renderLink("DOI", links.doi),
    renderLink("OpenAlex", links.openalex),
  ].filter(Boolean);

  if (!items.length) {
    return `<span class="muted">暂无外链</span>`;
  }
  return items.join("");
}

function renderSection(title, content, extraClass = "", icon = "§") {
  if (!content) return "";
  return `
    <section class="paper-section ${extraClass}">
      <h4><span class="section-icon">${icon}</span>${title}</h4>
      <p>${escapeHtml(content)}</p>
    </section>
  `;
}

function translationCacheKey(paperId, section) {
  return `parl:${paperId}:${section}`;
}

function readLocalTranslation(paperId, section) {
  if (!paperId) return "";
  try {
    return localStorage.getItem(translationCacheKey(paperId, section)) || "";
  } catch (_error) {
    return "";
  }
}

function writeLocalTranslation(paperId, section, textZh) {
  if (!paperId || !textZh) return;
  try {
    localStorage.setItem(translationCacheKey(paperId, section), textZh);
  } catch (_error) {
    /* ignore quota errors */
  }
}

function renderExpandableBlock(textEn, textZh = "", sectionKey = "text") {
  if (!textEn) return "";
  return `
    <div class="expandable-text" data-section-key="${escapeHtml(sectionKey)}">
      <textarea class="text-source" hidden readonly>${escapeHtml(textEn)}</textarea>
      <textarea class="text-zh-source" hidden readonly>${escapeHtml(textZh)}</textarea>
      <p class="text-en">${escapeHtml(textEn)}</p>
      <button type="button" class="expand-zh-btn" aria-expanded="false" ${textZh ? 'data-loaded="true"' : ""}>
        <span class="expand-label">展开中文译文</span>
        <span class="expand-icon" aria-hidden="true">▾</span>
      </button>
      <div class="text-zh-panel hidden">
        <p class="zh-content">${textZh ? escapeHtml(textZh) : ""}</p>
      </div>
    </div>
  `;
}

function renderExpandableSection(title, textEn, icon = "§", extraClass = "", textZh = "", sectionKey = "text") {
  const block = renderExpandableBlock(textEn, textZh, sectionKey);
  if (!block) return "";
  return `
    <section class="paper-section ${extraClass}">
      <h4><span class="section-icon">${icon}</span>${title}</h4>
      ${block}
    </section>
  `;
}

function renderBibSection(bibtex, index) {
  if (!bibtex) return "";
  const bibId = `bib-${index}`;
  return `
    <section class="paper-section bib-section">
      <div class="section-heading">
        <h4><span class="section-icon">{ }</span>BibTeX 引用</h4>
        <button type="button" class="copy-btn" data-copy-target="${bibId}">复制</button>
      </div>
      <pre id="${bibId}" class="bibtex-block">${escapeHtml(bibtex)}</pre>
    </section>
  `;
}

function renderConclusionSection(paper) {
  const conclusionEn = paper.conclusion_en || "";
  const conclusionZh = paper.conclusion_zh || "";
  const source = paper.conclusion_source || "unavailable";
  const sectionTitle = paper.conclusion_section_title || "Conclusion";

  const sourceLabel =
    source === "ar5iv_conclusion"
      ? `✓ 已从 ar5iv「${escapeHtml(sectionTitle)}」章节提取`
      : source === "abstract_fallback"
        ? "该论文无 arXiv HTML 版本，以下为摘要中的结论性语句"
        : "暂无结论内容";

  let body = `<div class="figure-empty">未能提取 Conclusion 章节，请通过论文链接查看原文。</div>`;
  if (conclusionEn) {
    body = `
      <p class="conclusion-source">${sourceLabel}${paper.arxiv_id ? ` · arXiv:${escapeHtml(paper.arxiv_id)}` : ""}</p>
      ${renderExpandableBlock(conclusionEn, conclusionZh, "conclusion")}
    `;
  }

  return `
    <section class="paper-section conclusion-section">
      <h4><span class="section-icon">结</span>主要结论 · Conclusion</h4>
      ${body}
    </section>
  `;
}

function renderPaperCard(paper, index) {
  const authors = (paper.authors || []).slice(0, 6).join(", ");
  const moreAuthors =
    (paper.authors || []).length > 6
      ? ` 等 ${paper.authors.length} 人`
      : "";
  const year = paper.year || "未知年份";
  const venue = paper.venue || "未知来源";
  const score = paper.relevance_score ?? "-";
  const tier = paper.tier || "partially_relevant";
  const abstract = paper.abstract || "";
  const evidence = paper.markdown_evidence || "";
  const citations = paper.citation_count ?? 0;
  const arxivId = paper.arxiv_id || "";
  const paperId = paper.paper_id || paper.openalex_id || `paper-${index}`;
  const abstractZh = paper.abstract_zh || readLocalTranslation(paperId, "abstract") || "";
  const conclusionZh = paper.conclusion_zh || readLocalTranslation(paperId, "conclusion") || "";
  const showEvidence =
    evidence &&
    evidence !== abstract &&
    !abstract.startsWith(evidence.slice(0, Math.min(evidence.length, 80)));

  return `
    <article class="paper-card tier-${tier}" data-paper-id="${escapeHtml(paperId)}" data-arxiv-id="${escapeHtml(arxivId)}" data-openalex-id="${escapeHtml(paper.openalex_id || "")}" data-doi="${escapeHtml(paper.doi || "")}" data-index="${index}">
      <header class="paper-header">
        <div class="paper-rank">${index + 1}</div>
        <div class="paper-header-main">
          <div class="paper-badges">
            <span class="badge ${tier}">${tierLabel(tier)}</span>
            <span class="badge score">相关度 ${Number(score).toFixed ? Number(score).toFixed(2) : score}</span>
            <span class="badge meta-badge">引用 ${citations}</span>
          </div>
          <h3>${escapeHtml(paper.title || "Untitled")}</h3>
          <div class="meta">${escapeHtml(authors)}${moreAuthors} · ${year} · ${escapeHtml(venue)}</div>
          <div class="link-row">${renderLinks(paper.links || {})}</div>
        </div>
      </header>

      <div class="paper-body">
        ${renderExpandableSection("摘要 · Abstract", abstract || "暂无摘要", "摘", "", abstractZh, "abstract")}
        ${showEvidence ? renderSection("相关证据", evidence, "evidence-section", "证") : ""}
        ${renderConclusionSection({ ...paper, conclusion_zh: conclusionZh })}
        ${renderBibSection(paper.bibtex || "", index)}
        <section class="paper-section figures-section">
          <h4><span class="section-icon">图</span>论文配图</h4>
          <div class="figure-grid" data-figure-host="${escapeHtml(arxivId)}">
            ${
              arxivId
                ? `<div class="figure-loading">正在加载配图…</div>`
                : `<div class="figure-empty">非 arXiv 论文暂不支持自动配图预览。</div>`
            }
          </div>
        </section>
      </div>
    </article>
  `;
}

function toggleZhTranslation(button) {
  const container = button.closest(".expandable-text");
  if (!container) return;

  const panel = container.querySelector(".text-zh-panel");
  const content = container.querySelector(".zh-content");
  const label = button.querySelector(".expand-label");
  const isOpen = button.getAttribute("aria-expanded") === "true";

  if (isOpen) {
    panel.classList.add("hidden");
    button.setAttribute("aria-expanded", "false");
    label.textContent = "展开中文译文";
    return;
  }

  panel.classList.remove("hidden");
  button.setAttribute("aria-expanded", "true");
  label.textContent = "收起中文译文";

  const zhSource = container.querySelector(".text-zh-source");
  const cached = zhSource ? zhSource.value.trim() : "";
  if (cached) {
    content.textContent = cached;
    button.dataset.loaded = "true";
    return;
  }

  content.textContent = "译文暂未就绪，请重新检索后再试。";
}

function persistTranslationsForCard(card, paper) {
  const paperId = card.dataset.paperId || paper.paper_id || "";
  if (!paperId) return;
  if (paper.abstract_zh) {
    writeLocalTranslation(paperId, "abstract", paper.abstract_zh);
  }
  if (paper.conclusion_zh) {
    writeLocalTranslation(paperId, "conclusion", paper.conclusion_zh);
  }
}

async function loadFiguresForCard(card) {
  const arxivId = card.dataset.arxivId;
  const host = card.querySelector("[data-figure-host]");
  if (!arxivId || !host) return;

  try {
    const response = await fetch(`/api/paper-figures?arxiv_id=${encodeURIComponent(arxivId)}`);
    const body = await response.json();
    const figures = body.figures || [];

    if (!figures.length) {
      host.innerHTML = `<div class="figure-empty">暂无可用配图（可能尚未提供 ar5iv HTML 版本）。</div>`;
      return;
    }

    host.innerHTML = figures
      .map(
        (figure) => `
          <figure class="paper-figure">
            <a href="${escapeHtml(figure.url)}" target="_blank" rel="noopener noreferrer">
              <img src="${escapeHtml(figure.url)}" alt="${escapeHtml(figure.caption || "论文配图")}" loading="lazy" />
            </a>
            ${figure.caption ? `<figcaption>${escapeHtml(figure.caption)}</figcaption>` : ""}
          </figure>
        `
      )
      .join("");
  } catch (_error) {
    host.innerHTML = `<div class="figure-empty">配图加载失败，可通过上方 HTML 预览链接查看全文。</div>`;
  }
}

function renderClusterSummary(payload) {
  const clusters =
    payload.full_output?.clusters ||
    payload.clusters ||
    [];
  if (!clusters.length) return "";

  const items = clusters
    .slice(0, 3)
    .map(
      (cluster) => `
        <div class="cluster-item">
          <strong>${escapeHtml(cluster.name || "结果聚类")}</strong>
          <p>${escapeHtml(cluster.summary || "")}</p>
        </div>
      `
    )
    .join("");

  return `
    <section class="summary-panel">
      <h2>检索概览</h2>
      ${items}
    </section>
  `;
}

function renderResults(payload, stats = {}) {
  const results = payload.results || payload.full_output?.results || [];
  if (!results.length) {
    const candidateCount = stats.candidate_count;
    const hint = candidateCount
      ? `检索到 ${candidateCount} 篇候选论文，但均未通过相关性筛选。请换种表述再试。`
      : "未检索到论文，请换种表述或尝试英文关键词再试。";
    resultsEl.innerHTML = `
      <div class="empty">
        <div class="empty-icon">⌕</div>
        <p>${hint}</p>
      </div>
    `;
    return;
  }

  resultsEl.innerHTML = `
    ${renderClusterSummary(payload)}
    <div class="results-list">
      ${results.map((paper, index) => renderPaperCard(paper, index)).join("")}
    </div>
  `;

  resultsEl.querySelectorAll(".paper-card").forEach((card, index) => {
    const paper = results[index];
    if (paper) {
      persistTranslationsForCard(card, paper);
    }
    if (card.dataset.arxivId) {
      loadFiguresForCard(card);
    }
  });
  bindCopyButtons();
}

function bindCopyButtons() {
  resultsEl.querySelectorAll(".copy-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const targetId = button.dataset.copyTarget;
      const target = document.getElementById(targetId);
      if (!target) return;
      const text = target.textContent || "";
      try {
        await navigator.clipboard.writeText(text);
        const original = button.textContent;
        button.textContent = "已复制";
        setTimeout(() => {
          button.textContent = original;
        }, 1500);
      } catch (_error) {
        button.textContent = "复制失败";
      }
    });
  });
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
  searchBtn.querySelector(".search-btn-text").textContent = "检索中…";
  resultsEl.innerHTML = "";
  clearStats();
  setStatus("正在检索论文并预处理译文，请稍候…", "loading");

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
    searchBtn.querySelector(".search-btn-text").textContent = "开始搜索";
  }
}

searchBtn.addEventListener("click", runSearch);

queryEl.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    runSearch();
  }
});

resultsEl.addEventListener("click", (event) => {
  const button = event.target.closest(".expand-zh-btn");
  if (button) {
    toggleZhTranslation(button);
  }
});
