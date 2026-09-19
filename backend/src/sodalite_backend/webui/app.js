"use strict";

// Same-origin: the FastAPI backend serves this page and the API, so relative
// URLs suffice and no CORS handling is needed.
const API = "/api/v1";
const POLL_INTERVAL_MS = 400;

const $ = (id) => document.getElementById(id);

const els = {
  prompt: $("prompt"),
  negativePrompt: $("negative-prompt"),
  sampler: $("sampler"),
  seed: $("seed"),
  steps: $("steps"),
  stepsOut: $("steps-out"),
  cfg: $("cfg"),
  cfgOut: $("cfg-out"),
  strength: $("strength"),
  strengthOut: $("strength-out"),
  sourceImage: $("source-image"),
  sourceImageName: $("source-image-name"),
  width: $("width"),
  height: $("height"),
  batch: $("batch"),
  loras: $("loras"),
  advancedSection: $("advanced-section"),
  generate: $("generate"),
  generateSpinner: $("generate-spinner"),
  generateLabel: $("generate-label"),
  cancel: $("cancel"),
  status: $("status"),
  statusSpinner: $("status-spinner"),
  statusTextContent: $("status-text-content"),
  resultImage: $("result-image"),
  resultPlaceholder: $("result-placeholder"),
  navGallery: $("nav-gallery"),
  navModel: $("nav-model"),
  menuToggle: $("menu-toggle"),
  menu: $("menu"),
  viewGeneration: $("view-generation"),
  viewGallery: $("view-gallery"),
  viewModel: $("view-model"),
  gallery: $("gallery"),
  galleryEmpty: $("gallery-empty"),
  galleryRefresh: $("gallery-refresh"),
  modelSelectTrigger: $("model-select-trigger"),
  modelSelectCurrent: $("model-select-current"),
  modelSelectModal: $("model-select-modal"),
  modelSelectList: $("model-select-list"),
  modelSelectClose: $("model-select-close"),
  loraList: $("lora-list"),
  modelDir: $("model-dir"),
  loraDir: $("lora-dir"),
  saveModelDir: $("save-model-dir"),
  saveLoraDir: $("save-lora-dir"),
  lightbox: $("lightbox"),
  lightboxImg: $("lightbox-img"),
  lightboxParams: $("lightbox-params"),
  lbReuse: $("lb-reuse"),
  lbDelete: $("lb-delete"),
  lbClose: $("lb-close"),
};

let runningJobId = null;
let activeModelId = null;
let lightboxImage = null;
let initialImageBase64 = null;

async function getJson(path) {
  const res = await fetch(API + path);
  if (!res.ok) {
    throw new Error(`GET ${path} -> ${res.status}`);
  }
  return res.json();
}

function setStatus(text) {
  els.statusTextContent.textContent = text;
}

// 点字パターンのフレームを順に表示するスピナー (Braille spinner)。
// ステータスバーと生成ボタンなど、複数要素を同じタイマーで同期して回す。
const BRAILLE_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const BRAILLE_SPINNER_INTERVAL_MS = 80;
const spinnerElements = [];
let spinnerTimer = null;

function startStatusSpinner(...elements) {
  spinnerElements.push(...elements);
  for (const el of elements) {
    el.hidden = false;
    el.textContent = BRAILLE_SPINNER_FRAMES[0];
  }
  if (spinnerTimer !== null) {
    return;
  }
  let frame = 0;
  spinnerTimer = setInterval(() => {
    frame = (frame + 1) % BRAILLE_SPINNER_FRAMES.length;
    for (const el of spinnerElements) {
      el.textContent = BRAILLE_SPINNER_FRAMES[frame];
    }
  }, BRAILLE_SPINNER_INTERVAL_MS);
}

function stopStatusSpinner() {
  if (spinnerTimer !== null) {
    clearInterval(spinnerTimer);
    spinnerTimer = null;
  }
  for (const el of spinnerElements) {
    el.hidden = true;
  }
  spinnerElements.length = 0;
}

// --- ビュー切替 (WinUI3 の横スライド遷移を踏襲) ---

function showView(view) {
  for (const v of [els.viewGeneration, els.viewGallery, els.viewModel]) {
    v.classList.remove("view-active", "view-left", "view-right");
  }
  if (view === "generation") {
    els.viewGeneration.classList.add("view-active");
    els.viewGallery.classList.add("view-right");
    els.viewModel.classList.add("view-right");
  } else {
    els.viewGeneration.classList.add("view-left");
    const target = view === "gallery" ? els.viewGallery : els.viewModel;
    const other = view === "gallery" ? els.viewModel : els.viewGallery;
    target.classList.add("view-active");
    other.classList.add("view-right");
  }
}

// ステータスバーのリンクとスマホのハンバーガーメニューの両方から使う共通遷移。
// URL ハッシュに現在のビューを反映し、ブラウザの更新/戻る/進むでも同じビューに戻れるようにする。
async function navigateTo(view, { replace = false } = {}) {
  closeMenu();
  showView(view);
  setViewHash(view, replace);
  if (view === "gallery") {
    await loadGallery();
  } else if (view === "model") {
    await Promise.all([loadDirectories(), loadModels(), loadLoras()]);
  }
}

function setViewHash(view, replace) {
  const hash = view === "generation" ? "" : `#${view}`;
  if ((location.hash || "") === hash) {
    return;
  }
  const url = location.pathname + location.search + hash;
  if (replace) {
    history.replaceState(null, "", url);
  } else {
    history.pushState(null, "", url);
  }
}

function viewFromHash() {
  const hash = location.hash.replace(/^#/, "");
  return hash === "gallery" || hash === "model" ? hash : "generation";
}

function toggleMenu() {
  const open = els.menu.hidden;
  els.menu.hidden = !open;
  els.menuToggle.setAttribute("aria-expanded", String(open));
}

function closeMenu() {
  els.menu.hidden = true;
  els.menuToggle.setAttribute("aria-expanded", "false");
}

// --- 起動時ロード ---

async function loadHealth() {
  try {
    const health = await getJson("/health");
    activeModelId = health.loaded_model;
    els.navModel.textContent = `${health.device} · ${modelDisplayName(health.loaded_model)}`;
  } catch {
    els.navModel.textContent = "接続できません";
  }
}

function modelDisplayName(modelId) {
  if (!modelId) {
    return "?";
  }
  return modelId.split(/[\\/]/).pop();
}

// ベースモデル選択 UI 専用: ファイル名から拡張子を除いた表示名。
// (HF repo id には拡張子がないため影響なし)
function modelFileDisplayName(modelId) {
  const base = modelDisplayName(modelId);
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(0, dot) : base;
}

async function loadSamplers() {
  const samplers = await getJson("/samplers");
  els.sampler.innerHTML = "";
  for (const name of samplers) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    els.sampler.appendChild(option);
  }
}

async function loadLoras() {
  let loras;
  try {
    loras = await getJson("/loras");
  } catch {
    return;
  }

  // 生成ビューの選択用 (チェックボックス + weight)
  els.loras.innerHTML = "";
  if (loras.length === 0) {
    const empty = document.createElement("p");
    empty.className = "placeholder";
    empty.textContent = "利用可能な LoRA はありません (モデル画面でフォルダを指定)";
    els.loras.appendChild(empty);
  } else {
    for (const lora of loras) {
      els.loras.appendChild(buildLoraItem(lora.lora_id));
    }
  }

  // モデルビューの一覧表示 (検出された LoRA の確認用)
  if (els.loraList) {
    els.loraList.innerHTML = "";
    if (loras.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "LoRA が見つかりません";
      els.loraList.appendChild(empty);
    } else {
      for (const lora of loras) {
        els.loraList.appendChild(buildFileRow(lora.lora_id, lora.size_on_disk_bytes));
      }
    }
  }
}

function buildFileRow(id, sizeBytes) {
  const item = document.createElement("div");
  item.className = "model-item";
  item.style.cursor = "default";

  const spacer = document.createElement("span");
  spacer.className = "check";

  const name = document.createElement("span");
  name.className = "model-name";
  name.textContent = id.split(/[\\/]/).pop();
  name.title = id;

  const size = document.createElement("span");
  size.className = "model-size";
  size.textContent = sizeBytes > 0 ? formatBytes(sizeBytes) : "";

  item.append(spacer, name, size);
  return item;
}

function buildLoraItem(loraId) {
  const item = document.createElement("div");
  item.className = "lora-item";
  item.dataset.modelId = loraId;

  const head = document.createElement("label");
  head.className = "lora-head";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  const name = document.createElement("span");
  name.className = "lora-name";
  name.textContent = loraId.split(/[\\/]/).pop();
  name.title = loraId;
  head.append(checkbox, name);

  const weightRow = document.createElement("div");
  weightRow.className = "lora-weight";
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "-2";
  slider.max = "2";
  slider.step = "0.05";
  slider.value = "1";
  const weightOut = document.createElement("span");
  weightOut.textContent = "1.00";
  slider.addEventListener("input", () => {
    weightOut.textContent = Number(slider.value).toFixed(2);
  });
  weightRow.append(slider, weightOut);

  checkbox.addEventListener("change", () => {
    item.classList.toggle("enabled", checkbox.checked);
  });

  item.append(head, weightRow);
  item._checkbox = checkbox;
  item._slider = slider;
  return item;
}

function collectLoras() {
  const specs = [];
  for (const item of els.loras.querySelectorAll(".lora-item")) {
    if (item._checkbox.checked) {
      specs.push({ model_id: item.dataset.modelId, weight: Number(item._slider.value) });
    }
  }
  return specs;
}

function buildRequest() {
  const seedText = els.seed.value.trim();
  const seed = seedText === "" ? null : Number.parseInt(seedText, 10);
  const body = {
    prompt: els.prompt.value,
    negative_prompt: els.negativePrompt.value,
    steps: Number(els.steps.value),
    cfg_scale: Number(els.cfg.value),
    width: Number(els.width.value),
    height: Number(els.height.value),
    batch_size: Number(els.batch.value),
    sampler: els.sampler.value,
    seed: Number.isNaN(seed) ? null : seed,
    loras: collectLoras(),
  };
  if (initialImageBase64 !== null) {
    body.initial_image = initialImageBase64;
    body.strength = Number(els.strength.value);
  }
  return body;
}

// --- プロンプト等の永続化 (ブラウザ更新をまたいで保持) ---

const PROMPT_STORAGE_KEY = "sodalite.promptState";

function savePromptState() {
  const state = {
    prompt: els.prompt.value,
    negative_prompt: els.negativePrompt.value,
    sampler: els.sampler.value,
    seed: els.seed.value,
    steps: els.steps.value,
    cfg: els.cfg.value,
    width: els.width.value,
    height: els.height.value,
    batch: els.batch.value,
  };
  try {
    localStorage.setItem(PROMPT_STORAGE_KEY, JSON.stringify(state));
  } catch {
    // ストレージが使えない環境 (プライベートモード等) では保存をあきらめる。
  }
}

function loadPromptState() {
  try {
    const raw = localStorage.getItem(PROMPT_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function restorePromptState() {
  const state = loadPromptState();
  if (!state) {
    return;
  }
  els.prompt.value = state.prompt ?? "";
  els.negativePrompt.value = state.negative_prompt ?? "";
  if (state.sampler) {
    els.sampler.value = state.sampler;
  }
  els.seed.value = state.seed ?? "";
  if (state.steps != null) {
    els.steps.value = state.steps;
    els.stepsOut.textContent = state.steps;
  }
  if (state.cfg != null) {
    els.cfg.value = state.cfg;
    els.cfgOut.textContent = Number(state.cfg).toFixed(1);
  }
  if (state.width != null) {
    els.width.value = state.width;
  }
  if (state.height != null) {
    els.height.value = state.height;
  }
  if (state.batch != null) {
    els.batch.value = state.batch;
  }
}

function showResultImage(url) {
  els.resultImage.src = url;
  els.resultImage.hidden = false;
  els.resultPlaceholder.hidden = true;
}

// --- 生成 ---

async function onGenerate() {
  const body = buildRequest();
  if (body.prompt.trim() === "") {
    setStatus("プロンプトを入力してください");
    return;
  }

  setGenerating(true);
  els.generateLabel.textContent = "生成中…";
  startStatusSpinner(els.statusSpinner, els.generateSpinner);
  setStatus("生成を開始しています…");

  try {
    const endpoint = body.initial_image ? "/generations/image-to-image" : "/generations/text-to-image";
    const res = await fetch(API + endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      throw new Error(`生成開始に失敗しました (${res.status})`);
    }
    const job = await res.json();
    runningJobId = job.job_id;
    await pollUntilDone(job.job_id);
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error));
  } finally {
    stopStatusSpinner();
    els.generateLabel.textContent = "生成";
    setGenerating(false);
    runningJobId = null;
  }
}

// Mirrors the WinUI3 client: display each image as soon as images_completed
// increases, resetting the per-image timer, and stop on a terminal status.
async function pollUntilDone(jobId) {
  let lastCompleted = 0;
  const batchStart = performance.now();
  let imageStart = performance.now();

  while (true) {
    const job = await getJson(`/generations/${encodeURIComponent(jobId)}`);

    if (job.images_completed > lastCompleted && job.image_url) {
      lastCompleted = job.images_completed;
      showResultImage(job.image_url);
      imageStart = performance.now();
    }

    if (job.status === "completed") {
      const seconds = (performance.now() - batchStart) / 1000;
      setStatus(`完了 (${seconds.toFixed(1)} 秒)`);
      return;
    }
    if (job.status === "cancelled") {
      setStatus("中止しました");
      return;
    }
    if (job.status === "failed") {
      setStatus(`エラー: ${job.error ?? "不明"}`);
      return;
    }

    updateProgressStatus(job, imageStart, batchStart);
    await sleep(POLL_INTERVAL_MS);
  }
}

function stepPercentText(job) {
  if (!job.total_steps) {
    return "";
  }
  const percent = Math.round((Math.min(job.current_step, job.total_steps) / job.total_steps) * 100);
  return ` (${percent}%)`;
}

function updateProgressStatus(job, imageStart, batchStart) {
  const imageSeconds = (performance.now() - imageStart) / 1000;
  const percentText = stepPercentText(job);
  if (job.total_images <= 1) {
    setStatus(`生成中…${percentText} ${imageSeconds.toFixed(1)} 秒`);
    return;
  }
  const current = Math.min(job.images_completed + 1, job.total_images);
  const totalSeconds = (performance.now() - batchStart) / 1000;
  setStatus(
    `生成中…${percentText} ${current}/${job.total_images} 枚目 ` +
      `(この画像 ${imageSeconds.toFixed(1)} 秒 / 累計 ${totalSeconds.toFixed(1)} 秒)`,
  );
}

async function onCancel() {
  if (runningJobId === null) {
    return;
  }
  els.cancel.disabled = true;
  setStatus("中止しています…");
  try {
    await fetch(`${API}/generations/${encodeURIComponent(runningJobId)}`, { method: "DELETE" });
  } catch {
    // The next poll reflects the actual job status regardless.
  }
}

function setGenerating(active) {
  els.generate.disabled = active;
  els.cancel.hidden = !active;
  els.cancel.disabled = false;
  els.sourceImage.disabled = active;
  els.strength.disabled = active || initialImageBase64 === null;
}

async function selectSourceImage() {
  const file = els.sourceImage.files?.[0];
  if (!file) {
    initialImageBase64 = null;
    els.sourceImageName.textContent = "画像は選択されていません";
    els.strength.disabled = true;
    els.steps.value = "20";
    els.stepsOut.textContent = "20";
    return;
  }
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(reader.result));
    reader.addEventListener("error", reject);
    reader.readAsDataURL(file);
  });
  initialImageBase64 = String(dataUrl).split(",", 2)[1] ?? null;
  els.sourceImageName.textContent = file.name;
  els.strength.disabled = initialImageBase64 === null;
  els.steps.value = "40";
  els.stepsOut.textContent = "40";
}

// --- モデル選択 ---

async function loadModels() {
  let models;
  try {
    models = await getJson("/models");
  } catch {
    return;
  }
  els.modelSelectList.innerHTML = "";
  for (const model of models) {
    els.modelSelectList.appendChild(buildModelItem(model));
  }
  const active = models.find((model) => model.is_active);
  els.modelSelectCurrent.textContent = active ? modelFileDisplayName(active.model_id) : "未選択";
}

function buildModelItem(model) {
  const item = document.createElement("div");
  item.className = "model-item";

  const check = document.createElement("span");
  check.className = "check";
  check.textContent = model.is_active ? "✓" : "";

  const name = document.createElement("span");
  name.className = "model-name";
  name.textContent = modelFileDisplayName(model.model_id);
  name.title = model.model_id;

  const size = document.createElement("span");
  size.className = "model-size";
  size.textContent = model.size_on_disk_bytes > 0 ? formatBytes(model.size_on_disk_bytes) : "";

  item.append(check, name, size);
  item.addEventListener("click", () => switchModel(model.model_id));
  return item;
}

function formatBytes(bytes) {
  const gib = bytes / (1024 * 1024 * 1024);
  return gib >= 1 ? `${gib.toFixed(1)} GiB` : `${(bytes / (1024 * 1024)).toFixed(0)} MiB`;
}

async function switchModel(modelId) {
  closeModelSelectModal();
  if (modelId === activeModelId) {
    return;
  }
  setStatus(`モデルを切り替えています: ${modelDisplayName(modelId)}`);
  try {
    const res = await fetch(API + "/models/active", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail ?? `切り替えに失敗しました (${res.status})`);
    }
    activeModelId = modelId;
    setStatus(`モデルを切り替えました: ${modelDisplayName(modelId)}`);
    await Promise.all([loadHealth(), loadModels()]);
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error));
  }
}

function openModelSelectModal() {
  els.modelSelectModal.hidden = false;
}

function closeModelSelectModal() {
  els.modelSelectModal.hidden = true;
}

// --- スキャンディレクトリ設定 (WinUI のフォルダ選択に相当) ---
// ブラウザからはサーバー上のパスを直接選べないため、パス文字列を入力して保存する。

async function loadDirectories() {
  let dirs;
  try {
    dirs = await getJson("/settings/directories");
  } catch {
    return;
  }
  els.modelDir.value = dirs.model_dir ?? "";
  els.loraDir.value = dirs.lora_dir ?? "";
}

async function saveDirectories() {
  const body = {
    model_dir: els.modelDir.value.trim() || null,
    lora_dir: els.loraDir.value.trim() || null,
  };
  const res = await fetch(API + "/settings/directories", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `保存に失敗しました (${res.status})`);
  }
  const saved = await res.json();
  els.modelDir.value = saved.model_dir ?? "";
  els.loraDir.value = saved.lora_dir ?? "";
}

async function onSaveDirectories() {
  setStatus("フォルダ設定を保存しています…");
  try {
    await saveDirectories();
    setStatus("フォルダ設定を保存しました");
    await Promise.all([loadModels(), loadLoras()]);
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error));
  }
}

// --- ギャラリー ---

const GALLERY_SKELETON_COUNT = 12;

function showGallerySkeleton() {
  els.galleryEmpty.hidden = true;
  els.gallery.innerHTML = "";
  for (let i = 0; i < GALLERY_SKELETON_COUNT; i++) {
    const item = document.createElement("div");
    item.className = "gallery-item skeleton";
    els.gallery.appendChild(item);
  }
}

let hasLoadedGalleryOnce = false;

async function loadGallery() {
  if (hasLoadedGalleryOnce) {
    await refreshGalleryDiff();
    return;
  }

  showGallerySkeleton();
  let images;
  try {
    images = await getJson("/gallery/images");
  } catch {
    els.gallery.innerHTML = "";
    return;
  }
  els.gallery.innerHTML = "";
  els.galleryEmpty.hidden = images.length > 0;
  for (const image of images) {
    els.gallery.appendChild(buildGalleryItem(image));
  }
  hasLoadedGalleryOnce = true;
}

// 既に一覧を表示済みの状態で再度ギャラリーを開いたときに使う。全件を取得し直すが、
// 画像要素は追加・削除された分だけ組み替えて、既存のサムネイル表示を維持する。
async function refreshGalleryDiff() {
  let images;
  try {
    images = await getJson("/gallery/images");
  } catch {
    return;
  }

  const latestIds = new Set(images.map((image) => String(image.image_id ?? "")));
  for (const el of Array.from(els.gallery.querySelectorAll(".gallery-item"))) {
    if (!latestIds.has(el.dataset.imageId)) {
      el.remove();
    }
  }

  // サーバーは新しい順で返すため、その並び順のまま先頭から確定させていけば表示順も一致する。
  // 既存要素はそのまま (サムネイルの再読み込みを避ける)、新規分だけ組み立てて挿入する。
  let anchor = els.gallery.firstChild;
  for (const image of images) {
    const imageId = String(image.image_id ?? "");
    const existing = anchor?.dataset.imageId === imageId ? anchor : els.gallery.querySelector(
      `.gallery-item[data-image-id="${CSS.escape(imageId)}"]`,
    );

    if (existing) {
      if (existing !== anchor) {
        els.gallery.insertBefore(existing, anchor);
      }
      anchor = existing.nextSibling;
    } else {
      const item = buildGalleryItem(image);
      els.gallery.insertBefore(item, anchor);
    }
  }

  updateGalleryEmptyState();
}

async function onGalleryRefresh() {
  els.galleryRefresh.disabled = true;
  els.galleryRefresh.classList.add("spinning");
  try {
    await loadGallery();
  } finally {
    els.galleryRefresh.classList.remove("spinning");
    els.galleryRefresh.disabled = false;
  }
}

function buildGalleryItem(image) {
  const item = document.createElement("div");
  item.className = "gallery-item";
  item.dataset.imageId = image.image_id ?? "";

  const img = document.createElement("img");
  img.src = image.image_url;
  img.alt = "";
  img.loading = "lazy";

  item.appendChild(img);
  item.addEventListener("click", () => openLightbox(image));
  return item;
}

// --- ライトボックス (拡大 + パラメータ再利用 + 削除) ---

function openLightbox(image) {
  lightboxImage = image;
  els.lightboxImg.src = image.image_url;
  els.lightboxParams.textContent = formatParams(image.parameters);
  els.lbReuse.hidden = image.parameters == null;
  els.lbDelete.hidden = image.image_id == null;
  els.lightbox.hidden = false;
}

function closeLightbox() {
  els.lightbox.hidden = true;
  els.lightboxImg.src = "";
  lightboxImage = null;
}

function formatParams(params) {
  if (!params) {
    return "パラメータ情報なし";
  }
  const lines = [
    `プロンプト: ${params.prompt || "(なし)"}`,
    `ネガティブ: ${params.negative_prompt || "(なし)"}`,
    `ステップ: ${params.steps ?? "?"}`,
    `CFG: ${params.cfg_scale ?? "?"}`,
    `サイズ: ${params.width ?? "?"}×${params.height ?? "?"}`,
    `サンプラー: ${params.sampler ?? "?"}`,
    `シード: ${params.seed ?? "ランダム"}`,
  ];
  if (params.loras && params.loras.length > 0) {
    lines.push(`LoRA: ${params.loras.map((l) => `${modelDisplayName(l.model_id)}(${l.weight})`).join(", ")}`);
  }
  return lines.join("\n");
}

function reuseParams() {
  const params = lightboxImage?.parameters;
  if (!params) {
    return;
  }
  els.prompt.value = params.prompt ?? "";
  els.negativePrompt.value = params.negative_prompt ?? "";
  if (params.steps != null) {
    els.steps.value = params.steps;
    els.stepsOut.textContent = params.steps;
  }
  if (params.cfg_scale != null) {
    els.cfg.value = params.cfg_scale;
    els.cfgOut.textContent = Number(params.cfg_scale).toFixed(1);
  }
  if (params.width != null) {
    els.width.value = params.width;
  }
  if (params.height != null) {
    els.height.value = params.height;
  }
  if (params.batch_size != null) {
    els.batch.value = params.batch_size;
  }
  if (params.sampler) {
    els.sampler.value = params.sampler;
  }
  els.seed.value = params.seed != null ? String(params.seed) : "";
  els.advancedSection.open = true;
  savePromptState();

  closeLightbox();
  showView("generation");
  setViewHash("generation", false);
  setStatus("パラメータを再利用しました");
}

async function deleteLightboxImage() {
  const imageId = lightboxImage?.image_id;
  if (imageId == null) {
    return;
  }
  closeLightbox();

  const item = els.gallery.querySelector(`.gallery-item[data-image-id="${CSS.escape(String(imageId))}"]`);
  const nextSibling = item?.nextSibling ?? null;
  item?.remove();
  updateGalleryEmptyState();

  try {
    const res = await fetch(`${API}/gallery/images/${encodeURIComponent(imageId)}`, { method: "DELETE" });
    if (!res.ok) {
      throw new Error(`DELETE -> ${res.status}`);
    }
  } catch {
    // 削除に失敗した場合は元の位置に戻す。
    if (item) {
      els.gallery.insertBefore(item, nextSibling);
      updateGalleryEmptyState();
    }
  }
}

function updateGalleryEmptyState() {
  els.galleryEmpty.hidden = els.gallery.querySelector(".gallery-item") != null;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// --- 初期化 ---

function wireEvents() {
  els.steps.addEventListener("input", () => {
    els.stepsOut.textContent = els.steps.value;
  });
  els.cfg.addEventListener("input", () => {
    els.cfgOut.textContent = Number(els.cfg.value).toFixed(1);
  });
  els.strength.addEventListener("input", () => {
    els.strengthOut.textContent = Number(els.strength.value).toFixed(2);
  });
  els.sourceImage.addEventListener("change", () => {
    selectSourceImage().catch(() => {
      initialImageBase64 = null;
      els.sourceImageName.textContent = "画像を読み込めませんでした";
    });
  });
  els.generate.addEventListener("click", onGenerate);
  els.cancel.addEventListener("click", onCancel);

  // プロンプト等はブラウザ更新後も残るよう、変更のたびに保存する。
  for (const el of [
    els.prompt,
    els.negativePrompt,
    els.sampler,
    els.seed,
    els.steps,
    els.cfg,
    els.width,
    els.height,
    els.batch,
  ]) {
    el.addEventListener("input", savePromptState);
    el.addEventListener("change", savePromptState);
  }

  els.navGallery.addEventListener("click", () => navigateTo("gallery"));
  els.galleryRefresh.addEventListener("click", onGalleryRefresh);
  els.navModel.addEventListener("click", () => navigateTo("model"));
  els.saveModelDir.addEventListener("click", onSaveDirectories);
  els.saveLoraDir.addEventListener("click", onSaveDirectories);
  for (const back of document.querySelectorAll("[data-back]")) {
    back.addEventListener("click", () => navigateTo("generation"));
  }

  // ハンバーガーメニュー (スマホ)
  els.menuToggle.addEventListener("click", (event) => {
    event.stopPropagation();
    toggleMenu();
  });
  for (const item of els.menu.querySelectorAll(".menu-item")) {
    item.addEventListener("click", () => navigateTo(item.dataset.nav));
  }
  // メニュー外をクリックしたら閉じる
  document.addEventListener("click", (event) => {
    if (!els.menu.hidden && !els.menu.contains(event.target) && event.target !== els.menuToggle) {
      closeMenu();
    }
  });

  els.resultImage.addEventListener("click", () => {
    if (els.resultImage.src) {
      openLightbox({ image_url: els.resultImage.src, parameters: null, image_id: null });
    }
  });

  els.lbReuse.addEventListener("click", reuseParams);
  els.lbDelete.addEventListener("click", deleteLightboxImage);
  els.lbClose.addEventListener("click", closeLightbox);
  els.lightbox.addEventListener("click", (event) => {
    if (event.target === els.lightbox) {
      closeLightbox();
    }
  });

  els.modelSelectTrigger.addEventListener("click", openModelSelectModal);
  els.modelSelectClose.addEventListener("click", closeModelSelectModal);
  els.modelSelectModal.addEventListener("click", (event) => {
    if (event.target === els.modelSelectModal) {
      closeModelSelectModal();
    }
  });
}

async function init() {
  wireEvents();
  window.addEventListener("popstate", () => navigateTo(viewFromHash(), { replace: true }));
  await loadHealth();
  await Promise.allSettled([loadSamplers(), loadLoras()]);
  restorePromptState();
  els.generate.disabled = false;
  await navigateTo(viewFromHash(), { replace: true });
}

init();
