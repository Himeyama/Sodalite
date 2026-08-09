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
  width: $("width"),
  height: $("height"),
  batch: $("batch"),
  loras: $("loras"),
  generate: $("generate"),
  cancel: $("cancel"),
  status: $("status"),
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
  modelList: $("model-list"),
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

async function getJson(path) {
  const res = await fetch(API + path);
  if (!res.ok) {
    throw new Error(`GET ${path} -> ${res.status}`);
  }
  return res.json();
}

function setStatus(text) {
  els.status.textContent = text;
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
async function navigateTo(view) {
  closeMenu();
  showView(view);
  if (view === "gallery") {
    await loadGallery();
  } else if (view === "model") {
    await Promise.all([loadDirectories(), loadModels(), loadLoras()]);
  }
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
  return {
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
  setStatus("生成を開始しています…");

  try {
    const res = await fetch(API + "/generations/text-to-image", {
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

function updateProgressStatus(job, imageStart, batchStart) {
  const imageSeconds = (performance.now() - imageStart) / 1000;
  if (job.total_images <= 1) {
    setStatus(`生成中… ${imageSeconds.toFixed(1)} 秒`);
    return;
  }
  const current = Math.min(job.images_completed + 1, job.total_images);
  const totalSeconds = (performance.now() - batchStart) / 1000;
  setStatus(
    `生成中… ${current}/${job.total_images} 枚目 ` +
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
}

// --- モデル選択 ---

async function loadModels() {
  let models;
  try {
    models = await getJson("/models");
  } catch {
    return;
  }
  els.modelList.innerHTML = "";
  for (const model of models) {
    els.modelList.appendChild(buildModelItem(model));
  }
}

function buildModelItem(model) {
  const item = document.createElement("div");
  item.className = "model-item";

  const check = document.createElement("span");
  check.className = "check";
  check.textContent = model.is_active ? "✓" : "";

  const name = document.createElement("span");
  name.className = "model-name";
  name.textContent = model.model_id;
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

async function loadGallery() {
  let images;
  try {
    images = await getJson("/gallery/images");
  } catch {
    return;
  }
  els.gallery.innerHTML = "";
  els.galleryEmpty.hidden = images.length > 0;
  for (const image of images) {
    els.gallery.appendChild(buildGalleryItem(image));
  }
}

function buildGalleryItem(image) {
  const item = document.createElement("div");
  item.className = "gallery-item";

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

  closeLightbox();
  showView("generation");
  setStatus("パラメータを再利用しました");
}

async function deleteLightboxImage() {
  const imageId = lightboxImage?.image_id;
  if (imageId == null) {
    return;
  }
  try {
    await fetch(`${API}/gallery/images/${encodeURIComponent(imageId)}`, { method: "DELETE" });
  } catch {
    // Leave it if deletion fails; the reload below reconciles.
  }
  closeLightbox();
  await loadGallery();
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
  els.generate.addEventListener("click", onGenerate);
  els.cancel.addEventListener("click", onCancel);

  els.navGallery.addEventListener("click", () => navigateTo("gallery"));
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
}

async function init() {
  wireEvents();
  await loadHealth();
  await Promise.allSettled([loadSamplers(), loadLoras()]);
  els.generate.disabled = false;
}

init();
