# SDApp のビルドオーケストレーション。
# run.ps1 から呼び出され、変更のないターゲットはスキップする。
# GnuWin32 make (3.81) を Windows 上で使うため、シェルは cmd に固定する。

SHELL := cmd
.SHELLFLAGS := /c

BACKEND_DIR := backend
FRONTEND_DIR := frontend/SDApp

BACKEND_SYNC_STAMP := $(BACKEND_DIR)/.venv/.sync-stamp
BACKEND_PYPROJECT := $(BACKEND_DIR)/pyproject.toml
BACKEND_LOCK := $(BACKEND_DIR)/uv.lock

FRONTEND_EXE := $(FRONTEND_DIR)/bin/Debug/net9.0-windows10.0.26100.0/win-x64/SDApp.exe
FRONTEND_CSPROJ := $(FRONTEND_DIR)/SDApp.csproj
FRONTEND_SOURCES := $(wildcard $(FRONTEND_DIR)/*.cs) \
	$(wildcard $(FRONTEND_DIR)/*.xaml) \
	$(wildcard $(FRONTEND_DIR)/Views/*.cs) \
	$(wildcard $(FRONTEND_DIR)/Views/*.xaml) \
	$(wildcard $(FRONTEND_DIR)/ViewModels/*.cs) \
	$(wildcard $(FRONTEND_DIR)/Services/*.cs) \
	$(wildcard $(FRONTEND_DIR)/Models/*.cs)

.PHONY: all run backend frontend clean

all: backend frontend

# バックエンドの依存関係を同期する。pyproject.toml/uv.lock が
# 前回の同期より新しい場合のみ uv sync を実行する。
backend: $(BACKEND_SYNC_STAMP)

$(BACKEND_SYNC_STAMP): $(BACKEND_PYPROJECT) $(BACKEND_LOCK)
	@echo ==^> uv sync (backend)
	cd $(BACKEND_DIR) && uv sync
	@echo. > $(subst /,\,$(BACKEND_SYNC_STAMP))

# フロントエンドをビルドする。ソース/csproj が現在の exe より
# 新しい場合のみ dotnet build を実行する。
frontend: $(FRONTEND_EXE)

$(FRONTEND_EXE): $(FRONTEND_CSPROJ) $(FRONTEND_SOURCES)
	@echo ==^> dotnet build (frontend)
	cd $(FRONTEND_DIR) && dotnet build -c Debug

run: all
	@echo ==^> Launching SDApp
	$(subst /,\,$(FRONTEND_EXE))

clean:
	if exist $(subst /,\,$(BACKEND_SYNC_STAMP)) del /q $(subst /,\,$(BACKEND_SYNC_STAMP))
	if exist $(subst /,\,$(FRONTEND_DIR))\bin rmdir /s /q $(subst /,\,$(FRONTEND_DIR))\bin
	if exist $(subst /,\,$(FRONTEND_DIR))\obj rmdir /s /q $(subst /,\,$(FRONTEND_DIR))\obj
