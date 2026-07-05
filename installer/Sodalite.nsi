; Sodalite NSIS インストーラースクリプト (MUI2)
;
; build-installer.ps1 から makensis で呼び出される。事前に installer\staging\ 配下へ
; 配布物(app\ と backend\)がステージングされている前提。
;
; ライセンス: NSIS は zlib/libpng ライク (permissive) で本プロジェクトの方針に適合する。

Unicode true

!include "MUI2.nsh"
!include "x64.nsh"

;--------------------------------
; 製品メタ情報
;--------------------------------
!ifndef PRODUCT_VERSION
  !define PRODUCT_VERSION "0.2.0"
!endif

!define PRODUCT_NAME "Sodalite"
!define PRODUCT_PUBLISHER "ひかり"
!define PRODUCT_EXE "Sodalite.exe"
!define UNINSTALL_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"

; installer\ ディレクトリからの相対パス。build-installer.ps1 が用意する。
!define STAGING_DIR "staging"
!define DIST_DIR "dist"

Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "${DIST_DIR}\${PRODUCT_NAME}-Setup-${PRODUCT_VERSION}.exe"
InstallDir "$LOCALAPPDATA\Programs\${PRODUCT_NAME}"
InstallDirRegKey HKCU "${UNINSTALL_KEY}" "InstallLocation"
RequestExecutionLevel user
SetCompressor /SOLID lzma

;--------------------------------
; MUI2 の外観・ページ構成
;--------------------------------
!define MUI_ICON "${STAGING_DIR}\app\Assets\AppIcon.ico"
!define MUI_UNICON "${STAGING_DIR}\app\Assets\AppIcon.ico"
!define MUI_ABORTWARNING

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "..\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES

; インストール完了後にアプリを起動するオプション
!define MUI_FINISHPAGE_RUN "$INSTDIR\app\${PRODUCT_EXE}"
!define MUI_FINISHPAGE_RUN_TEXT "${PRODUCT_NAME} を起動する"
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; 言語(日本語を優先し、英語をフォールバックとして併載)
!insertmacro MUI_LANGUAGE "Japanese"
!insertmacro MUI_LANGUAGE "English"

;--------------------------------
; インストール本体
;--------------------------------
Section "${PRODUCT_NAME}" SecMain
  SectionIn RO

  ; フロントエンド一式 (dotnet publish の出力) を app\ へ
  SetOutPath "$INSTDIR\app"
  File /r "${STAGING_DIR}\app\*.*"

  ; Python バックエンドソースを backend\ へ (.venv は同梱せず、初回起動時に uv sync で作成)
  SetOutPath "$INSTDIR\backend"
  File /r "${STAGING_DIR}\backend\*.*"

  ; ショートカット
  CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
  CreateShortcut "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk" \
    "$INSTDIR\app\${PRODUCT_EXE}" "" "$INSTDIR\app\Assets\AppIcon.ico"
  CreateShortcut "$DESKTOP\${PRODUCT_NAME}.lnk" \
    "$INSTDIR\app\${PRODUCT_EXE}" "" "$INSTDIR\app\Assets\AppIcon.ico"

  ; アンインストーラー生成
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  ; 「アプリと機能」に表示するための Uninstall レジストリ登録 (ユーザーインストールなので HKCU へ)
  WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayName" "${PRODUCT_NAME}"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "Publisher" "${PRODUCT_PUBLISHER}"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayIcon" "$INSTDIR\app\${PRODUCT_EXE}"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "UninstallString" "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "QuietUninstallString" "$INSTDIR\Uninstall.exe /S"
  WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoRepair" 1
SectionEnd

;--------------------------------
; アンインストール
;--------------------------------
Section "Uninstall"
  ; インストール先を丸ごと削除する。
  RMDir /r "$INSTDIR\app"
  RMDir /r "$INSTDIR\backend"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"

  Delete "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk"
  RMDir "$SMPROGRAMS\${PRODUCT_NAME}"
  Delete "$DESKTOP\${PRODUCT_NAME}.lnk"

  DeleteRegKey HKCU "${UNINSTALL_KEY}"

  ; アプリ情報をすべて消去する。%LOCALAPPDATA%\Sodalite には仮想環境 (.venv)・.venv-ready
  ; マーカー・設定・生成画像・HF キャッシュが入っており、アンインストールでこれらも消す。
  ; (数 GB になり得るが、残さずクリーンに削除する方針)
  RMDir /r "$LOCALAPPDATA\${PRODUCT_NAME}"
SectionEnd
