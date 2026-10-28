# json_sanitizer.py
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional, Tuple

_UTF8_BOM = b"\xef\xbb\xbf"
_BOM_U = "\ufeff"

class JSONSanitizer:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._log = []

    # --------------------------
    # Logging
    # --------------------------
    def _log_msg(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        self._log.append(msg)

    def get_log(self) -> str:
        return "\n".join(self._log)

    # --------------------------
    # Auditorias / Decodificação
    # --------------------------
    def audit_utf8_decode(self, data: bytes, preview_radius: int = 16) -> Optional[str]:
        """Retorna string com diagnóstico se falhar; None se OK."""
        try:
            data.decode("utf-8")
            # também marca presença de BOM
            if data.startswith(_UTF8_BOM):
                return "[audit] UTF-8 OK, porém com BOM no início (será removido)."
            return None
        except UnicodeDecodeError as e:
            start, end = e.start, e.end
            bstart = max(0, start - preview_radius)
            bend = min(len(data), end + preview_radius)
            window = data[bstart:bend]
            hex_bytes = " ".join(f"{b:02X}" for b in window)
            caret = "   " * (start - bstart) + "^^"
            return (
                "[audit] Falha ao decodificar como UTF-8 (strict):\n"
                f"  mensagem: {e}\n"
                f"  byte offset: {start}\n"
                "  bytes ao redor (hex):\n"
                f"    {hex_bytes}\n"
                f"    {caret}"
            )

    def sniff_and_decode(self, data: bytes) -> Tuple[str, str]:
        """
        Tenta decodificar priorizando UTF-8. Se houver BOM, usa utf-8-sig para remover.
        Se UTF-8 falhar, tenta latin-1 (último recurso) e registra no log.
        Retorna (texto, encoding_utilizado).
        """
        # 1) BOM?
        if data.startswith(_UTF8_BOM):
            text = data.decode("utf-8-sig")  # strip BOM automaticamente
            self._log_msg("[decode] Detectado BOM UTF-8 → usando utf-8-sig (BOM removido)")
            return text, "utf-8-sig"

        # 2) UTF-8 estrito
        try:
            text = data.decode("utf-8")
            self._log_msg("[decode] UTF-8 (strict) OK")
            return text, "utf-8"
        except UnicodeDecodeError as e:
            self._log_msg(f"[decode] UTF-8 falhou: {e!s}")

        # 3) Tenta latin-1 (não perde bytes, mas pode precisar de limpeza posterior)
        text = data.decode("latin-1")
        self._log_msg("[decode] Fallback latin-1 (atenção: pode conter caracteres indevidos)")
        return text, "latin-1"

    # --------------------------
    # Limpezas textuais
    # --------------------------
    def _strip_illegal_controls(self, s: str) -> str:
        # Remove controles C0 exceto tab(0x09), LF(0x0A), CR(0x0D)
        return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)

    def _fix_trailing_commas(self, s: str) -> str:
        # Remove vírgulas antes de ] ou } em JSON
        s = re.sub(r",(\s*])", r"\1", s)
        s = re.sub(r",(\s*})", r"\1", s)
        return s

    def clean_text(self, text: str) -> str:
        """
        Aplica correções leves e valida com json.loads.
        Remove BOM se sobrar, controles ilegais e vírgulas finais.
        """
        s = text

        # 1) Remove BOM se estiver no texto
        if s.startswith(_BOM_U):
            s = s.lstrip(_BOM_U)
            self._log_msg("[fix] BOM removido do início do texto (Unicode)")

        # 2) Normaliza quebras de linha
        s = s.replace("\r\n", "\n").replace("\r", "\n")

        # 3) Remove controles ilegais
        s2 = self._strip_illegal_controls(s)
        if s2 != s:
            self._log_msg("[fix] Caracteres de controle C0 removidos")
        s = s2

        # 4) Remove vírgulas finais
        s3 = self._fix_trailing_commas(s)
        if s3 != s:
            self._log_msg("[fix] Vírgulas finais removidas")
        s = s3

        # 5) Valida JSON
        try:
            json.loads(s)
            self._log_msg("[validate] JSON válido após correções")
            return s
        except json.JSONDecodeError as e:
            # Se acusar BOM, reforça a remoção
            if "Unexpected UTF-8 BOM" in str(e) or (s and s[0] == _BOM_U):
                s = s.lstrip(_BOM_U)
                self._log_msg("[fix] Remoção extra de BOM e nova validação")
                json.loads(s)  # se falhar, cai no except externo
                self._log_msg("[validate] JSON válido após remoção extra de BOM")
                return s
            # Se falhar por outros motivos, relança com contexto
            raise

    # --------------------------
    # API principal
    # --------------------------
    def check_and_fix_file(
        self,
        path_in: Path,
        out_dir: Optional[Path] = None,
        write_fixed: bool = True,
        out_suffix: str = "_clean.json",
    ) -> Path:
        """
        Audita/decodifica/limpa e grava uma cópia corrigida.
        Retorna o caminho do arquivo de saída (ou o original se write_fixed=False).
        """
        path_in = Path(path_in)
        raw = path_in.read_bytes()

        # Auditoria UTF-8
        audit = self.audit_utf8_decode(raw)
        if audit:
            self._log_msg(audit)

        # Decodifica
        text, enc = self.sniff_and_decode(raw)
        self._log_msg(f"[decode] encoding usado para parsing: {enc}")

        # Limpa + valida JSON
        clean = self.clean_text(text)

        if not write_fixed:
            return path_in

        # Caminho de saída
        out_dir = out_dir or path_in.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (path_in.stem + out_suffix)
        out_path.write_text(clean, encoding="utf-8")
        self._log_msg(f"[write] Arquivo limpo salvo em: {out_path}")
        return out_path

    def robust_load_json(self, path: Path):
        """
        Carrega JSON com tolerância: tenta direto; se falhar, passa pelo pipeline de limpeza
        (em memória) e retorna o objeto Python.
        """
        path = Path(path)
        raw = path.read_bytes()
        try:
            # Tenta decodificar/parsear direto em utf-8-sig (remove BOM se houver)
            text = raw.decode("utf-8-sig")
            return json.loads(text)
        except Exception as e:
            self._log_msg(f"[robust] Falhou leitura direta: {e!s}")
            text, _ = self.sniff_and_decode(raw)
            clean = self.clean_text(text)
            return json.loads(clean)
