"""Tools given to the AgentNet *executor* on GAIA.

Original AgentNet agents have no tools; GAIA questions need web lookup and file
reading, so the executor's answer step becomes a small tool loop (see gaia/executor.py).
Every tool call is timed in gaia.tracker.

  web_search(query)            Tavily (TAVILY_API_KEY) > Serper (SERPER_API_KEY) > DuckDuckGo HTML
  fetch_url(url, offset)       page / PDF text, paged
  read_attachment(offset)      text extraction of the task's attached file
  python(code)                 run Python in a temp dir that contains the attachment (timeout)
  describe_image(question)     vision call on an image attachment (LLM_SUPPORTS_VISION=1)
  transcribe_audio()           /audio/transcriptions on an audio attachment (AUDIO_TRANSCRIBE_MODEL set)
"""
import base64
import html
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser

from .tracker import TRACKER

PAGE_CHARS = int(os.environ.get("TOOL_PAGE_CHARS", 8000))
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
TEXT_EXT = {".txt", ".md", ".py", ".json", ".jsonld", ".xml", ".csv", ".tsv", ".html", ".htm", ".pdb", ".js", ".yaml", ".yml"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
AUDIO_EXT = {".mp3", ".m4a", ".wav", ".flac", ".ogg"}


# ------------------------------------------------------------------ helpers
class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"):
            self.skip += 1
        if tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "table", "section"):
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self.skip:
            self.skip -= 1
        if tag in ("td", "th"):
            self.out.append(" | ")

    def handle_data(self, d):
        if not self.skip:
            self.out.append(d)


def html_to_text(s):
    p = _Text()
    try:
        p.feed(s)
    except Exception:
        return re.sub(r"<[^>]+>", " ", s)
    t = "".join(p.out)
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n\n", t).strip()


def _page(text, offset):
    offset = max(0, int(offset or 0))
    chunk = text[offset:offset + PAGE_CHARS]
    more = offset + PAGE_CHARS < len(text)
    tail = f"\n\n[showing chars {offset}-{offset + len(chunk)} of {len(text)}" + \
           (f"; call again with offset={offset + PAGE_CHARS} for more]" if more else "; end]")
    return chunk + tail


def _http_get(url, timeout=30, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "")


def _is_pdf(data):
    """Detect PDFs by magic bytes; URL suffixes and Content-Type can be misleading."""
    return data[:5] == b"%PDF-"


def _pdf_text(data):
    if not _is_pdf(data):
        # A PDF URL may redirect to an HTML page; avoid passing that response to pypdf.
        text = html_to_text(data.decode("utf-8", "ignore"))
        return (text + "\n\n[note: this response was not a PDF file despite the URL]") if text \
            else "[note: the server returned no readable text for this PDF URL]"
    try:
        from pypdf import PdfReader
    except ImportError:
        return "[pypdf not installed: pip install pypdf]"
    try:
        r = PdfReader(io.BytesIO(data))
        return "\n\n".join(f"--- page {i + 1} ---\n{(p.extract_text() or '')}" for i, p in enumerate(r.pages))
    except Exception as e:
        return f"[failed to parse PDF: {type(e).__name__}: {e}]"


# ------------------------------------------------------------------ file extraction
def extract_file(path):
    ext = os.path.splitext(path)[1].lower()
    name = os.path.basename(path)
    if ext in TEXT_EXT:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    if ext == ".pdf":
        with open(path, "rb") as f:
            return _pdf_text(f.read())
    if ext in (".xlsx", ".xlsm", ".xls"):
        try:
            import openpyxl
        except ImportError:
            return "[openpyxl not installed]"
        wb = openpyxl.load_workbook(path, data_only=True)
        parts = []
        for ws in wb.worksheets:
            parts.append(f"=== sheet: {ws.title} (rows={ws.max_row}, cols={ws.max_column}) ===")
            for row in ws.iter_rows():
                cells = []
                for c in row:
                    v = c.value
                    color = None
                    try:
                        fg = c.fill.fgColor
                        if c.fill.fill_type and fg is not None and fg.type == "rgb" and fg.rgb not in ("00000000", "FFFFFFFF"):
                            color = fg.rgb
                    except Exception:
                        pass
                    if v is None and color is None:
                        continue
                    cells.append(f"{c.coordinate}={v!r}" + (f"[fill={color}]" if color else ""))
                if cells:
                    parts.append(", ".join(cells))
        return "\n".join(parts)
    if ext == ".docx":
        try:
            import docx
        except ImportError:
            return "[python-docx not installed]"
        d = docx.Document(path)
        out = [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                out.append(" | ".join(c.text for c in row.cells))
        return "\n".join(out)
    if ext == ".pptx":
        try:
            from pptx import Presentation
        except ImportError:
            return "[python-pptx not installed]"
        prs = Presentation(path)
        out = []
        for i, s in enumerate(prs.slides, 1):
            out.append(f"--- slide {i} ---")
            for sh in s.shapes:
                if getattr(sh, "has_text_frame", False) and sh.has_text_frame:
                    out.append(sh.text_frame.text)
        return "\n".join(out)
    if ext == ".zip":
        out = []
        tmp = tempfile.mkdtemp(prefix="gaia_zip_")
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
            for n in z.namelist():
                p = os.path.join(tmp, n)
                if os.path.isfile(p):
                    out.append(f"=== {n} ===\n{extract_file(p)}")
        return "\n\n".join(out)
    if ext in IMG_EXT:
        return f"[{name} is an image. Use the describe_image tool to ask questions about it.]"
    if ext in AUDIO_EXT:
        return f"[{name} is audio. Use the transcribe_audio tool to get its transcript.]"
    return f"[{name}: unsupported file type {ext}; you may try the python tool on it.]"


# ------------------------------------------------------------------ tool impls
class ToolBox:
    """One instance per GAIA task (knows that task's attachment)."""

    def __init__(self, attachment_path=None, enable_python=True):
        self.attachment = attachment_path if attachment_path and os.path.isfile(attachment_path) else None
        self.enable_python = enable_python
        self._attachment_text = None
        self.workdir = tempfile.mkdtemp(prefix="gaia_task_")
        if self.attachment:
            shutil.copy(self.attachment, self.workdir)

    # schemas in OpenAI tool format
    def schemas(self):
        tools = [
            _fn("web_search", "Search the web. Returns titles, URLs and snippets.",
                {"query": {"type": "string"}}, ["query"]),
            _fn("fetch_url", "Fetch a web page or PDF and return its text (paged).",
                {"url": {"type": "string"}, "offset": {"type": "integer", "description": "char offset for paging"}}, ["url"]),
        ]
        if self.attachment:
            tools.append(_fn("read_attachment", f"Read the text content of the attached file ({os.path.basename(self.attachment)}).",
                             {"offset": {"type": "integer"}}, []))
            ext = os.path.splitext(self.attachment)[1].lower()
            if ext in IMG_EXT and os.environ.get("LLM_SUPPORTS_VISION", "0") == "1":
                tools.append(_fn("describe_image", "Ask a vision model a question about the attached image.",
                                 {"question": {"type": "string"}}, ["question"]))
            if ext in AUDIO_EXT and os.environ.get("AUDIO_TRANSCRIBE_MODEL"):
                tools.append(_fn("transcribe_audio", "Transcribe the attached audio file.", {}, []))
        if self.enable_python:
            hint = f" The attached file is available in the working directory as '{os.path.basename(self.attachment)}'." if self.attachment else ""
            tools.append(_fn("python", "Execute Python 3 code and return stdout/stderr. Use print() for output." + hint,
                             {"code": {"type": "string"}}, ["code"]))
        return tools

    def call(self, name, args):
        t0 = time.time()
        try:
            fn = getattr(self, "t_" + name, None)
            if fn is None:
                out = f"ERROR: unknown tool {name}"
            else:
                out = fn(**(args or {}))
        except TypeError as e:
            out = f"ERROR: bad arguments for {name}: {e}"
        except Exception as e:
            out = f"ERROR: {type(e).__name__}: {e}"
        out = str(out)
        if len(out) > PAGE_CHARS + 400:
            out = out[:PAGE_CHARS] + "\n[truncated]"
        with TRACKER.role_scope("tool"):
            TRACKER.record("tool", time.time() - t0, extra={"tool": name, "args": json.dumps(args, ensure_ascii=False)[:300],
                                                            "ok": not out.startswith("ERROR")})
        return out

    def t_web_search(self, query, **_):
        if os.environ.get("TAVILY_API_KEY"):
            body = json.dumps({"api_key": os.environ["TAVILY_API_KEY"], "query": query, "max_results": 8}).encode()
            req = urllib.request.Request("https://api.tavily.com/search", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=40) as r:
                res = json.loads(r.read())
            return "\n\n".join(f"[{i + 1}] {x.get('title')}\n{x.get('url')}\n{x.get('content', '')[:500]}"
                               for i, x in enumerate(res.get("results", []))) or "no results"
        if os.environ.get("SERPER_API_KEY"):
            body = json.dumps({"q": query, "num": 8}).encode()
            req = urllib.request.Request("https://google.serper.dev/search", data=body,
                                         headers={"Content-Type": "application/json", "X-API-KEY": os.environ["SERPER_API_KEY"]})
            with urllib.request.urlopen(req, timeout=40) as r:
                res = json.loads(r.read())
            items = res.get("organic", [])
            ab = res.get("answerBox")
            head = f"Answer box: {json.dumps(ab, ensure_ascii=False)[:600]}\n\n" if ab else ""
            return head + ("\n\n".join(f"[{i + 1}] {x.get('title')}\n{x.get('link')}\n{x.get('snippet', '')}"
                                       for i, x in enumerate(items)) or "no results")
        data, _ = _http_get("https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query}))
        page = data.decode("utf-8", "ignore")
        results = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>', page, re.S)
        out = []
        for i, (href, title, snip) in enumerate(results[:8]):
            m = re.search(r"uddg=([^&]+)", href)
            url = urllib.parse.unquote(m.group(1)) if m else href
            out.append(f"[{i + 1}] {html_to_text(title)}\n{url}\n{html_to_text(snip)}")
        return "\n\n".join(out) or "no results (DuckDuckGo may be rate-limiting; set TAVILY_API_KEY or SERPER_API_KEY)"

    def t_fetch_url(self, url, offset=0, **_):
        try:
            data, ctype = _http_get(url, timeout=40)
        except urllib.error.HTTPError as e:
            # Explain failures so the model can switch sources instead of repeating the same URL.
            hint = {401: "login required", 403: "the site blocks automated access",
                    404: "page not found", 429: "rate limited"}.get(e.code, "request rejected")
            return f"ERROR: HTTP {e.code} ({hint}) for {url} — use a different source for this fact."
        except urllib.error.URLError as e:
            return f"ERROR: cannot reach {url} ({e.reason}) — use a different source for this fact."
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e} while fetching {url} — use a different source."
        if _is_pdf(data):
            text = _pdf_text(data)
        else:
            text = html_to_text(data.decode("utf-8", "ignore"))
        return _page(text, offset)

    def t_read_attachment(self, offset=0, **_):
        if not self.attachment:
            return "ERROR: this task has no attachment"
        if self._attachment_text is None:
            self._attachment_text = extract_file(self.attachment)
        return _page(self._attachment_text, offset)

    def t_python(self, code, **_):
        if not self.enable_python:
            return "ERROR: python tool disabled"
        timeout = int(os.environ.get("PYTHON_TOOL_TIMEOUT", 60))
        script = os.path.join(self.workdir, "_snippet.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            p = subprocess.run([sys.executable, script], cwd=self.workdir, capture_output=True, text=True, timeout=timeout)
            out = p.stdout
            if p.stderr:
                out += "\n[stderr]\n" + p.stderr[-3000:]
            return out.strip() or "(no output; use print())"
        except subprocess.TimeoutExpired:
            return f"ERROR: timed out after {timeout}s"

    def t_describe_image(self, question, **_):
        from .llm import chat
        ext = os.path.splitext(self.attachment)[1].lower().lstrip(".")
        ext = "jpeg" if ext == "jpg" else ext
        with open(self.attachment, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        msg = chat([{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}}]}],
            model=os.environ.get("VISION_MODEL") or None, role="vision")
        return msg.get("content") or ""

    def t_transcribe_audio(self, **_):
        import uuid
        base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        boundary = uuid.uuid4().hex
        with open(self.attachment, "rb") as f:
            audio = f.read()
        fname = os.path.basename(self.attachment)
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{os.environ['AUDIO_TRANSCRIBE_MODEL']}\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fname}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + audio + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(f"{base}/audio/transcriptions", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                                              "Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as r:
            res = json.loads(r.read())
        TRACKER.record("audio", time.time() - t0, extra={"model": os.environ["AUDIO_TRANSCRIBE_MODEL"]})
        return res.get("text", "")

    def cleanup(self):
        shutil.rmtree(self.workdir, ignore_errors=True)


def _fn(name, desc, props, required):
    params = {"type": "object", "properties": props or {"note": {"type": "string", "description": "optional, leave empty"}}}
    if required:
        params["required"] = required
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": params}}
