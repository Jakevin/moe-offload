import json, subprocess, time, urllib.request, base64, websocket, os, shutil
CH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"; PORT = 9339; prof = os.path.abspath("chrome_profile")
p = subprocess.Popen([CH, "--headless=new", f"--remote-debugging-port={PORT}", "--remote-debugging-address=127.0.0.1", f"--remote-allow-origins=http://127.0.0.1:{PORT}", f"--user-data-dir={prof}",
                      "--window-size=1500,1000", "--no-first-run", "--no-default-browser-check", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(50):
        try: tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json")); break
        except Exception: time.sleep(0.2)
    ws = websocket.create_connection([t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"], timeout=600, suppress_origin=True); n = [0]
    def cmd(m, **pa):
        n[0] += 1; ws.send(json.dumps({"id": n[0], "method": m, "params": pa}))
        while True:
            r = json.loads(ws.recv())
            if r.get("id") == n[0]: return r.get("result", r)
    ev = lambda js: cmd("Runtime.evaluate", expression=js, awaitPromise=True, returnByValue=True).get("result", {}).get("value")
    cmd("Page.enable"); cmd("Page.navigate", url="http://127.0.0.1:8095/ui/"); time.sleep(3)
    def snap(name):
        d = cmd("Page.captureScreenshot", format="png"); open(name, "wb").write(base64.b64decode(d["data"])); print("saved", name)
    ev("document.getElementById('modeThinkBtn').click()")
    ev("""(()=>{const t=document.getElementById('userInput');t.value='用兩句話解釋為什麼天空是藍色的。';t.dispatchEvent(new Event('input',{bubbles:true}));document.getElementById('sendBtn').click();})()""")
    t0 = time.time()
    while time.time() - t0 < 400:
        time.sleep(2)
        busy = ev("!document.getElementById('stopBtn').disabled && document.getElementById('stopBtn').offsetParent!==null")
        if not busy and time.time() - t0 > 6: break
    ev("document.getElementById('togglePerfBtn').click()"); time.sleep(3)
    snap("webui_thinking_perf.png")
    print("messages_text:", (ev("document.getElementById('messagesContainer').innerText") or "")[:1500])
    print("perf_text:", (ev("document.getElementById('perfPanel').innerText") or "")[:800])
finally:
    p.terminate(); p.wait(10); shutil.rmtree(prof, ignore_errors=True)
