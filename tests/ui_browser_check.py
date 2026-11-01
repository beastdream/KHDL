"""Optional Windows Chrome CDP check; serves only the trusted localhost app.

Run with the existing app on 127.0.0.1:8513. No training or dataset reads here.
"""
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'reports/ui_screenshots'
    folder.mkdir(exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix='houseprice-ui-', ignore_cleanup_errors=True) as profile:
        process = subprocess.Popen([
            r'C:\Program Files\Google\Chrome\Application\chrome.exe',
            '--headless=new', '--no-sandbox', '--disable-gpu', '--no-first-run',
            '--remote-debugging-port=0', '--user-data-dir=' + profile, 'about:blank'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            active = Path(profile) / 'DevToolsActivePort'
            for _ in range(150):
                if active.exists():
                    break
                time.sleep(.1)
            port = int(active.read_text().splitlines()[0])
            pages = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json'))
            url = next(p['webSocketDebuggerUrl'] for p in pages if p['type'] == 'page')
            with connect(url) as socket:
                sequence = 0

                def call(method, params=None):
                    nonlocal sequence
                    sequence += 1
                    socket.send(json.dumps({'id': sequence, 'method': method, 'params': params or {}}))
                    while True:
                        response = json.loads(socket.recv(timeout=30))
                        if response.get('id') == sequence:
                            if 'error' in response:
                                raise RuntimeError(response)
                            return response.get('result', {})

                def evaluate(expression):
                    response = call('Runtime.evaluate', {'expression': expression, 'returnByValue': True})
                    if 'exceptionDetails' in response:
                        raise RuntimeError(response)
                    return response['result'].get('value')

                def wait(expression):
                    for _ in range(120):
                        if evaluate(expression):
                            return
                        time.sleep(.5)
                    raise RuntimeError('UI condition timed out: ' + expression)

                def screenshot(name):
                    data = call('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': False})
                    (folder / (name + '.png')).write_bytes(base64.b64decode(data['data']))

                def click(selector):
                    evaluate(f"document.querySelector({json.dumps(selector)}).scrollIntoView({{block:'center',behavior:'instant'}})")
                    time.sleep(.5)
                    rectangle = evaluate(f"(() => {{const r=document.querySelector({json.dumps(selector)}).getBoundingClientRect();return {{x:r.x+r.width/2,y:r.y+r.height/2}}}})()")
                    call('Input.dispatchMouseEvent', {'type':'mousePressed','button':'left','clickCount':1,**rectangle})
                    call('Input.dispatchMouseEvent', {'type':'mouseReleased','button':'left','clickCount':1,**rectangle})

                def inspect(name, width):
                    result = evaluate('''(() => {
                        const main=document.querySelector('[data-testid="stMain"]');
                        const side=document.querySelector('[data-testid="stSidebar"]');
                        const texts=[...document.querySelectorAll('[data-testid="stWidgetLabel"] p')];
                        const submit=document.querySelector('[data-testid="stFormSubmitButton"] button');
                        const metric=document.querySelector('.st-key-prediction_result [data-testid="stMetricValue"]');
                        const color=e=>e?{color:getComputedStyle(e).color,background:getComputedStyle(e).backgroundColor,font:getComputedStyle(e).fontSize,weight:getComputedStyle(e).fontWeight}:null;
                        return {bodyWidth:document.documentElement.scrollWidth, viewport:innerWidth,
                          mainWidth:main.clientWidth,mainScroll:main.scrollWidth,
                          sidebar:color(side),labels:texts.map(e=>({text:e.innerText,...color(e)})),
                          button:color(submit),price:color(metric),
                          buttonText:color(submit?.querySelector('p')),
                          selectedTag:color(document.querySelector('[data-testid="stMultiSelectTagsContainer"] [data-tag]')),
                          sliderHTML:document.querySelector('[data-testid="stSlider"]')?.outerHTML,
                          chartCount:document.querySelectorAll('[data-testid="stPlotlyChart"]').length,
                          exceptions:document.querySelectorAll('[data-testid="stException"]').length};
                    })()''')
                    result.update(page=name, width=width)
                    slider_html = result.pop('sliderHTML', None)
                    if slider_html:
                        (folder / 'slider_dom.html').write_text(slider_html,encoding='utf-8')
                    assert result['exceptions'] == 0, result
                    assert result['bodyWidth'] <= width + 1, result
                    assert result['mainScroll'] <= result['mainWidth'] + 1, result
                    if result['selectedTag']:
                        assert result['selectedTag']['background'] == 'rgb(21, 34, 56)', result
                    results.append(result)
                    print(json.dumps(result, ensure_ascii=True), flush=True)

                call('Page.enable')
                call('Emulation.setDeviceMetricsOverride', {'width':1440,'height':1050,'deviceScaleFactor':1,'mobile':False})
                call('Page.navigate', {'url':'http://127.0.0.1:8513'})
                wait("document.querySelectorAll('[data-testid=stRadio] input').length===5")
                wait("document.querySelectorAll('[data-testid=stPlotlyChart]').length===2")
                screenshot('desktop-overview'); inspect('overview',1440)
                names = ['Khám phá dữ liệu','Dự đoán giá nhà','Hiệu năng mô hình','Về dự án']
                expected = ['Khám phá dữ liệu Train','Dự đoán giá bất động sản','Hiệu năng trên Validation','Về dự án']
                for index, (name, title) in enumerate(zip(names, expected), 1):
                    evaluate(f"document.querySelectorAll('[data-testid=stRadio] input')[{index}].click()")
                    wait(f"[...document.querySelectorAll('h2')].some(e=>e.innerText.includes({json.dumps(title)}))")
                    if index == 1:
                        wait("document.querySelectorAll('[data-testid=stPlotlyChart]').length===7")
                    if index == 2:
                        wait("!!document.querySelector('[data-testid=stFormSubmitButton] button')")
                        evaluate("document.querySelector('[data-testid=stFormSubmitButton] button').click()")
                        wait("!!document.querySelector('.st-key-prediction_result [data-testid=stMetricValue]')")
                        # Real pointer hover, keyboard focus, and temporary disabled DOM state.
                        selector = '[data-testid=stFormSubmitButton] button'
                        evaluate("document.querySelector('[data-testid=stFormSubmitButton] button').scrollIntoView({block:'center',behavior:'instant'})")
                        time.sleep(1)
                        rectangle = evaluate("(() => {const r=document.querySelector('[data-testid=stFormSubmitButton] button').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()")
                        call('Input.dispatchMouseEvent', {'type':'mouseMoved',**rectangle})
                        wait("document.querySelector('[data-testid=stFormSubmitButton] button').matches(':hover')")
                        states = {'hover':evaluate("getComputedStyle(document.querySelector('[data-testid=stFormSubmitButton] button')).backgroundColor")}
                        assert states['hover'] == 'rgb(230, 62, 69)', states
                        call('Input.dispatchKeyEvent',{'type':'keyDown','key':'Tab','code':'Tab','windowsVirtualKeyCode':9})
                        call('Input.dispatchKeyEvent',{'type':'keyUp','key':'Tab','code':'Tab','windowsVirtualKeyCode':9})
                        evaluate("document.querySelector('[data-testid=stFormSubmitButton] button').focus()")
                        states['focus'] = evaluate("getComputedStyle(document.querySelector('[data-testid=stFormSubmitButton] button')).outlineStyle")
                        assert states['focus'] == 'solid', states
                        evaluate("document.querySelector('[data-testid=stFormSubmitButton] button').disabled=true")
                        states['disabled'] = evaluate("getComputedStyle(document.querySelector('[data-testid=stFormSubmitButton] button')).backgroundColor")
                        assert states['disabled'] == 'rgb(220, 227, 237)', states
                        evaluate("document.querySelector('[data-testid=stFormSubmitButton] button').disabled=false")
                        results.append({'button_states':states})
                        click('[data-testid=stSelectbox] [role=combobox]')
                        evaluate("document.querySelector('[data-testid=stSelectbox] [role=combobox]').focus()")
                        call('Input.dispatchKeyEvent',{'type':'keyDown','key':'ArrowDown','code':'ArrowDown','windowsVirtualKeyCode':40})
                        call('Input.dispatchKeyEvent',{'type':'keyUp','key':'ArrowDown','code':'ArrowDown','windowsVirtualKeyCode':40})
                        print('Dropdown opened using keyboard',flush=True)
                        wait("document.querySelectorAll('[role=option]').length>0")
                        screenshot('desktop-selectbox')
                        results.append({'dropdown':evaluate("[...document.querySelectorAll('[role=option]')].slice(0,3).map(e=>({text:e.innerText,color:getComputedStyle(e).color,background:getComputedStyle(e).backgroundColor}))")})
                        call('Input.dispatchKeyEvent',{'type':'keyDown','key':'Escape','code':'Escape','windowsVirtualKeyCode':27})
                        call('Input.dispatchKeyEvent',{'type':'keyUp','key':'Escape','code':'Escape','windowsVirtualKeyCode':27})
                        evaluate("document.querySelector('.st-key-prediction_result').scrollIntoView({block:'center'})")
                    if index == 3:
                        wait("document.querySelectorAll('[data-testid=stPlotlyChart]').length===4")
                    time.sleep(1)
                    screenshot('desktop-' + str(index)); inspect(name,1440)
                # Mobile prediction: close sidebar using its native collapse control.
                evaluate("document.querySelectorAll('[data-testid=stRadio] input')[2].click()")
                wait("!!document.querySelector('[data-testid=stFormSubmitButton] button')")
                for width in (1024,768,360):
                    call('Emulation.setDeviceMetricsOverride', {'width':width,'height':1000,'deviceScaleFactor':1,'mobile':False})
                    time.sleep(1)
                    evaluate("document.querySelector('[data-testid=stSidebarCollapseButton] button')?.click()")
                    time.sleep(.5)
                    evaluate("document.querySelector('[data-testid=stFormSubmitButton] button').click()")
                    wait("!!document.querySelector('.st-key-prediction_result [data-testid=stMetricValue]')")
                    evaluate("document.querySelector('.st-key-prediction_result').scrollIntoView({block:'center'})")
                    time.sleep(1)
                    screenshot('prediction-' + str(width)); inspect('prediction',width)
                for index, name in [(0,'overview'),(1,'exploration'),(3,'performance'),(4,'about')]:
                    evaluate(f"document.querySelectorAll('[data-testid=stRadio] input')[{index}].click()")
                    chart_count = {0:2,1:7,3:4,4:0}[index]
                    wait(f"document.querySelectorAll('[data-testid=stPlotlyChart]').length==={chart_count}")
                    time.sleep(1)
                    evaluate("document.querySelector('[data-testid=stMain]').scrollTop=0")
                    screenshot('mobile-' + name); inspect(name,360)
                    if index in (1,3):
                        for chart_index in ([0,4] if index == 1 else [0,1]):
                            evaluate(f"document.querySelectorAll('[data-testid=stPlotlyChart]')[{chart_index}].scrollIntoView({{block:'center'}})")
                            time.sleep(.5)
                            screenshot('mobile-' + name + '-chart-' + str(chart_index))
                (ROOT / 'reports/ui_browser_verification.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
                try:
                    call('Browser.close')
                except Exception:
                    pass
        finally:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == '__main__':
    main()
