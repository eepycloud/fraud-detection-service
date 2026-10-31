"""Cloud-only end-to-end checks and desktop/mobile screenshots."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright,expect

BASE="https://fraud-detection-service-srsh.onrender.com"
out=Path("reports");out.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch()
    page=browser.new_page(viewport={"width":1440,"height":1000},device_scale_factor=1)
    errors=[]
    page.on("pageerror",lambda e:errors.append(str(e)))
    page.goto(BASE,wait_until="networkidle",timeout=120000)
    expect(page).to_have_title("Fraud Observatory · Model monitoring")
    expect(page.locator("#count")).to_have_text("5,000",timeout=30000)
    expect(page.locator("#pr")).to_have_text("0.827")
    page.get_by_role("button",name="Replay 500 transactions",exact=True).click()
    expect(page.get_by_role("button",name="Replay 500 transactions",exact=True)).to_be_enabled(timeout=60000)
    expect(page.locator("#source-note")).to_contain_text("Transactions scored by browser")
    assert int(page.locator("#count").inner_text().replace(",",""))>=500
    page.get_by_role("button",name="Load higher-risk example",exact=True).click()
    page.get_by_role("button",name="Score transaction",exact=True).click()
    expect(page.locator("#score-result")).to_contain_text("Fraud score",timeout=30000)
    page.locator("#source").select_option("live")
    expect(page.locator("#source-note")).to_contain_text("Requests received")
    expect(page.locator("#pr")).to_have_text("—")
    expect(page.locator("#label-note")).to_have_text("0 known outcomes")
    page.locator("#source").select_option("demo")
    expect(page.locator("#count")).to_have_text("5,000")
    page.screenshot(path=str(out/"dashboard-desktop.png"),full_page=True)
    page.set_viewport_size({"width":390,"height":844})
    expect(page.locator("h1")).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"),"Mobile page overflows horizontally"
    page.screenshot(path=str(out/"dashboard-mobile.png"),full_page=True)
    assert not errors,errors
    (out/"browser-results.json").write_text(json.dumps({"status":"passed","desktop":[1440,1000],"mobile":[390,844],
                                                      "checks":["dashboard","labeled sample","replay 500","live scoring","unlabeled metric handling","mobile overflow","console errors"]},indent=2))
    browser.close()
print("PASS: cloud browser replay, scoring, monitoring, desktop and mobile")
