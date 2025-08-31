from playwright.sync_api import Page, expect

def test_backtesting_ui(page: Page):
    """
    This test verifies that the backtesting UI can be loaded and a backtest can be run.
    """
    # 1. Arrange: Go to the application homepage.
    page.goto("http://localhost:8080")
    page.screenshot(path="jules-scratch/verification/verification_before.png")

    # 2. Act: Click the "Run Backtest" button.
    run_button = page.get_by_role("button", name="Run Backtest")
    run_button.click()

    # Wait for the output to appear
    page.wait_for_timeout(30000)
    page.screenshot(path="jules-scratch/verification/verification_after.png")

    # 3. Assert: Check that the output is displayed.
    # We expect the raw output to contain some text.
    expect(page.locator('code').get_by_text("STRATEGY PERFORMANCE SUMMARY")).to_be_visible(timeout=60000)

    # 4. Screenshot: Capture the final result for visual verification.
    page.screenshot(path="jules-scratch/verification/verification.png")
