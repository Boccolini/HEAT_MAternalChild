import { test, expect } from "@playwright/test";

test("overview route loads", async ({ page }) => {
  await page.goto("http://127.0.0.1:5173/overview");
  await expect(page.getByText("Child Health Observatory")).toBeVisible();
});
