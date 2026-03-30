export default {
    content: ["./index.html", "./src/**/*.{ts,tsx}"],
    theme: {
        extend: {
            colors: {
                navy: "var(--color-navy)",
                blue: "var(--color-blue)",
                canvas: "var(--color-canvas)",
                rule: "var(--color-rule)",
                heat: "var(--color-heat)",
                humidity: "var(--color-humidity)",
                caution: "var(--color-caution)"
            },
            fontFamily: {
                sans: ["ui-sans-serif", "system-ui", "sans-serif"]
            },
            boxShadow: {
                panel: "0 12px 32px rgba(18, 50, 74, 0.08)"
            }
        }
    },
    plugins: []
};
