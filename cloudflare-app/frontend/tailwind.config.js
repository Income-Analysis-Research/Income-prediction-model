/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        navy: {
          950: "#040b18",
          900: "#060e20",
          850: "#0a1428",
          800: "#0d1b35",
          700: "#112240",
          600: "#1a3050",
          500: "#1e3a5f",
        },
        slate: {
          850: "#0f172a",
        },
        gold: {
          400: "#f5c842",
          500: "#e6b800",
          600: "#c9a027",
          700: "#a07820",
        },
        signal: {
          green: "#00d68f",
          red:   "#ff3d71",
          blue:  "#0095ff",
          amber: "#ffaa00",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Menlo", "monospace"],
      },
      boxShadow: {
        card:  "0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3)",
        glow:  "0 0 20px rgba(37,99,235,0.15)",
        inner: "inset 0 1px 0 rgba(255,255,255,0.05)",
      },
      animation: {
        "fade-in": "fadeIn 0.3s ease-in-out",
        pulse: "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
      keyframes: {
        fadeIn: {
          "0%":   { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};
