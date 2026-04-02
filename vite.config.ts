import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/numerical_experiment_cosmosim/",
  plugins: [react()],
});
