import esbuild from "esbuild";

async function build(platform, format, name) {
  return esbuild
    .build({
      entryPoints: ["./src/orpheus.ts"],
      bundle: true,
      minify: true,
      minifySyntax: true,
      treeShaking: true,
      logLevel: "info",

      outfile: `./dist/${name}`,
      platform,
      format,
      external: platform === "node" ? ["@huggingface/transformers"] : undefined,
    })
    .catch(() => process.exit(1));
}

build("browser", "esm", "orpheus.js");
build("node", "esm", "orpheus.node.mjs");
build("node", "cjs", "orpheus.node.cjs");
