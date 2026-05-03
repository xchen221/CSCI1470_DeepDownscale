import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const pptxgen = require("/Users/xichen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const pptx = new pptxgen();
pptx.defineLayout({ name: "WIDE", width: 16, height: 9 });
pptx.layout = "WIDE";
pptx.author = "Codex";
pptx.subject = "Editable model schematics";
pptx.title = "Model Schematics";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};

const C = {
  bg: "FCFBF7",
  ink: "17212B",
  muted: "4B5B66",
  line: "17212B",
  input: "DDF3F7",
  enc: "CDEBCE",
  bridge: "F9DDAA",
  bottleneck: "F4C7DA",
  dec: "D7DAFA",
  out: "FFD5DE",
  tagFill: "FFF7ED",
  orange: "C2410C",
  white: "FFFFFF",
};

function text(slide, value, x, y, w, h, opts = {}) {
  slide.addText(value, {
    x, y, w, h,
    fontFace: opts.fontFace || "Aptos",
    fontSize: opts.size || 13,
    bold: opts.bold || false,
    color: opts.color || C.ink,
    margin: opts.margin ?? 0.04,
    valign: opts.valign || "mid",
    align: opts.align || "left",
    fit: "shrink",
    breakLine: opts.breakLine || false,
  });
}

function block(slide, x, y, w, h, fill, title, lines = []) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: 0.08,
    fill: { color: fill },
    line: { color: C.line, width: 1.2 },
  });
  text(slide, title, x + 0.18, y + 0.11, w - 0.36, 0.5, { size: 14.2, bold: true });
  lines.forEach((line, i) => {
    text(slide, line, x + 0.18, y + 0.66 + i * 0.24, w - 0.36, 0.21, {
      size: i === 0 ? 12.2 : 11.2,
      color: i === 0 ? C.ink : C.muted,
    });
  });
}

function arrow(slide, x1, y1, x2, y2, color = C.line) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color, width: 2.1, beginArrowType: "none", endArrowType: "triangle" },
  });
}

function skipTag(slide, label, x, y) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w: 0.72, h: 0.28,
    rectRadius: 0.04,
    fill: { color: C.tagFill },
    line: { color: C.orange, width: 1.1 },
  });
  text(slide, label, x + 0.07, y + 0.04, 0.58, 0.13, {
    size: 9.4,
    bold: true,
    color: C.orange,
  });
}

function concat(slide, label, x, y) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x: x - 0.13, y: y - 0.13, w: 0.26, h: 0.26,
    fill: { color: C.white },
    line: { color: C.orange, width: 1.2 },
  });
  text(slide, label, x - 0.1, y - 0.08, 0.2, 0.12, {
    size: 9.4,
    bold: true,
    color: C.orange,
    align: "center",
  });
}

function section(slide, label, x, y) {
  text(slide, label, x, y, 1.3, 0.22, {
    size: 11.2,
    bold: true,
    color: "33424E",
  });
}

function header(slide, title, subtitle) {
  slide.background = { color: C.bg };
  text(slide, title, 0.55, 0.32, 11.4, 0.45, {
    size: 27,
    bold: true,
    fontFace: "Aptos Display",
  });
  text(slide, subtitle, 0.58, 0.83, 13.5, 0.28, { size: 13.5, color: C.muted });
  slide.addShape(pptx.ShapeType.line, {
    x: 0.55, y: 1.25, w: 14.9, h: 0,
    line: { color: "D0D7DE", width: 1 },
  });
}

function legend(slide) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 12.4, y: 0.38, w: 3.05, h: 0.78,
    rectRadius: 0.05,
    fill: { color: C.white },
    line: { color: "9AA6B2", width: 0.8 },
  });
  arrow(slide, 12.62, 0.68, 13.28, 0.68);
  text(slide, "main tensor flow", 13.4, 0.58, 1.5, 0.18, { size: 9.4, color: C.muted });
  skipTag(slide, "skip A", 12.62, 0.84);
  text(slide, "copied to concat", 13.4, 0.85, 1.4, 0.16, { size: 9.4, color: C.muted });
}

function note(slide, value) {
  text(slide, value, 0.65, 8.2, 14.8, 0.28, { size: 11.5, color: C.muted });
}

function unetSlide() {
  const s = pptx.addSlide();
  header(s, "Model 1: CNN U-Net Downscaling Network", "Architecture from model_cnn.py: two encoder downsamples, two decoder upsamples, and two skip concatenations.");
  legend(s);
  section(s, "INPUT", 1.55, 1.45);
  section(s, "ENCODER", 4.1, 1.45);
  section(s, "DECODER", 10.4, 1.45);
  section(s, "OUTPUT", 13.5, 1.45);

  block(s, 0.65, 2.15, 2.15, 1.25, C.input, "Input tensor", ["Variable feature inputs", "B x C_in x H x W"]);
  block(s, 3.6, 2.15, 2.35, 1.25, C.enc, "DoubleConv", ["Feature extraction", "H x W"]);
  block(s, 6.55, 2.15, 2.35, 1.25, C.enc, "MaxPool + DoubleConv", ["Downsample features", "H/2 x W/2"]);
  block(s, 6.55, 4.78, 2.35, 1.25, C.bridge, "MaxPool + DoubleConv", ["Deepest CNN features", "H/4 x W/4"]);
  block(s, 9.85, 4.78, 2.55, 1.25, C.dec, "UpConv + concat", ["Upsample with skip B", "H/2 x W/2"]);
  block(s, 9.85, 2.15, 2.55, 1.25, C.dec, "UpConv + concat", ["Upsample with skip A", "H x W"]);
  block(s, 13.25, 2.15, 2.1, 1.25, C.out, "1x1 Conv", ["Predict target field", "B x C_out x H x W"]);

  arrow(s, 2.8, 2.78, 3.55, 2.78);
  arrow(s, 5.95, 2.78, 6.5, 2.78);
  arrow(s, 7.72, 3.4, 7.72, 4.7);
  text(s, "downsample", 7.94, 4.08, 1.1, 0.18, { size: 10.4, bold: true });
  arrow(s, 8.9, 5.4, 9.8, 5.4);
  arrow(s, 11.12, 4.78, 11.12, 3.5);
  text(s, "upsample", 11.34, 4.08, 0.9, 0.18, { size: 10.4, bold: true });
  arrow(s, 12.4, 2.78, 13.2, 2.78);

  skipTag(s, "skip A", 5.1, 1.83);
  concat(s, "+A", 9.75, 2.78);
  skipTag(s, "skip B", 8.05, 3.58);
  concat(s, "+B", 9.75, 5.4);
  note(s, "Correctness check: decoder first concatenates the H/2 encoder feature, then the H encoder feature, matching the forward pass.");
}

function transformerSlide() {
  const s = pptx.addSlide();
  header(s, "Model 2: U-Net with Transformer Bottleneck", "Architecture from model_transformer.py: three encoder downsamples, transformer at H/8, then three decoder upsamples.");
  legend(s);
  section(s, "INPUT", 1.15, 1.45);
  section(s, "ENCODER", 3.7, 1.45);
  section(s, "BOTTLENECK", 7.8, 1.45);
  section(s, "DECODER", 12.05, 1.45);
  section(s, "OUTPUT", 14.6, 1.45);

  block(s, 0.35, 2.15, 1.95, 1.18, C.input, "Input tensor", ["Variable features", "B x C_in x H x W"]);
  block(s, 2.95, 2.15, 2.05, 1.18, C.enc, "DoubleConv", ["Feature extraction", "H x W"]);
  block(s, 5.55, 2.15, 2.05, 1.18, C.enc, "MaxPool + DoubleConv", ["Downsample features", "H/2 x W/2"]);
  block(s, 5.55, 3.88, 2.05, 1.18, C.enc, "MaxPool + DoubleConv", ["Deeper features", "H/4 x W/4"]);
  block(s, 5.55, 5.62, 2.05, 1.18, C.bridge, "MaxPool + DoubleConv", ["Lowest resolution", "H/8 x W/8"]);
  block(s, 8.0, 5.62, 1.75, 1.18, C.bottleneck, "1x1 projection", ["To embeddings", "H/8 tokens"]);
  block(s, 10.0, 5.62, 1.95, 1.18, C.bottleneck, "Transformer", ["2D pos. encoding", "self-attention"]);
  block(s, 12.2, 5.62, 1.75, 1.18, C.bottleneck, "1x1 projection", ["Back to features", "H/8 x W/8"]);
  block(s, 12.2, 3.88, 2.05, 1.18, C.dec, "UpConv + concat", ["Upsample with skip C", "H/4 x W/4"]);
  block(s, 12.2, 2.15, 2.05, 1.18, C.dec, "UpConv + concat", ["Upsample with skip B", "H/2 x W/2"]);
  block(s, 12.2, 0.98, 2.05, 1.02, C.dec, "UpConv + concat", ["Upsample with skip A", "H x W"]);
  block(s, 14.65, 0.98, 1.25, 1.02, C.out, "1x1 Conv", ["Output", "target field"]);

  arrow(s, 2.3, 2.74, 2.9, 2.74);
  arrow(s, 5.0, 2.74, 5.5, 2.74);
  arrow(s, 6.58, 3.33, 6.58, 3.82);
  arrow(s, 6.58, 5.06, 6.58, 5.55);
  arrow(s, 7.6, 6.21, 7.95, 6.21);
  arrow(s, 9.75, 6.21, 9.95, 6.21);
  arrow(s, 11.95, 6.21, 12.15, 6.21);
  arrow(s, 13.22, 5.62, 13.22, 5.12);
  arrow(s, 13.22, 3.88, 13.22, 3.43);
  arrow(s, 13.22, 2.15, 13.22, 2.05);
  arrow(s, 14.25, 1.49, 14.6, 1.49);

  text(s, "downsample", 6.78, 3.52, 1.0, 0.16, { size: 9.6, bold: true });
  text(s, "downsample", 6.78, 5.25, 1.0, 0.16, { size: 9.6, bold: true });
  text(s, "upsample", 13.43, 5.25, 0.9, 0.16, { size: 9.6, bold: true });
  text(s, "upsample", 13.43, 3.52, 0.9, 0.16, { size: 9.6, bold: true });
  text(s, "upsample", 13.43, 2.04, 0.9, 0.16, { size: 9.6, bold: true });

  skipTag(s, "skip A", 4.25, 1.84);
  skipTag(s, "skip B", 7.62, 2.45);
  skipTag(s, "skip C", 7.62, 4.18);
  concat(s, "+A", 12.1, 1.49);
  concat(s, "+B", 12.1, 2.74);
  concat(s, "+C", 12.1, 4.47);
  note(s, "Correctness check: after post-transform at H/8, the decoder enters skip C at H/4, then skip B at H/2, then skip A at H.");
}

unetSlide();
transformerSlide();

await pptx.writeFile({ fileName: "CSCI1470-final-project/figures/model_schematics.pptx" });
