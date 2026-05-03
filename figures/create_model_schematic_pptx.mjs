import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const pptxgen = require("/Users/xichen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Codex";
pptx.subject = "Editable model schematics";
pptx.title = "Model Schematics";
pptx.company = "CSCI1470 Final Project";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};
pptx.defineLayout({ name: "POSTER", width: 16, height: 9 });
pptx.layout = "POSTER";

const C = {
  bg: "FBFBF8",
  ink: "1F2933",
  muted: "52616B",
  line: "23313D",
  gray: "6B7280",
  orange: "C2410C",
  input: "D9F0F4",
  enc: "C8E6C9",
  bridge: "F7D7A8",
  transformer: "F4C7D8",
  dec: "D7D9F7",
  out: "FFD6DC",
  white: "FFFFFF",
};

const font = "Aptos";
const titleFont = "Aptos Display";

function addText(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x, y, w, h,
    fontFace: opts.fontFace || font,
    fontSize: opts.size || 12,
    color: opts.color || C.ink,
    bold: opts.bold || false,
    margin: opts.margin ?? 0.04,
    breakLine: opts.breakLine || false,
    fit: "shrink",
    valign: opts.valign || "mid",
    align: opts.align || "left",
  });
}

function addBlock(slide, { x, y, w, h, fill, title, lines }) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: 0.08,
    fill: { color: fill },
    line: { color: C.line, width: 1.4 },
  });
  addText(slide, title, x + 0.25, y + 0.22, w - 0.5, 0.28, { size: 13, bold: true });
  lines.forEach((line, i) => {
    addText(slide, line.text, x + 0.25, y + 0.62 + i * 0.25, w - 0.5, 0.22, {
      size: line.small ? 9.5 : 10.5,
      color: line.small ? C.muted : C.ink,
    });
  });
}

function addArrow(slide, x1, y1, x2, y2, color = C.line, width = 1.8, dash = undefined) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color, width, beginArrowType: "none", endArrowType: "triangle", dash },
  });
}

function addLine(slide, x1, y1, x2, y2, color = C.line, width = 1.8, dash = undefined) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color, width, beginArrowType: "none", endArrowType: "none", dash },
  });
}

function addSkipPath(slide, points) {
  for (let i = 0; i < points.length - 2; i += 1) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[i + 1];
    addLine(slide, x1, y1, x2, y2, C.orange, 1.8, "dash");
  }
  const [x1, y1] = points[points.length - 2];
  const [x2, y2] = points[points.length - 1];
  addArrow(slide, x1, y1, x2, y2, C.orange, 1.8, "dash");
}

function addLabel(slide, text, x, y, w = 1.6) {
  addText(slide, text, x, y, w, 0.25, { size: 9, color: C.orange, bold: true });
}

function addFlowLabel(slide, text, x, y, w = 1.25) {
  addText(slide, text, x, y, w, 0.22, { size: 8.5, color: C.line, bold: true });
}

function addSkipTag(slide, text, x, y) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w: 0.65, h: 0.24,
    rectRadius: 0.04,
    fill: { color: "FFF7ED" },
    line: { color: C.orange, width: 1 },
  });
  addText(slide, text, x + 0.05, y + 0.035, 0.55, 0.13, {
    size: 8.5,
    bold: true,
    color: C.orange,
  });
}

function addConcat(slide, x, y, label = "+") {
  slide.addShape(pptx.ShapeType.ellipse, {
    x: x - 0.11,
    y: y - 0.11,
    w: 0.22,
    h: 0.22,
    fill: { color: C.white },
    line: { color: C.orange, width: 1.3 },
  });
  addText(slide, label, x - 0.09, y - 0.075, 0.18, 0.11, {
    size: 8.5,
    bold: true,
    color: C.orange,
    align: "center",
  });
}

function addLegend(slide, x, y) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w: 3.65, h: 0.72,
    rectRadius: 0.04,
    fill: { color: C.white },
    line: { color: "7B8794", width: 0.8 },
  });
  addArrow(slide, x + 0.25, y + 0.25, x + 0.88, y + 0.25);
  addText(slide, "main tensor flow", x + 1.02, y + 0.14, 1.5, 0.2, { size: 8.5, color: C.muted });
  addSkipTag(slide, "skip A", x + 0.25, y + 0.4);
  addText(slide, "copied to matching decoder concat", x + 1.02, y + 0.41, 2.35, 0.2, { size: 8.5, color: C.muted });
}

function addHeader(slide, title, subtitle) {
  slide.background = { color: C.bg };
  addText(slide, title, 0.55, 0.35, 12.6, 0.45, {
    size: 24,
    bold: true,
    fontFace: titleFont,
  });
  addText(slide, subtitle, 0.55, 0.86, 14.4, 0.3, { size: 11.5, color: C.muted });
}

function addSection(slide, text, x, y) {
  addText(slide, text, x, y, 1.5, 0.25, { size: 9.5, bold: true, color: "36454F" });
}

function addNote(slide, text) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.55, y: 7.55, w: 14.9, h: 0.85,
    rectRadius: 0.06,
    fill: { color: C.white },
    line: { color: "7B8794", width: 1 },
  });
  addText(slide, "Poster note", 0.82, 7.72, 1.4, 0.24, { size: 11.5, bold: true });
  addText(slide, text, 0.82, 8.02, 14.1, 0.24, { size: 10.5 });
}

function slideUnet() {
  const s = pptx.addSlide();
  addHeader(
    s,
    "Model 1: CNN U-Net Downscaling Network",
    "Fully convolutional encoder-decoder with two skip connections; each DoubleConv is Conv 3x3 + ReLU repeated twice."
  );
  addSection(s, "INPUT", 2.05, 1.55);
  addSection(s, "ENCODER", 4.2, 1.55);
  addSection(s, "DECODER", 10.25, 1.55);
  addSection(s, "OUTPUT", 13.45, 1.55);
  addLegend(s, 11.35, 0.86);

  addBlock(s, { x: 0.9, y: 2.1, w: 2.1, h: 1.25, fill: C.input, title: "Input tensor", lines: [
    { text: "variable feature inputs" }, { text: "e.g., Tmax, elevation, urban", small: true }, { text: "B x C_in x H x W", small: true },
  ]});
  addBlock(s, { x: 3.9, y: 2.1, w: 2.3, h: 1.25, fill: C.enc, title: "DoubleConv", lines: [
    { text: "feature extraction" }, { text: "preserves H x W", small: true },
  ]});
  addBlock(s, { x: 6.9, y: 2.1, w: 2.3, h: 1.25, fill: C.enc, title: "MaxPool + DoubleConv", lines: [
    { text: "downsample features" }, { text: "H/2 x W/2", small: true },
  ]});
  addBlock(s, { x: 6.9, y: 4.75, w: 2.3, h: 1.25, fill: C.bridge, title: "MaxPool + DoubleConv", lines: [
    { text: "deeper features" }, { text: "H/4 x W/4", small: true },
  ]});
  addBlock(s, { x: 9.9, y: 4.75, w: 2.45, h: 1.25, fill: C.dec, title: "UpConv + concat", lines: [
    { text: "upsample features" }, { text: "skip from matching encoder layer", small: true }, { text: "H/2 x W/2", small: true },
  ]});
  addBlock(s, { x: 9.9, y: 2.1, w: 2.45, h: 1.25, fill: C.dec, title: "UpConv + concat", lines: [
    { text: "refine full-resolution features" }, { text: "skip from matching encoder layer", small: true }, { text: "H x W", small: true },
  ]});
  addBlock(s, { x: 13.1, y: 2.1, w: 2.1, h: 1.25, fill: C.out, title: "1x1 Conv", lines: [
    { text: "predict target field" }, { text: "Downscaled Tmax", small: true }, { text: "B x C_out x H x W", small: true },
  ]});

  addArrow(s, 3.0, 2.73, 3.85, 2.73);
  addArrow(s, 6.2, 2.73, 6.85, 2.73);
  addArrow(s, 8.05, 3.35, 8.05, 4.65);
  addFlowLabel(s, "downsample", 8.25, 4.05);
  addArrow(s, 9.2, 5.38, 9.85, 5.38);
  addArrow(s, 11.13, 4.75, 11.13, 3.45);
  addFlowLabel(s, "upsample", 11.33, 4.05);
  addArrow(s, 12.35, 2.73, 13.05, 2.73);
  addSkipTag(s, "skip A", 5.48, 1.82);
  addConcat(s, 9.84, 2.73, "+A");
  addSkipTag(s, "skip B", 8.42, 3.55);
  addConcat(s, 9.84, 5.38, "+B");

  addNote(s, "Model keeps the target grid size through padding after each transpose convolution, then concatenates encoder features with decoder features.");
}

function slideTransformer() {
  const s = pptx.addSlide();
  addHeader(
    s,
    "Model 2: U-Net with Transformer Bottleneck",
    "Convolutional encoder-decoder with global spatial context added by a transformer at the lowest-resolution feature map."
  );
  addSection(s, "INPUT", 1.8, 1.55);
  addSection(s, "ENCODER", 4.1, 1.55);
  addSection(s, "BOTTLENECK", 8.35, 1.55);
  addSection(s, "DECODER", 12.15, 1.55);
  addSection(s, "OUTPUT", 14.35, 1.55);
  addLegend(s, 12.05, 0.86);

  addBlock(s, { x: 0.55, y: 2.05, w: 2.0, h: 1.15, fill: C.input, title: "Input tensor", lines: [
    { text: "variable feature inputs" }, { text: "e.g., Tmax, elevation, urban", small: true }, { text: "B x C_in x H x W", small: true },
  ]});
  addBlock(s, { x: 3.3, y: 2.05, w: 2.05, h: 1.15, fill: C.enc, title: "DoubleConv", lines: [
    { text: "feature extraction" }, { text: "H x W", small: true },
  ]});
  addBlock(s, { x: 5.95, y: 2.05, w: 2.05, h: 1.15, fill: C.enc, title: "MaxPool + DoubleConv", lines: [
    { text: "downsample features" }, { text: "H/2 x W/2", small: true },
  ]});
  addBlock(s, { x: 5.95, y: 4.0, w: 2.05, h: 1.15, fill: C.enc, title: "MaxPool + DoubleConv", lines: [
    { text: "deeper features" }, { text: "H/4 x W/4", small: true },
  ]});
  addBlock(s, { x: 5.95, y: 5.95, w: 2.05, h: 1.15, fill: C.bridge, title: "MaxPool + DoubleConv", lines: [
    { text: "lowest-resolution features" }, { text: "H/8 x W/8", small: true },
  ]});
  addBlock(s, { x: 8.55, y: 5.95, w: 2.4, h: 1.15, fill: C.transformer, title: "1x1 projection", lines: [
    { text: "project to embedding space" }, { text: "prepare spatial tokens", small: true },
  ]});
  addBlock(s, { x: 8.55, y: 3.9, w: 2.4, h: 1.35, fill: C.transformer, title: "Transformer encoder", lines: [
    { text: "3 layers, 8 heads" }, { text: "2D positional encoding", small: true }, { text: "tokens: H/8 x W/8", small: true },
  ]});
  addBlock(s, { x: 8.55, y: 2.05, w: 2.4, h: 1.15, fill: C.transformer, title: "1x1 projection", lines: [
    { text: "project back to feature map" }, { text: "restore decoder features", small: true },
  ]});
  addBlock(s, { x: 11.7, y: 5.95, w: 2.2, h: 1.15, fill: C.dec, title: "UpConv + concat", lines: [
    { text: "lowest-resolution features" }, { text: "skip from H/4 layer", small: true },
  ]});
  addBlock(s, { x: 11.7, y: 4.0, w: 2.2, h: 1.15, fill: C.dec, title: "UpConv + concat", lines: [
    { text: "upsample features" }, { text: "skip from H/2 layer", small: true },
  ]});
  addBlock(s, { x: 11.7, y: 2.05, w: 2.2, h: 1.15, fill: C.dec, title: "UpConv + concat", lines: [
    { text: "refine full-resolution features" }, { text: "skip from H layer", small: true },
  ]});
  addBlock(s, { x: 14.35, y: 2.05, w: 1.95, h: 1.15, fill: C.out, title: "1x1 Conv", lines: [
    { text: "predict target field" }, { text: "Downscaled Tmax", small: true }, { text: "B x C_out x H x W", small: true },
  ]});

  addArrow(s, 2.55, 2.62, 3.25, 2.62);
  addArrow(s, 5.35, 2.62, 5.9, 2.62);
  addArrow(s, 6.98, 3.2, 6.98, 3.9);
  addFlowLabel(s, "downsample", 7.18, 3.52);
  addArrow(s, 6.98, 5.15, 6.98, 5.85);
  addFlowLabel(s, "downsample", 7.18, 5.47);
  addArrow(s, 8.0, 6.52, 8.5, 6.52);
  addArrow(s, 9.75, 5.95, 9.75, 5.35);
  addFlowLabel(s, "tokens", 9.95, 5.66, 0.75);
  addArrow(s, 9.75, 3.9, 9.75, 3.3);
  addFlowLabel(s, "project back", 9.95, 3.6);
  addArrow(s, 10.95, 2.62, 11.65, 2.62);
  addArrow(s, 12.8, 3.2, 12.8, 3.9);
  addFlowLabel(s, "upsample", 13.0, 3.52);
  addArrow(s, 12.8, 5.15, 12.8, 5.85);
  addFlowLabel(s, "upsample", 13.0, 5.47);
  addArrow(s, 10.95, 6.52, 11.65, 6.52);
  addArrow(s, 13.9, 2.62, 14.3, 2.62);
  addSkipTag(s, "skip C", 8.12, 4.55);
  addConcat(s, 11.65, 6.52, "+C");
  addSkipTag(s, "skip A", 4.78, 1.82);
  addConcat(s, 11.65, 2.62, "+A");
  addSkipTag(s, "skip B", 8.12, 2.58);
  addConcat(s, 11.65, 4.57, "+B");

  addNote(s, "The transformer operates only at H/8 x W/8 resolution, adding long-range spatial context while the U-Net decoder recovers full-resolution structure.");
}

slideUnet();
slideTransformer();

await pptx.writeFile({ fileName: "CSCI1470-final-project/figures/model_schematics_editable.pptx" });
