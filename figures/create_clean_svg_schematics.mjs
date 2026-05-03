import { writeFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const sharp = require("/Users/xichen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp");

const css = `
  .bg{fill:transparent}
  .title{font:700 48px Arial,Helvetica,sans-serif;fill:#17212b}
  .subtitle{font:400 25px Arial,Helvetica,sans-serif;fill:#4b5b66}
  .section{font:700 25px Arial,Helvetica,sans-serif;fill:#33424e;letter-spacing:.8px}
  .block-title{font:700 25px Arial,Helvetica,sans-serif;fill:#17212b}
  .block-main{font:400 22px Arial,Helvetica,sans-serif;fill:#17212b}
  .block-sub{font:400 19px Arial,Helvetica,sans-serif;fill:#4b5b66}
  .note{font:400 20px Arial,Helvetica,sans-serif;fill:#4b5b66}
  .shape{stroke:#17212b;stroke-width:2.5;rx:9;ry:9}
  .input{fill:#ddf3f7}.enc{fill:#cdebce}.bridge{fill:#f9ddaa}.bot{fill:#f4c7da}.dec{fill:#d7dafa}.out{fill:#ffd5de}
  .arrow{stroke:#17212b;stroke-width:3;fill:none;marker-end:url(#arrow)}
  .flow-label{font:700 20px Arial,Helvetica,sans-serif;fill:#17212b}
  .skip-tag{fill:#fff7ed;stroke:#c2410c;stroke-width:2;rx:6;ry:6}
  .skip-text{font:700 17px Arial,Helvetica,sans-serif;fill:#c2410c}
  .concat{fill:#fff;stroke:#c2410c;stroke-width:2.2}
  .concat-text{font:700 18px Arial,Helvetica,sans-serif;fill:#c2410c}
  .legend{fill:#fff;stroke:#9aa6b2;stroke-width:1.5;rx:8;ry:8}
`;

function defs() {
  return `<defs>
    <style>${css}</style>
    <marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">
      <path d="M0,0 L9,4.5 L0,9 z" fill="#17212b"/>
    </marker>
  </defs>`;
}

function block(x, y, w, h, cls, title, main, sub) {
  const titleLines = title.includes(" + ") ? title.split(" + ").map((part, i) => i === 0 ? part + " +" : part) : [title];
  const titleSvg = titleLines.map((line, i) =>
    `<text class="block-title" x="${x + 28}" y="${y + 39 + i * 29}">${line}</text>`
  ).join("\n  ");
  const mainY = y + (titleLines.length > 1 ? 100 : 84);
  const subY = mainY + 30;
  return `<rect class="shape ${cls}" x="${x}" y="${y}" width="${w}" height="${h}"/>
  ${titleSvg}
  <text class="block-main" x="${x + 28}" y="${mainY}">${main}</text>
  <text class="block-sub" x="${x + 28}" y="${subY}">${sub}</text>`;
}

function arrow(x1, y1, x2, y2) {
  return `<path class="arrow" d="M ${x1} ${y1} L ${x2} ${y2}"/>`;
}

function tag(label, x, y) {
  return `<rect class="skip-tag" x="${x}" y="${y}" width="92" height="34"/>
  <text class="skip-text" x="${x + 12}" y="${y + 24}">${label}</text>`;
}

function concat(label, x, y) {
  return `<circle class="concat" cx="${x}" cy="${y}" r="18"/>
  <text class="concat-text" x="${x - 12}" y="${y + 6}">${label}</text>`;
}

function legend(x, y) {
  return `<rect class="legend" x="${x}" y="${y}" width="395" height="86"/>
  ${arrow(x + 28, y + 32, x + 105, y + 32)}
  <text class="block-sub" x="${x + 125}" y="${y + 39}">main tensor flow</text>
  ${tag("skip A", x + 28, y + 49)}
  <text class="block-sub" x="${x + 125}" y="${y + 70}">copied to matching concat</text>`;
}

function unetSvg() {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1000" viewBox="0 0 1800 1000">
  ${defs()}
  <rect class="bg" width="1800" height="1000"/>
  <text class="title" x="80" y="78">Model 1: CNN U-Net Downscaling Network</text>
  <text class="subtitle" x="80" y="118">Two encoder downsamples, two decoder upsamples, and two skip concatenations.</text>
  ${legend(1320, 48)}
  <line x1="80" y1="150" x2="1720" y2="150" stroke="#d0d7de" stroke-width="1.5"/>

  <text class="section" x="200" y="200">INPUT</text>
  <text class="section" x="535" y="200">ENCODER</text>
  <text class="section" x="1245" y="200">DECODER</text>
  <text class="section" x="1580" y="200">OUTPUT</text>

  ${block(95, 255, 245, 145, "input", "Input tensor", "Variable feature inputs", "B x C_in x H x W")}
  ${block(430, 255, 270, 145, "enc", "DoubleConv", "Feature extraction", "H x W")}
  ${block(785, 255, 285, 165, "enc", "MaxPool + DoubleConv", "Downsample features", "H/2 x W/2")}
  ${block(785, 575, 285, 165, "bridge", "MaxPool + DoubleConv", "Deepest CNN features", "H/4 x W/4")}
  ${block(1165, 575, 295, 145, "dec", "UpConv + concat", "Upsample with skip B", "H/2 x W/2")}
  ${block(1165, 255, 295, 145, "dec", "UpConv + concat", "Upsample with skip A", "H x W")}
  ${block(1560, 255, 240, 145, "out", "1x1 Conv", "Predict target field", "B x C_out x H x W")}

  ${arrow(340, 327, 424, 327)}
  ${arrow(700, 327, 779, 327)}
  ${arrow(927, 420, 927, 568)}
  <text class="flow-label" x="954" y="492">downsample</text>
  ${arrow(1070, 657, 1158, 657)}
  ${arrow(1312, 575, 1312, 410)}
  <text class="flow-label" x="1340" y="492">upsample</text>
  ${arrow(1460, 327, 1553, 327)}

  ${tag("skip A", 610, 214)}
  ${concat("+A", 1152, 327)}
  ${tag("skip B", 965, 214)}
  ${concat("+B", 1152, 657)}

  <text class="note" x="95" y="865">Correctness check: decoder first concatenates the H/2 encoder feature, then the H encoder feature, matching model_cnn.py.</text>
  </svg>`;
}

function transformerSvg() {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="1250" viewBox="0 0 2400 1250">
  ${defs()}
  <rect class="bg" width="2400" height="1250"/>
  <text class="title" x="80" y="78">Model 2: U-Net with Transformer Bottleneck</text>
  <text class="subtitle" x="80" y="118">Three encoder downsamples, transformer at H/8, then three decoder upsamples.</text>
  ${legend(1910, 48)}
  <line x1="80" y1="150" x2="2320" y2="150" stroke="#d0d7de" stroke-width="1.5"/>

  <text class="section" x="190" y="198">INPUT</text>
  <text class="section" x="530" y="198">ENCODER</text>
  <text class="section" x="1110" y="198">BOTTLENECK</text>
  <text class="section" x="1785" y="198">DECODER</text>
  <text class="section" x="2190" y="198">OUTPUT</text>

  ${block(95, 250, 245, 140, "input", "Input tensor", "Variable features", "B x C_in x H x W")}
  ${block(430, 250, 255, 140, "enc", "DoubleConv", "Feature extraction", "H x W")}
  ${block(770, 270, 305, 165, "enc", "MaxPool + DoubleConv", "Downsample features", "H/2 x W/2")}
  ${block(770, 520, 305, 165, "enc", "MaxPool + DoubleConv", "Deeper features", "H/4 x W/4")}
  ${block(770, 770, 305, 165, "bridge", "MaxPool + DoubleConv", "Lowest resolution", "H/8 x W/8")}

  ${block(1150, 880, 250, 145, "bot", "1x1 projection", "To embeddings", "H/8 x W/8 feature map")}
  ${block(1450, 880, 260, 145, "bot", "Transformer", "2D position + attention", "flattened spatial tokens")}
  ${block(1760, 880, 250, 145, "bot", "1x1 projection", "Back to features", "H/8 x W/8")}

  ${block(1760, 660, 310, 145, "dec", "UpConv + concat", "Upsample with skip C", "H/4 x W/4")}
  ${block(1760, 450, 310, 145, "dec", "UpConv + concat", "Upsample with skip B", "H/2 x W/2")}
  ${block(1760, 240, 310, 145, "dec", "UpConv + concat", "Upsample with skip A", "H x W")}
  ${block(2160, 240, 190, 145, "out", "1x1 Conv", "Target field", "C_out")}

  ${arrow(340, 320, 424, 320)}
  ${arrow(685, 320, 764, 320)}
  ${arrow(922, 435, 922, 512)}
  <text class="flow-label" x="952" y="482">downsample</text>
  ${arrow(922, 685, 922, 762)}
  <text class="flow-label" x="952" y="732">downsample</text>

  ${arrow(1075, 952, 1144, 952)}
  ${arrow(1400, 952, 1444, 952)}
  ${arrow(1710, 952, 1754, 952)}
  ${arrow(1885, 880, 1885, 813)}
  ${arrow(1885, 660, 1885, 603)}
  ${arrow(1885, 450, 1885, 393)}
  <text class="flow-label" x="1915" y="845">upsample</text>
  <text class="flow-label" x="1915" y="635">upsample</text>
  <text class="flow-label" x="1915" y="425">upsample</text>
  ${arrow(2070, 312, 2154, 312)}

  ${tag("skip A", 610, 230)}
  ${tag("skip B", 985, 230)}
  ${tag("skip C", 1100, 550)}
  ${concat("+C", 1745, 732)}
  ${concat("+B", 1745, 522)}
  ${concat("+A", 1745, 312)}

  <text class="note" x="95" y="1185">Correctness check: after transformer/post-projection at H/8, decoder uses skip C at H/4, skip B at H/2, then skip A at H, matching model_transformer.py.</text>
  </svg>`;
}

const files = [
  ["model_schematic_unet.svg", unetSvg()],
  ["model_schematic_transformer.svg", transformerSvg()],
  [
    "model_schematic_unet_poster_ready.svg",
    unetSvg()
      .replace(
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1000" viewBox="0 0 1800 1000">',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1780" height="640" viewBox="60 170 1780 640">'
      )
      .replace('<rect class="bg" width="1800" height="1000"/>', '<rect x="60" y="170" width="1780" height="640" fill="#ffffff"/>'),
  ],
  [
    "model_schematic_transformer_poster_ready.svg",
    transformerSvg()
      .replace(
        '<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="1250" viewBox="0 0 2400 1250">',
        '<svg xmlns="http://www.w3.org/2000/svg" width="2320" height="930" viewBox="60 170 2320 930">'
      )
      .replace('<rect class="bg" width="2400" height="1250"/>', '<rect x="60" y="170" width="2320" height="930" fill="#ffffff"/>'),
  ],
];

for (const [name, svg] of files) {
  const path = `CSCI1470-final-project/figures/${name}`;
  writeFileSync(path, svg);
  await sharp(Buffer.from(svg)).png().toFile(path.replace(/\.svg$/, ".png"));
}
