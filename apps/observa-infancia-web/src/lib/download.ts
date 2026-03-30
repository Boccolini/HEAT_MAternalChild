const FILE_PREFIX = "observa-infancia";
const DOWNLOAD_BACKGROUND = "#ffffff";
const DOWNLOAD_TEXT = "#12324a";
import { DASHBOARD_DOI, DASHBOARD_SUGGESTED_CITATION } from "./dashboardRelease";

export type DownloadMetadata = {
  title: string;
  notes?: string[];
};

function timestampLabel() {
  const now = new Date();
  const parts = [
    now.getFullYear().toString(),
    String(now.getMonth() + 1).padStart(2, "0"),
    String(now.getDate()).padStart(2, "0"),
    String(now.getHours()).padStart(2, "0"),
    String(now.getMinutes()).padStart(2, "0"),
    String(now.getSeconds()).padStart(2, "0"),
  ];
  return `${parts[0]}${parts[1]}${parts[2]}-${parts[3]}${parts[4]}${parts[5]}`;
}

export function buildDownloadFilename(baseName: string, extension: string) {
  const slug = baseName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
  return `${FILE_PREFIX}-${slug || "download"}-${timestampLabel()}.${extension}`;
}

function triggerDownload(href: string, filename: string) {
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = filename;
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
}

export function downloadDataUrl(dataUrl: string, filename: string) {
  triggerDownload(dataUrl, filename);
}

export function downloadBlob(blob: Blob, filename: string) {
  const objectUrl = URL.createObjectURL(blob);
  triggerDownload(objectUrl, filename);
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
}

function csvEscape(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  const text = String(value);
  if (text.includes(",") || text.includes("\"") || text.includes("\n")) {
    return `"${text.replace(/"/g, "\"\"")}"`;
  }
  return text;
}

function normaliseMetadata(metadata?: DownloadMetadata) {
  const citationNotes = [
    `Suggested citation: ${DASHBOARD_SUGGESTED_CITATION}`,
    `DOI: ${DASHBOARD_DOI}`,
  ];
  if (!metadata) {
    return { title: "", notes: citationNotes };
  }
  return {
    title: metadata.title.trim(),
    notes: [...(metadata.notes ?? []), ...citationNotes].map((note) => note.trim()).filter(Boolean),
  };
}

function wrapText(text: string, maxCharacters: number) {
  if (!text) {
    return [];
  }
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let current = "";
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length <= maxCharacters || !current) {
      current = candidate;
      continue;
    }
    lines.push(current);
    current = word;
  }
  if (current) {
    lines.push(current);
  }
  return lines;
}

function metadataLines(metadata?: DownloadMetadata) {
  const normalised = normaliseMetadata(metadata);
  const lines: string[] = [];
  if (normalised.title) {
    lines.push(`Title: ${normalised.title}`);
  }
  if (normalised.notes.length > 0) {
    lines.push("Notes:");
    normalised.notes.forEach((note) => lines.push(`- ${note}`));
  }
  return lines;
}

export function rowsToCsvContent(rows: Array<Record<string, unknown>>, metadata?: DownloadMetadata) {
  const meta = metadataLines(metadata);
  if (rows.length === 0) {
    return `${meta.map((line) => `# ${line}`).join("\n")}${meta.length ? "\n\n" : ""}No data available\n`;
  }

  const headers = Array.from(
    rows.reduce((set, row) => {
      Object.keys(row).forEach((key) => set.add(key));
      return set;
    }, new Set<string>()),
  );
  const content = [
    headers.join(","),
    ...rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")),
  ].join("\n");
  const prefix = meta.map((line) => `# ${line}`).join("\n");
  return `${prefix}${prefix ? "\n\n" : ""}${content}\n`;
}

export function downloadCsv(rows: Array<Record<string, unknown>>, filename: string, metadata?: DownloadMetadata) {
  const blob = new Blob([`\uFEFF${rowsToCsvContent(rows, metadata)}`], { type: "text/csv;charset=utf-8" });
  downloadBlob(blob, filename);
}

function escapeXml(value: string) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function svgNumericAttribute(svgElement: SVGSVGElement, attribute: "width" | "height") {
  const raw = svgElement.getAttribute(attribute);
  if (!raw) {
    return null;
  }
  const numeric = Number.parseFloat(raw);
  return Number.isFinite(numeric) ? numeric : null;
}

async function loadImage(dataUrl: string) {
  return await new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Unable to load generated chart image."));
    image.src = dataUrl;
  });
}

export async function downloadChartImage(dataUrl: string, filename: string, metadata?: DownloadMetadata) {
  const image = await loadImage(dataUrl);
  const normalised = normaliseMetadata(metadata);
  const margin = 48;
  const headingFont = "600 28px ui-sans-serif, system-ui, sans-serif";
  const noteFont = "16px ui-sans-serif, system-ui, sans-serif";
  const lineHeight = 24;
  const titleLines = normalised.title ? wrapText(normalised.title, 64) : [];
  const noteLines = normalised.notes.flatMap((note) => wrapText(note, 88));

  const topHeight = margin + titleLines.length * 34 + 16;
  const bottomHeight = noteLines.length > 0 ? 28 + noteLines.length * lineHeight + margin : margin;
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = topHeight + image.height + bottomHeight;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Unable to create canvas context for chart download.");
  }

  context.fillStyle = DOWNLOAD_BACKGROUND;
  context.fillRect(0, 0, canvas.width, canvas.height);

  context.fillStyle = DOWNLOAD_TEXT;
  context.font = headingFont;
  titleLines.forEach((line, index) => {
    context.fillText(line, margin, margin + 28 + index * 34);
  });

  context.drawImage(image, 0, topHeight, image.width, image.height);

  if (noteLines.length > 0) {
    context.font = "600 18px ui-sans-serif, system-ui, sans-serif";
    context.fillText("Notes", margin, topHeight + image.height + 28);
    context.font = noteFont;
    noteLines.forEach((line, index) => {
      context.fillText(line, margin, topHeight + image.height + 58 + index * lineHeight);
    });
  }

  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) {
    throw new Error("Unable to generate chart PNG download.");
  }
  downloadBlob(blob, filename);
}

export function downloadSvg(svgElement: SVGSVGElement, filename: string, metadata?: DownloadMetadata) {
  const normalised = normaliseMetadata(metadata);
  const viewBox = svgElement.viewBox.baseVal;
  const width =
    (viewBox && viewBox.width ? viewBox.width : null) ??
    svgNumericAttribute(svgElement, "width") ??
    Math.ceil(svgElement.getBoundingClientRect().width) ??
    1200;
  const height =
    (viewBox && viewBox.height ? viewBox.height : null) ??
    svgNumericAttribute(svgElement, "height") ??
    Math.ceil(svgElement.getBoundingClientRect().height) ??
    800;
  const titleLines = normalised.title ? wrapText(normalised.title, 72) : [];
  const noteLines = normalised.notes.flatMap((note) => wrapText(note, 96));
  const topHeight = 48 + titleLines.length * 30 + 16;
  const bottomHeight = noteLines.length > 0 ? 36 + noteLines.length * 22 + 28 : 36;
  const serializer = new XMLSerializer();
  const markup = serializer.serializeToString(svgElement);
  const notesBlock = noteLines.length
    ? `<text x="48" y="${topHeight + height + 28}" font-family="ui-sans-serif, system-ui, sans-serif" font-size="18" font-weight="600" fill="${DOWNLOAD_TEXT}">Notes</text>
${noteLines
  .map(
    (line, index) =>
      `<text x="48" y="${topHeight + height + 58 + index * 22}" font-family="ui-sans-serif, system-ui, sans-serif" font-size="16" fill="${DOWNLOAD_TEXT}">${escapeXml(line)}</text>`,
  )
  .join("\n")}`
    : "";
  const titleBlock = titleLines
    .map(
      (line, index) =>
        `<text x="48" y="${48 + 24 + index * 30}" font-family="ui-sans-serif, system-ui, sans-serif" font-size="26" font-weight="600" fill="${DOWNLOAD_TEXT}">${escapeXml(line)}</text>`,
    )
    .join("\n");
  const combinedMarkup = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${topHeight + height + bottomHeight}" viewBox="0 0 ${width} ${topHeight + height + bottomHeight}">
  <rect width="100%" height="100%" fill="${DOWNLOAD_BACKGROUND}"/>
  ${titleBlock}
  <g transform="translate(0 ${topHeight})">
    ${markup}
  </g>
  ${notesBlock}
</svg>`;
  const blob = new Blob([combinedMarkup], { type: "image/svg+xml;charset=utf-8" });
  downloadBlob(blob, filename);
}
