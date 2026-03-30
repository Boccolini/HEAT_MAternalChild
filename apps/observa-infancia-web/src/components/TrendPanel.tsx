import { useRef } from "react";
import ReactECharts from "echarts-for-react";
import type { SeriesPoint } from "../lib/api";
import { PanelActionButton } from "./PanelActionButton";
import { PanelDownloadActions } from "./PanelDownloadActions";
import { buildDownloadFilename, downloadChartImage, downloadCsv, type DownloadMetadata } from "../lib/download";

type TrendPanelProps = {
  title: string;
  points?: SeriesPoint[];
  seriesCollections?: Array<{
    label: string;
    color: string;
    points: SeriesPoint[];
  }>;
  valueUnit?: string;
  note?: string;
  downloadMetadata?: DownloadMetadata;
};

function formatNumericValue(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "NA";
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: value < 1 ? 3 : 2 });
}

export function TrendPanel({ title, points = [], seriesCollections, valueUnit, note, downloadMetadata }: TrendPanelProps) {
  const chartRef = useRef<ReactECharts>(null);
  const collections =
    seriesCollections && seriesCollections.length > 0
      ? seriesCollections
      : [{ label: title, color: "#12324A", points }];
  const xAxisValues = collections[0]?.points.map((point) => point.x) ?? [];
  const option = {
    tooltip: {
      trigger: "axis",
      formatter: (items: Array<{ axisValueLabel: string; value: number | null; seriesName: string; color: string }>) => {
        if (!items || items.length === 0) {
          return "";
        }
        const header = `<strong>${items[0].axisValueLabel}</strong>`;
        const rows = items.map((item) => `${item.seriesName}: ${formatNumericValue(item.value)}${valueUnit ? ` ${valueUnit}` : ""}`);
        return [
          header,
          ...rows,
        ].join("<br/>");
      },
    },
    grid: { left: 40, right: 20, top: 30, bottom: 40 },
    xAxis: {
      type: "category",
      data: xAxisValues,
      axisLabel: { color: "#365C72" }
    },
    yAxis: {
      type: "value",
      name: valueUnit,
      nameLocation: "middle",
      nameGap: 56,
      axisLabel: { color: "#365C72" },
      splitLine: { lineStyle: { color: "#D8DEE3" } }
    },
    legend:
      collections.length > 1
        ? {
            top: 0,
            textStyle: { color: "#365C72" },
          }
        : undefined,
    series: [
      ...collections.map((collection, index) => ({
        name: collection.label,
        type: "line",
        data: collection.points.map((point) => point.y),
        smooth: true,
        lineStyle: { color: collection.color, width: 3 },
        itemStyle: { color: collection.color },
        areaStyle: collections.length === 1 && index === 0 ? { color: "rgba(18, 50, 74, 0.08)" } : undefined,
      })),
    ]
  };

  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <h3 className="text-base font-semibold">{title}</h3>
          {valueUnit ? <p className="mt-1 text-xs text-slate-500">Unit: {valueUnit}</p> : null}
        </div>
        <PanelDownloadActions>
          <PanelActionButton
            label="PNG"
            onClick={() => {
              const chart = chartRef.current?.getEchartsInstance();
              if (!chart) {
                return;
              }
              void downloadChartImage(chart.getDataURL({ pixelRatio: 2, backgroundColor: "#ffffff" }), buildDownloadFilename(title, "png"), downloadMetadata);
            }}
          />
          <PanelActionButton
            label="CSV"
            onClick={() =>
              downloadCsv(
                collections.flatMap((collection) =>
                  collection.points.map((point) => ({
                    series: collection.label,
                    time_bucket: point.x,
                    value: point.y,
                    denominator: point.denominator,
                    quality_flag_any: point.quality_flag_any,
                  })),
                ),
                buildDownloadFilename(title, "csv"),
                downloadMetadata,
              )
            }
          />
        </PanelDownloadActions>
      </div>
      <div className="panel-body">
        <ReactECharts ref={chartRef} option={option} style={{ height: 320 }} />
        {note ? <p className="mt-3 text-xs text-slate-500">{note}</p> : null}
      </div>
    </section>
  );
}
